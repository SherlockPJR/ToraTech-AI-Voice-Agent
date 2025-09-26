import asyncio
import base64
import json
import websockets
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
import uuid
from functions_google_calendar import FUNCTION_MAP
from functions_faq import get_faq_answer
from clinic_cache import clinic_cache, customer_cache

# Add FAQ function to FUNCTION_MAP
FUNCTION_MAP["get_faq_answer"] = get_faq_answer

from twilio.rest import Client
from supabase import create_client, Client
from typing import Optional, Dict, Tuple

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('voice_agent.log')
    ]
)
logger = logging.getLogger(__name__)

from loggers import ConversationLogger, PerformanceLogger

def init_supabase() -> Client:
    """Initialize Supabase client with environment credentials."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url:
        raise Exception("SUPABASE_URL not found.")
    
    if not supabase_key:
        raise Exception("SUPABASE_URL not found.")
    
    return create_client(supabase_url, supabase_key)

conversation_logger = ConversationLogger()
performance_logger = PerformanceLogger()


def load_dynamic_config(caller_phone: str, called_phone: str) -> dict:
    """Load dynamic configuration with clinic-specific variables injected."""
    try:
        # Load template configuration
        with open("config_template.json", "r") as f:
            config_template = json.load(f)
        
        # Get clinic data from cache, or use defaults if not found
        clinic_data = clinic_cache.get_cached_data(called_phone)
        
        if not clinic_data:
            logger.warning(f"No clinic data found for {called_phone}, using default values")
            # Use default values when clinic data is not available
            clinic_name = 'Our Clinic'
            clinic_location = 'Your Area'
            clinic_phone = 'Our Main Number'
            business_name = 'Business Name'
        else:
            # Extract clinic information from cache
            clinic_name = clinic_data.get('clinic_name', 'Our Clinic')
            clinic_address = clinic_data.get('address', 'Your Area')
            clinic_phone = clinic_data.get('phone_number', 'Our Main Number')
            business_name = clinic_data.get('business_name') or clinic_name
            
            # Parse location from address (extract city, state if possible)
            clinic_location = clinic_address
            if clinic_address and clinic_address != 'Your Area':
                # Try to extract meaningful location (city, state) from full address
                address_parts = clinic_address.split(',')
                if len(address_parts) >= 2:
                    # Assume format like "123 Main St, Gadsden, AL 35901"
                    # Take the last 2 parts as "City, State"
                    clinic_location = ', '.join(address_parts[-2:]).strip()
        
        # Handle prompt format conversion (array to string if needed)
        if isinstance(config_template["agent"]["think"]["prompt"], list):
            # Convert array format to single string by joining with newlines
            config_template["agent"]["think"]["prompt"] = "\n".join(config_template["agent"]["think"]["prompt"])
        
        # Convert config to JSON string for variable substitution
        config_str = json.dumps(config_template, indent=2)
        
        # Perform variable substitution
        config_str = config_str.replace('{CLINIC_NAME}', clinic_name)
        config_str = config_str.replace('{CLINIC_LOCATION}', clinic_location)
        config_str = config_str.replace('{CLINIC_PHONE}', clinic_phone)
        config_str = config_str.replace('{CALLER_PHONE}', caller_phone or 'the caller')
        config_str = config_str.replace('{BUSINESS_NAME}', business_name)
        
        # Convert back to dictionary
        dynamic_config = json.loads(config_str)
        
        logger.info(f"Generated dynamic config for {business_name} in {clinic_location}")
        return dynamic_config
        
    except FileNotFoundError:
        logger.error("config_template.json not found. This file is required for the system to function.")
        raise Exception("config_template.json is missing. Please ensure this file exists in the project root.")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config_template.json: {e}")
        raise Exception(f"config_template.json contains invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in dynamic config loading: {e}")
        raise Exception(f"Failed to load dynamic configuration: {e}")

def sts_connect():
    """Establish WebSocket connection to Deepgram STS with authentication."""

    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        raise Exception("DEEPGRAM_API_KEY not found")
    
    logger.info("Connecting to Deepgram STS...")
    sts_ws = websockets.connect(
        "wss://agent.deepgram.com/v1/agent/converse",
        subprotocols=["token", deepgram_api_key],
        ping_interval=20,
        ping_timeout=10
    )
    return sts_ws

async def handle_barge_in(decoded, twilio_ws, streamsid):
    """Handle user interruption by clearing Twilio audio buffer."""
    if decoded["type"] == "UserStartedSpeaking":
        conversation_logger.log_message("system", "User started speaking (barge-in)", "barge_in")
        
        # Removed raw audio timing event logging; using per-turn timestamps only
        # Start transport windows to measure inbound decode and uplink send during this utterance
        try:
            performance_logger.begin_inbound_window()
            performance_logger.begin_uplink_window()
        except Exception:
            pass
        # Do not begin a turn yet; we create a turn at user_speech_end
        
        # Start timing interruption response
        performance_logger.start_timing("interruption_response")
        
        # Clear Twilio audio buffer to interrupt agent speech
        clear_message = {
            "event": "clear",
            "streamSid": streamsid
        }
        await twilio_ws.send(json.dumps(clear_message))
        
        # Second clear for reliability
        await asyncio.sleep(0.1)
        await twilio_ws.send(json.dumps(clear_message))
        
        # End timing after clear messages sent
        performance_logger.end_timing("interruption_response")
        
        logger.info("Sent clear message for barge-in")

async def end_call_function(arguments):
    """Process agent-initiated call termination request."""
    conversation_logger.log_call_end_request()
    return {
        "status": "call_ending",
        "message": "Call will be terminated after farewell"
    }

async def execute_function_call(func_name, arguments):
    """Execute calendar function with timing, error handling, and clinic data injection."""
    global caller_phone, called_phone
    start_time = time.time()
    
    try:
        if func_name == "end_call":
            # Process call termination
            result = await end_call_function(arguments)
            execution_time = time.time() - start_time
            conversation_logger.log_function_call(func_name, arguments, result, execution_time)
            
            # Record performance metric for end_call function
            performance_logger.log_metric("function_execution", execution_time * 1000, "ms", {"function_name": func_name})
            
            return result
        elif func_name in FUNCTION_MAP:
            logger.info(f"Executing function: {func_name} with args: {arguments}")

            # Inject phone numbers and day-specific clinic data
            if called_phone:
                arguments["called_phone"] = called_phone
                
                # Extract target day from time input for optimized data passing
                time_input = arguments.get("time_input") or arguments.get("date_input") or "today"
                day_specific_data = clinic_cache.get_day_specific_data(called_phone, time_input)
                
                if day_specific_data:
                    arguments["_clinic_data"] = day_specific_data
                    target_day = day_specific_data['target_day_of_week']
                    clinic_name = day_specific_data['clinic_name']
                    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    day_name = weekday_names[target_day]
                    logger.debug(f"Passing {day_name} data for {clinic_name} to function {func_name} (day {target_day})")
                else:
                    # Use full clinic data when day parsing fails
                    clinic_data = clinic_cache.get_cached_data(called_phone)
                    if clinic_data:
                        arguments["_clinic_data"] = clinic_data
                        logger.warning(f"Could not extract day from '{time_input}', passing full clinic data to {func_name}")
                    else:
                        logger.error(f"CRITICAL: No clinic data cached for {called_phone} - function {func_name} will fail")
                    
            if caller_phone:
                arguments["caller_phone"] = caller_phone

            result = await FUNCTION_MAP[func_name](arguments)
            execution_time = time.time() - start_time
            
            logger.info(f"Function '{func_name}' completed in {execution_time:.3f}s")
            conversation_logger.log_function_call(func_name, arguments, result, execution_time)
            
            # Record performance metric for function execution
            performance_logger.log_metric("function_execution", execution_time * 1000, "ms", {"function_name": func_name})
            # Link this function to the current turn for additive breakdowns
            try:
                performance_logger.add_turn_function(
                    func_name,
                    execution_time * 1000,
                    start_iso=datetime.fromtimestamp(start_time).isoformat(),
                    end_iso=datetime.now().isoformat()
                )
            except Exception:
                pass
            
            return result
        else:
            result = {"error": f"Unknown function: {func_name}"}
            execution_time = time.time() - start_time
            
            logger.error(f"Unknown function: {func_name}")
            conversation_logger.log_function_call(func_name, arguments, result, execution_time)
            return result
            
    except Exception as e:
        execution_time = time.time() - start_time
        result = {"error": f"Function execution failed: {str(e)}"}
        
        logger.error(f"Function '{func_name}' failed after {execution_time:.3f}s: {e}")
        conversation_logger.log_function_call(func_name, arguments, result, execution_time)
        return result

def create_function_call_response(func_id, func_name, result):
    """Build Deepgram-compatible function call response message."""
    return {
        "type": "FunctionCallResponse",
        "id": func_id,
        "name": func_name,
        "content": json.dumps(result)
    }

call_should_end = False

async def handle_function_call_request(decoded, sts_ws):
    """Process function call requests from Deepgram agent with error handling."""
    global call_should_end
    
    try:
        for function_call in decoded["functions"]:
            func_name = function_call["name"]
            func_id = function_call["id"]
            arguments = json.loads(function_call["arguments"])
            
            result = await execute_function_call(func_name, arguments)
            function_result = create_function_call_response(func_id, func_name, result)
            
            await sts_ws.send(json.dumps(function_result))
            # Mark the time when the last tool result was sent for this turn, if a turn is active
            if getattr(performance_logger, "current_turn", None):
                performance_logger.note_tool_result_sent()
            logger.info(f"Sent function result for {func_name}")
            
            # Set termination flag if end_call was executed
            if func_name == "end_call" and result.get("status") == "call_ending":
                call_should_end = True
                logger.info("Call termination requested by user")
            
    except Exception as e:
        logger.error(f"Error in function call handling: {e}")
        # Send error response to maintain conversation flow
        error_result = create_function_call_response(
            func_id if "func_id" in locals() else "unknown",
            func_name if "func_name" in locals() else "unknown",
            {"error": f"Function call failed with: {str(e)}"}
        )
        try:
            await sts_ws.send(json.dumps(error_result))
        except Exception as send_error:
            logger.error(f"Failed to send error response: {send_error}")

async def handle_text_message(decoded, twilio_ws, sts_ws, streamsid):
    """Process JSON messages from Deepgram with logging and call termination detection."""
    global call_should_end
    
    message_type = decoded.get("type")
    
    # Debug: Log important message types for troubleshooting
    if message_type not in ["ConversationText", "History"]:  # Avoid spam from frequent messages
        logger.info(f"Deepgram message type: {message_type}")
    
    # Handle History events (which contain assistant responses)
    if message_type == "History":
        role = decoded.get("role", "unknown")
        if role == "assistant":
            # This is when agent starts speaking - History events indicate agent response
            logger.info("Agent started speaking (inferred from History event)")
            
            # End speech-to-response latency
            # Only stamp assistant_history_ts if we have an active turn with a user_speech_end
            if getattr(performance_logger, "current_turn", None) and performance_logger.current_turn.get("timestamps", {}).get("user_speech_end_ts"):
                performance_logger.set_turn_timestamp("assistant_history_ts")
            performance_logger.end_timing("speech_to_response")
            
            # End call setup timing when agent first starts speaking
            performance_logger.end_timing("call_setup")
    
    # Record conversation text for session logging
    if message_type == "ConversationText":
        role = decoded.get("role", "unknown")
        content = decoded.get("content", "")
        conversation_logger.log_message(role, content)
        
        # Track when user finishes speaking (inferred from receiving user text)
        if role == "user":
            # End of user's utterance defines a new turn boundary
            performance_logger.begin_turn()
            performance_logger.set_turn_timestamp("user_speech_end_ts")
            # Attach any transport averages measured during this utterance to the new turn
            try:
                performance_logger.attach_transport_avgs_to_current_turn()
            except Exception:
                pass
            performance_logger.start_timing("speech_to_response")
        
        # Detect agent farewell phrases for call termination
        if role == "assistant":
            farewell_phrases = [
                "have a great day",
                "thanks for calling",
                "take care",
                "we'll see you",
                "goodbye",
                "talk to you soon",
                "have a good day"
            ]
            
            # Trigger termination on farewell detection
            if any(phrase in content.lower() for phrase in farewell_phrases):
                logger.info("Agent farewell detected - preparing to end call")
                call_should_end = True

        # Detect user termination requests
        if role == "user" and any(phrase in content.lower() for phrase in ["end call", "hang up", "goodbye", "end this call", "finish call"]):
            logger.info("User requested call termination via message")
    
    # Record system events and session state changes
    elif message_type == "Welcome":
        session_id = decoded.get("session_id", "unknown")
        logger.info(f"Deepgram session established: {session_id}")
        conversation_logger.log_message("system", f"Session established: {session_id}", "welcome")
    
    elif message_type == "AgentAudioDone":
        logger.info("Agent finished speaking")
        
        # Track agent audio end timing
        now_iso = datetime.now().isoformat()
        # Prefer attaching to the current turn if it has assistant_history; else attach to the last open turn
        if getattr(performance_logger, "current_turn", None) and performance_logger.current_turn.get("timestamps", {}).get("assistant_history_ts"):
            performance_logger.set_turn_timestamp("agent_audio_end_ts", now_iso)
            try:
                performance_logger.finalize_turn()
            except Exception:
                pass
        else:
            try:
                performance_logger.attach_agent_audio_end(now_iso)
            except Exception:
                pass
        
        # Terminate call after agent completes farewell
        if call_should_end:
            logger.info("Terminating call after agent finished farewell")
            await asyncio.sleep(1)  # Brief pause
            # Close connection to end call
            await sts_ws.close()
    
    # Process user interruption events
    await handle_barge_in(decoded, twilio_ws, streamsid)
    
    # Execute function calls from agent
    if message_type == "FunctionCallRequest":
        # Mark the model decision moment only when there is an active turn
        if getattr(performance_logger, "current_turn", None):
            performance_logger.set_turn_timestamp("llm_decision_ts")
        await handle_function_call_request(decoded, sts_ws)

async def sts_sender(sts_ws, audio_queue):
    """Forward audio chunks from Twilio to Deepgram with monitoring."""
    logger.info("STS sender started")
    chunks_sent = 0
    
    try:
        while True:
            chunk = await audio_queue.get()
            
            # Measure time to send to Deepgram (uplink)
            try:
                t0 = time.time()
                await sts_ws.send(chunk)
                dt_ms = (time.time() - t0) * 1000
                performance_logger.add_uplink_send_ms(dt_ms)
            except Exception:
                await sts_ws.send(chunk)
            chunks_sent += 1
            
            if chunks_sent == 1:
                logger.info("First audio chunk sent to Deepgram")
                
    except Exception as e:
        logger.error(f"Error in STS sender: {e}")
    finally:
        logger.info(f"STS sender stopped - Total chunks sent: {chunks_sent}")

async def sts_receiver_with_config(sts_ws, twilio_ws, streamsid_queue):
    """Receive audio/JSON from Deepgram with dynamic config initialization."""
    global caller_phone, called_phone, phone_numbers_ready
    logger.info("STS receiver started, waiting for phone numbers...")
    
    # Wait for phone numbers to be available
    await phone_numbers_ready.wait()
    logger.info("Phone numbers available, sending dynamic config")
    
    # Send dynamic configuration based on phone numbers
    try:
        config_message = load_dynamic_config(caller_phone, called_phone)
        await sts_ws.send(json.dumps(config_message))
        logger.info("Dynamic configuration sent to Deepgram")
    except Exception as e:
        logger.error(f"Failed to load or send dynamic config: {e}")
        logger.error("Cannot proceed without valid configuration. Terminating connection.")
        return
    
    # Now proceed with normal sts_receiver functionality
    streamsid = await streamsid_queue.get()
    
    try:
        async for message in sts_ws:
            if type(message) is str:
                
                # Process JSON events and function calls
                try:
                    decoded = json.loads(message)
                    await handle_text_message(decoded, twilio_ws, sts_ws, streamsid)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                continue
            
            # Forward audio to Twilio in media message format
            raw_mulaw = message
            first_frame = False
            if getattr(performance_logger, "current_turn", None) and performance_logger.current_turn.get("timestamps", {}).get("assistant_history_ts") and not performance_logger.current_turn.get("timestamps", {}).get("first_agent_audio_ts"):
                performance_logger.set_turn_timestamp("first_agent_audio_ts")
                first_frame = True
            # Measure base64 encode and Twilio send times for the first frame only
            if first_frame:
                t_enc0 = time.time()
                payload = base64.b64encode(raw_mulaw).decode("ascii")
                tts_encode_ms = (time.time() - t_enc0) * 1000
                performance_logger.set_turn_transport_metric("tts_encode_ms_first", tts_encode_ms)
            else:
                payload = base64.b64encode(raw_mulaw).decode("ascii")
            media_message = {"event": "media", "streamSid": streamsid, "media": {"payload": payload}}
            if first_frame:
                t_send0 = time.time()
                await twilio_ws.send(json.dumps(media_message))
                twilio_send_ms = (time.time() - t_send0) * 1000
                performance_logger.set_turn_transport_metric("twilio_send_ms_first", twilio_send_ms)
                performance_logger.set_turn_timestamp("first_twilio_send_ts")
            else:
                await twilio_ws.send(json.dumps(media_message))
            
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Deepgram connection closed: code={e.code}, reason={e.reason}")
    except Exception as e:
        logger.error(f"Error in STS receiver with config: {e}")
    finally:
        logger.info("STS receiver with config stopped")

async def sts_receiver(sts_ws, twilio_ws, streamsid_queue):
    """Receive audio/JSON from Deepgram and forward to Twilio or process events."""
    logger.info("STS receiver started")
    streamsid = await streamsid_queue.get()
    
    try:
        async for message in sts_ws:
            if type(message) is str:
                
                # Process JSON events and function calls
                try:
                    decoded = json.loads(message)
                    await handle_text_message(decoded, twilio_ws, sts_ws, streamsid)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                continue
            
            # Forward audio to Twilio in media message format
            raw_mulaw = message
            first_frame = False
            if getattr(performance_logger, "current_turn", None) and performance_logger.current_turn.get("timestamps", {}).get("assistant_history_ts") and not performance_logger.current_turn.get("timestamps", {}).get("first_agent_audio_ts"):
                performance_logger.set_turn_timestamp("first_agent_audio_ts")
                first_frame = True
            # Measure base64 encode and Twilio send times for the first frame only
            if first_frame:
                t_enc0 = time.time()
                payload = base64.b64encode(raw_mulaw).decode("ascii")
                tts_encode_ms = (time.time() - t_enc0) * 1000
                performance_logger.set_turn_transport_metric("tts_encode_ms_first", tts_encode_ms)
            else:
                payload = base64.b64encode(raw_mulaw).decode("ascii")
            media_message = {"event": "media", "streamSid": streamsid, "media": {"payload": payload}}
            if first_frame:
                t_send0 = time.time()
                await twilio_ws.send(json.dumps(media_message))
                twilio_send_ms = (time.time() - t_send0) * 1000
                performance_logger.set_turn_transport_metric("twilio_send_ms_first", twilio_send_ms)
                performance_logger.set_turn_timestamp("first_twilio_send_ts")
            else:
                await twilio_ws.send(json.dumps(media_message))
            
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Deepgram connection closed: code={e.code}, reason={e.reason}")
    except Exception as e:
        logger.error(f"Error in STS receiver: {e}")
    finally:
        logger.info("STS receiver stopped")

caller_phone = None
called_phone = None

# Asyncio event to signal when phone numbers are available
phone_numbers_ready = asyncio.Event()

async def get_phone_numbers_from_call_sid(call_sid: str) -> Tuple[Optional[str], Optional[str]]:
    """Retrieve caller/called phone numbers from Twilio API using call SID."""
    try:
        from twilio.rest import Client

        # Use main Twilio account credentials
        account_sid = os.getenv('TWILIO_MAIN_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_MAIN_ACCOUNT_AUTH_TOKEN')

        if not account_sid or not auth_token:
            logger.error('TWILIO_MAIN_ACCOUNT_SID and TWILIO_MAIN_ACCOUNT_AUTH_TOKEN required')
            return None, None

        # Fetch call details from Twilio API
        client = Client(account_sid, auth_token)
        call = client.calls(call_sid).fetch()

        # Extract formatted phone numbers from call object
        caller_phone = call.from_formatted
        called_phone = call.to

        logger.info(f'Retrieved from Twilio API: {caller_phone} -> {called_phone}')
        return caller_phone, called_phone

    except Exception as e:
        logger.error(f'Failed to get phone numbers from Twilio API: {e}')
        return None, None

async def start_call_recording(call_sid: str) -> Optional[str]:
    """Start dual-channel recording for a call using Twilio REST API."""
    try:
        from twilio.rest import Client

        # Use main Twilio account credentials
        account_sid = os.getenv('TWILIO_MAIN_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_MAIN_ACCOUNT_AUTH_TOKEN')

        if not account_sid or not auth_token:
            logger.error('TWILIO_MAIN_ACCOUNT_SID and TWILIO_MAIN_ACCOUNT_AUTH_TOKEN required for recording')
            return None

        client = Client(account_sid, auth_token)
        
        # Optional: Set up recording status callback URL if configured
        callback_url = os.getenv('RECORDING_STATUS_CALLBACK_URL')
        recording_params = {}
        
        if callback_url:
            recording_params['recording_status_callback'] = callback_url
            recording_params['recording_status_callback_event'] = ['in-progress', 'completed', 'failed']
            logger.info(f'Recording status callbacks will be sent to: {callback_url}')
        
        # Set recording channels to dual for separate inbound/outbound tracks
        recording_params['recording_channels'] = 'dual'

        # Start recording using the correct REST API endpoint
        # POST https://api.twilio.com/2010-04-01/Accounts/{AccountSid}/Calls/{CallsSid}/Recordings.json
        recording = client.calls(call_sid).recordings.create(**recording_params)
        
        logger.info(f'RECORDING STARTED - SID: {recording.sid}, Call: {call_sid}, Status: {recording.status}')
        return recording.sid

    except Exception as e:
        logger.error(f'Failed to start call recording: {e}')
        return None


async def twilio_receiver(twilio_ws, audio_queue, streamsid_queue):
    """Process Twilio WebSocket events and forward audio to Deepgram queue."""
    global caller_phone, called_phone
    BUFFER_SIZE = 20 * 160
    inbuffer = bytearray(b"")
    call_sid = None  # Track call SID for recording logging
    logger.info("Twilio receiver started")
    
    try:
        async for message in twilio_ws:
            try:
                data = json.loads(message)
                event = data["event"]
                
                if event == "start":
                    # Extract stream ID and initialize session
                    start_data = data["start"]
                    streamsid = start_data["streamSid"]
                    
                    # Retrieve call SID for phone number lookup
                    call_sid = start_data.get("callSid")
                    
                    if call_sid:
                        logger.info(f"Retrieving phone numbers for call: {call_sid}")
                        caller_phone, called_phone = await get_phone_numbers_from_call_sid(call_sid)
                    else:
                        logger.warning("No callSid in start data")
                        caller_phone, called_phone = None, None
                    
                    # Record caller and called numbers
                    logger.info(f"Call from {caller_phone} to {called_phone}")
                    
                    # Signal that phone numbers are now available
                    phone_numbers_ready.set()
                    
                    # Start call recording if call_sid is available
                    recording_sid = None
                    if call_sid and os.getenv("ENABLE_CALL_RECORDING", "false").lower() == "true":
                        try:
                            recording_sid = await start_call_recording(call_sid)
                        except Exception as e:
                            logger.error(f"Failed to start call recording: {e}")
                            # Continue without recording - this is not critical for call functionality
                    
                    # Load clinic data into cache for function calls
                    if called_phone:
                        try:
                            logger.info(f"Initializing clinic cache for {called_phone}")
                            clinic_data = await clinic_cache.load_clinic_data(called_phone)
                            logger.info(f"Clinic cache loaded: {clinic_data['clinic_name']} (ID: {clinic_data['clinic_id']})")
                            
                            # Load FAQ data into cache for this clinic
                            from functions_faq import faq_cache
                            clinic_id = clinic_data['clinic_id']
                            logger.info(f"Initializing FAQ cache for clinic_id: {clinic_id}")
                            faq_data = await faq_cache.load_clinic_faqs(clinic_id)
                            logger.info(f"FAQ cache loaded: {len(faq_data.get('faqs', []))} FAQs available")
                            
                        except Exception as e:
                            logger.error(f"CRITICAL: Failed to load clinic cache for {called_phone}: {e}")
                            # Continue without clinic data - functions will fail gracefully
                    else:
                        logger.warning("No called_phone available - clinic cache not initialized")
                    
                    # Initialize conversation logging session
                    conversation_logger.start_session(streamsid)
                    # Store phone numbers and recording info in session metadata
                    if conversation_logger.current_session:
                        conversation_logger.current_session["caller_phone"] = caller_phone
                        conversation_logger.current_session["called_phone"] = called_phone
                        conversation_logger.current_session["agent_id"] = called_phone  # Use called number as agent ID
                        conversation_logger.current_session["call_sid"] = call_sid
                        if recording_sid:
                            conversation_logger.current_session["recording_sid"] = recording_sid
                    
                    # Initialize performance tracking session
                    performance_logger.start_session(streamsid, caller_phone, called_phone)
                    # Start timing call setup
                    performance_logger.start_timing("call_setup")
                    
                    # Recording is managed via TwiML (<Start><Record/>) when enabled; no REST start here

                    streamsid_queue.put_nowait(streamsid)
                    logger.info(f"Call started - StreamSID: {streamsid}")
                    
                elif event == "connected":
                    logger.info("Twilio connection established")
                    
                elif event == "media":
                    # Buffer inbound audio for processing
                    media = data["media"]
                    # Measure base64 decode time for inbound audio
                    try:
                        t0 = time.time()
                        chunk = base64.b64decode(media["payload"])
                        dt_ms = (time.time() - t0) * 1000
                        performance_logger.add_inbound_decode_ms(dt_ms)
                    except Exception:
                        chunk = base64.b64decode(media["payload"])  # Fallback
                    if media["track"] == "inbound":
                        inbuffer.extend(chunk)
                        
                elif event == "stop":
                    logger.info("Call ended")
                    # Log recording end if recording was active
                    if conversation_logger.current_session and conversation_logger.current_session.get("recording_sid"):
                        recording_sid = conversation_logger.current_session["recording_sid"]
                        logger.info(f'RECORDING ENDED - SID: {recording_sid}, Call: {call_sid}')
                    conversation_logger.end_session()
                    performance_logger.end_session()
                    break
                
                # Forward audio chunks to Deepgram when buffer is full
                while len(inbuffer) >= BUFFER_SIZE:
                    chunk = inbuffer[:BUFFER_SIZE]
                    audio_queue.put_nowait(chunk)
                    inbuffer = inbuffer[BUFFER_SIZE:]
                    
            except json.JSONDecodeError as e:
                logger.error(f"Twilio JSON decode error: {e}")
            except Exception as e:
                logger.error(f"Error processing Twilio message: {e}")
                
    except Exception as e:
        logger.error(f"Error in Twilio receiver: {e}")
    finally:
        # Close conversation logging on exit
        conversation_logger.end_session()
        performance_logger.end_session()
        logger.info("Twilio receiver stopped")


async def twilio_handler(twilio_ws):
    """Main handler that bridges Twilio WebSocket with Deepgram STS connection."""
    global call_should_end, phone_numbers_ready
    call_should_end = False  # Reset for each call
    phone_numbers_ready.clear()  # Reset event for new call
    
    audio_queue = asyncio.Queue()
    streamsid_queue = asyncio.Queue()
    
    logger.info("New Twilio connection established")
    
    try:
        async with sts_connect() as sts_ws:
            # Start bidirectional audio/message processing with dynamic config
            tasks = [
                asyncio.create_task(sts_sender(sts_ws, audio_queue)),
                asyncio.create_task(sts_receiver_with_config(sts_ws, twilio_ws, streamsid_queue)),
                asyncio.create_task(twilio_receiver(twilio_ws, audio_queue, streamsid_queue)),
            ]
            
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                # Clean up remaining async tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                            
    except Exception as e:
        logger.error(f"Error in Twilio handler: {e}")
        conversation_logger.end_session()
        performance_logger.end_session()
    finally:
        try:
            await twilio_ws.close()
        except Exception as e:
            logger.error(f"Error closing Twilio websocket: {e}")
        logger.info("Twilio handler completed")

def validate_config_template():
    """Validate that config_template.json exists and is valid JSON."""
    try:
        with open("config_template.json", "r") as f:
            config_template = json.load(f)
        
        # Validate that required clinic placeholders exist
        config_str = json.dumps(config_template)
        required_placeholders = ['{CLINIC_NAME}', '{CLINIC_LOCATION}', '{CLINIC_PHONE}']
        
        missing_placeholders = []
        for placeholder in required_placeholders:
            if placeholder not in config_str:
                missing_placeholders.append(placeholder)
        
        if missing_placeholders:
            logger.warning(f"Missing required placeholders in config_template.json: {missing_placeholders}")
        
        # Optional placeholder check
        optional_placeholders = ['{CALLER_PHONE}']
        for placeholder in optional_placeholders:
            if placeholder not in config_str:
                logger.info(f"Optional placeholder {placeholder} not found in config_template.json")
        
        logger.info("config_template.json validation passed")
        return True
        
    except FileNotFoundError:
        logger.error("config_template.json not found. This file is required for the system to function.")
        logger.error("Please ensure config_template.json exists in the project root directory.")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config_template.json: {e}")
        logger.error("Please check the JSON syntax in config_template.json")
        return False
    except Exception as e:
        logger.error(f"Unexpected error validating config_template.json: {e}")
        return False

async def handle_health_check(path, request_headers):
    # If path is for health check, respond
    if path in ['/', '/health']:
        return (200, [("Content-Type", "text/plain")], b"OK")
    return None

async def main():
    """Start WebSocket server to receive Twilio connections."""
    logger.info("Starting Voice Agent Server")
    logger.info("=" * 50)
    
    # Validate configuration template before starting
    if not validate_config_template():
        logger.error("Configuration validation failed. Cannot start server.")
        return
    
    logger.info("Features enabled:")
    logger.info("- Structured conversation logging")
    logger.info("- Performance metrics tracking")
    logger.info("- Error handling")  
    logger.info("- Function execution timing")
    logger.info("- Connection monitoring")
    logger.info("- Improved barge-in handling")
    logger.info("- Concise responses (50-80 chars)")
    logger.info("- Call termination support")
    logger.info("- Dynamic clinic configuration (config_template.json)")
    logger.info("=" * 50)
    
    try:
        port = int(os.environ.get("PORT", "5000"))
        await websockets.serve(twilio_handler, "0.0.0.0", port, process_request=handle_health_check)
        logger.info("Server started on 0.0.0.0:{port}")
        logger.info("Conversation logs will be saved to: conversations.json")
        logger.info("Performance logs will be saved to: performance.json")
        logger.info("System logs will be saved to: voice_agent.log")
        logger.info(f"Performance logging: {'ENABLED' if performance_logger.enabled else 'DISABLED'}")
        await asyncio.Future()  # Run forever
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())