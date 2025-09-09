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
from clinic_cache import clinic_cache, customer_cache

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

class ConversationLogger:
    """Logs conversation sessions with structured metadata and timing."""
    def __init__(self, log_file="conversations.json"):
        self.log_file = log_file
        self.current_session = None
        
    def start_session(self, stream_sid):
        """Initialize new conversation session with metadata tracking."""
        self.current_session = {
            "session_id": str(uuid.uuid4()),
            "stream_sid": stream_sid,
            "caller_phone": caller_phone,
            "called_phone": called_phone,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "messages": [],
            "function_calls": [],
            "metadata": {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "function_calls": 0,
                "call_ended_by_user": False
            }
        }
        logger.info(f"Started conversation session: {self.current_session['session_id']}")
        
    def log_message(self, role, content, message_type="text"):
        """Record conversation message with role attribution."""
        if not self.current_session:
            return
            
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "type": message_type
        }
        
        self.current_session["messages"].append(message)
        self.current_session["metadata"]["total_messages"] += 1
        
        if role == "user":
            self.current_session["metadata"]["user_messages"] += 1
        elif role == "assistant":
            self.current_session["metadata"]["assistant_messages"] += 1
            
        logger.info(f"Conversation - {role}: {content[:100]}...")
        
    def log_function_call(self, func_name, arguments, result, execution_time=None):
        """Record function execution with timing and success status."""
        if not self.current_session:
            return
            
        function_call = {
            "timestamp": datetime.now().isoformat(),
            "function_name": func_name,
            "arguments": arguments,
            "result": result,
            "execution_time_ms": execution_time * 1000 if execution_time else None,
            "success": "error" not in result if isinstance(result, dict) else True
        }
        
        self.current_session["function_calls"].append(function_call)
        self.current_session["metadata"]["function_calls"] += 1
        
        logger.info(f"Function call: {func_name} - Success: {function_call['success']}")
        
    def log_call_end_request(self):
        """Mark session as user-initiated termination."""
        if self.current_session:
            self.current_session["metadata"]["call_ended_by_user"] = True
        
    def end_session(self):
        """Finalize session with duration calculation and save to JSON file."""
        if not self.current_session:
            return
            
        self.current_session["end_time"] = datetime.now().isoformat()
        
        # Calculate session duration
        start = datetime.fromisoformat(self.current_session["start_time"])
        end = datetime.fromisoformat(self.current_session["end_time"])
        duration_seconds = (end - start).total_seconds()
        self.current_session["metadata"]["duration_seconds"] = duration_seconds
        
        # Load existing conversations or create new list
        conversations = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    conversations = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                conversations = []
                
        # Add current session
        conversations.append(self.current_session)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(conversations, f, indent=2)
            
        logger.info(f"Conversation ended - Duration: {duration_seconds:.1f}s, Messages: {self.current_session['metadata']['total_messages']}, Functions: {self.current_session['metadata']['function_calls']}")
        
        self.current_session = None

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

def load_config():
    """Load Deepgram agent configuration from JSON file."""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raise Exception("config.json not found or invalid")

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
        
        # Clear Twilio audio buffer to interrupt agent speech
        clear_message = {
            "event": "clear",
            "streamSid": streamsid
        }
        await twilio_ws.send(json.dumps(clear_message))
        
        # Second clear for reliability
        await asyncio.sleep(0.1)
        await twilio_ws.send(json.dumps(clear_message))
        
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
    
    # Record conversation text for session logging
    if message_type == "ConversationText":
        role = decoded.get("role", "unknown")
        content = decoded.get("content", "")
        conversation_logger.log_message(role, content)
        
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
    
    elif message_type == "AgentStartedSpeaking":
        logger.info("Agent started speaking")
        
    elif message_type == "AgentAudioDone":
        logger.info("Agent finished speaking")
        
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
        await handle_function_call_request(decoded, sts_ws)

async def sts_sender(sts_ws, audio_queue):
    """Forward audio chunks from Twilio to Deepgram with monitoring."""
    logger.info("STS sender started")
    chunks_sent = 0
    
    try:
        while True:
            chunk = await audio_queue.get()
            await sts_ws.send(chunk)
            chunks_sent += 1
            
            if chunks_sent == 1:
                logger.info("First audio chunk sent to Deepgram")
                
    except Exception as e:
        logger.error(f"Error in STS sender: {e}")
    finally:
        logger.info(f"STS sender stopped - Total chunks sent: {chunks_sent}")

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
            media_message = {
                "event": "media",
                "streamSid": streamsid,
                "media": {"payload": base64.b64encode(raw_mulaw).decode("ascii")}
            }
            await twilio_ws.send(json.dumps(media_message))
            
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Deepgram connection closed: code={e.code}, reason={e.reason}")
    except Exception as e:
        logger.error(f"Error in STS receiver: {e}")
    finally:
        logger.info("STS receiver stopped")

caller_phone = None
called_phone = None

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


async def twilio_receiver(twilio_ws, audio_queue, streamsid_queue):
    """Process Twilio WebSocket events and forward audio to Deepgram queue."""
    global caller_phone, called_phone
    BUFFER_SIZE = 20 * 160
    inbuffer = bytearray(b"")
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
                    
                    # Load clinic data into cache for function calls
                    if called_phone:
                        try:
                            logger.info(f"Initializing clinic cache for {called_phone}")
                            clinic_data = await clinic_cache.load_clinic_data(called_phone)
                            logger.info(f"Clinic cache loaded: {clinic_data['clinic_name']} (ID: {clinic_data['clinic_id']})")
                        except Exception as e:
                            logger.error(f"CRITICAL: Failed to load clinic cache for {called_phone}: {e}")
                            # Continue without clinic data - functions will fail gracefully
                    else:
                        logger.warning("No called_phone available - clinic cache not initialized")
                    
                    # Initialize conversation logging session
                    conversation_logger.start_session(streamsid)
                    # Store phone numbers in session metadata
                    if conversation_logger.current_session:
                        conversation_logger.current_session["caller_phone"] = caller_phone
                        conversation_logger.current_session["called_phone"] = called_phone
                        conversation_logger.current_session["agent_id"] = called_phone  # Use called number as agent ID
                    
                    streamsid_queue.put_nowait(streamsid)
                    logger.info(f"Call started - StreamSID: {streamsid}")
                    
                elif event == "connected":
                    logger.info("Twilio connection established")
                    
                elif event == "media":
                    # Buffer inbound audio for processing
                    media = data["media"]
                    chunk = base64.b64decode(media["payload"])
                    if media["track"] == "inbound":
                        inbuffer.extend(chunk)
                        
                elif event == "stop":
                    logger.info("Call ended")
                    conversation_logger.end_session()
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
        logger.info("Twilio receiver stopped")


async def twilio_handler(twilio_ws):
    """Main handler that bridges Twilio WebSocket with Deepgram STS connection."""
    global call_should_end
    call_should_end = False  # Reset for each call
    
    audio_queue = asyncio.Queue()
    streamsid_queue = asyncio.Queue()
    
    logger.info("New Twilio connection established")
    
    try:
        async with sts_connect() as sts_ws:
            # Initialize Deepgram agent with config
            config_message = load_config()
            await sts_ws.send(json.dumps(config_message))
            logger.info("Configuration sent to Deepgram")
            
            # Start bidirectional audio/message processing
            tasks = [
                asyncio.create_task(sts_sender(sts_ws, audio_queue)),
                asyncio.create_task(sts_receiver(sts_ws, twilio_ws, streamsid_queue)),
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
    finally:
        try:
            await twilio_ws.close()
        except Exception as e:
            logger.error(f"Error closing Twilio websocket: {e}")
        logger.info("Twilio handler completed")

async def main():
    """Start WebSocket server to receive Twilio connections."""
    logger.info("Starting Voice Agent Server")
    logger.info("=" * 50)
    logger.info("Features enabled:")
    logger.info("- Structured conversation logging")
    logger.info("- Error handling")  
    logger.info("- Function execution timing")
    logger.info("- Connection monitoring")
    logger.info("- Improved barge-in handling")
    logger.info("- Concise responses (50-80 chars)")
    logger.info("- Call termination support")
    logger.info("=" * 50)
    
    try:
        await websockets.serve(twilio_handler, "localhost", 5000)
        logger.info("Server started on localhost:5000")
        logger.info("Conversation logs will be saved to: conversations.json")
        logger.info("System logs will be saved to: voice_agent.log")
        await asyncio.Future()  # Run forever
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())