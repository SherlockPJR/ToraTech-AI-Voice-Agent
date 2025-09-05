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

from twilio.rest import Client
from supabase import create_client, Client
from typing import Optional, Dict, Tuple

load_dotenv()

logging.basicConfig(
    # Logging setup
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('voice_agent.log')
    ]
)
logger = logging.getLogger(__name__)

class ConversationLogger:
    # Structured conversation logging with timestamps
    def __init__(self, log_file="conversations.json"):
        self.log_file = log_file
        self.current_session = None
        
    def start_session(self, stream_sid):
        # Start a new conversation session
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
        # Log a conversation message
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
        # Log a function call
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
        # Log when user requests to end call
        if self.current_session:
            self.current_session["metadata"]["call_ended_by_user"] = True
        
    def end_session(self):
        # End the current session and save to file
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

# ---------------Supabase Initialisation---------------

def init_supabase() -> Client:
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url:
        raise Exception("SUPABASE_URL not found.")
    
    if not supabase_key:
        raise Exception("SUPABASE_URL not found.")
    
    return create_client(supabase_url, supabase_key)

# Global conversation logger instance
conversation_logger = ConversationLogger()

def load_config():
    # Load agent configuration file
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return minimal working config or re-raise
        raise Exception("config.json not found or invalid")

# --------------- Deepgram Connection ---------------

def sts_connect():
    # Connect to Deepgram STS websocket with retry logic

    # When we need to switch to OpenAI:
    # Realtime API with WebSocket: https://platform.openai.com/docs/guides/realtime-websocket?connection-example=python
    # Handling Audio with WebSockets: https://platform.openai.com/docs/guides/realtime-conversations#handling-audio-with-websockets

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
    # Barge-in handling with faster response
    if decoded["type"] == "UserStartedSpeaking":
        conversation_logger.log_message("system", "User started speaking (barge-in)", "barge_in")
        
        # Send clear message immediately and more aggressively
        clear_message = {
            "event": "clear",
            "streamSid": streamsid
        }
        await twilio_ws.send(json.dumps(clear_message))
        
        # Send a second clear to ensure interruption
        await asyncio.sleep(0.1)
        await twilio_ws.send(json.dumps(clear_message))
        
        logger.info("Sent clear message for barge-in")

# ---------------Function/Tool Calling---------------

async def end_call_function(arguments):
    # Function to handle call termination requests
    conversation_logger.log_call_end_request()
    return {
        "status": "call_ending",
        "message": "Call will be terminated after farewell"
    }

async def execute_function_call(func_name, arguments):
    # Execute function call with logging and timing
    global caller_phone, called_phone
    start_time = time.time()
    
    try:
        if func_name == "end_call":
            # Handle call ending
            result = await end_call_function(arguments)
            execution_time = time.time() - start_time
            conversation_logger.log_function_call(func_name, arguments, result, execution_time)
            return result
        elif func_name in FUNCTION_MAP:
            logger.info(f"Executing function: {func_name} with args: {arguments}")

            # Pass phone numbers to functions as part of the parameters
            if called_phone:
                arguments["called_phone"] = called_phone
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
    # Build response object for function call result
    return {
        "type": "FunctionCallResponse",
        "id": func_id,
        "name": func_name,
        "content": json.dumps(result)
    }

# Global flag to track call termination
call_should_end = False

async def handle_function_call_request(decoded, sts_ws):
    # Handle FunctionCallRequest events from Deepgram with error handling
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
            
            # Check if this was a call end request
            if func_name == "end_call" and result.get("status") == "call_ending":
                call_should_end = True
                logger.info("Call termination requested by user")
            
    except Exception as e:
        logger.error(f"Error in function call handling: {e}")
        # Send error response to prevent hanging
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
    # Process incoming JSON messages with conversation logging
    global call_should_end
    
    message_type = decoded.get("type")
    
    # Log conversation messages
    if message_type == "ConversationText":
        role = decoded.get("role", "unknown")
        content = decoded.get("content", "")
        conversation_logger.log_message(role, content)
        
        # Check is the agent is ending the call
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
            
            # Check if agent is giving farewell
            if any(phrase in content.lower() for phrase in farewell_phrases):
                logger.info("Agent farewell detected - preparing to end call")
                call_should_end = True

        # Check if user is requesting call end
        if role == "user" and any(phrase in content.lower() for phrase in ["end call", "hang up", "goodbye", "end this call", "finish call"]):
            logger.info("User requested call termination via message")
    
    # Log other important events
    elif message_type == "Welcome":
        session_id = decoded.get("session_id", "unknown")
        logger.info(f"Deepgram session established: {session_id}")
        conversation_logger.log_message("system", f"Session established: {session_id}", "welcome")
    
    elif message_type == "AgentStartedSpeaking":
        logger.info("Agent started speaking")
        
    elif message_type == "AgentAudioDone":
        logger.info("Agent finished speaking")
        
        # If call should end and agent finished speaking, terminate
        if call_should_end:
            logger.info("Terminating call after agent finished farewell")
            await asyncio.sleep(1)  # Brief pause
            # Close the websocket to end the call
            await sts_ws.close()
    
    # Handle barge-in with interruption
    await handle_barge_in(decoded, twilio_ws, streamsid)
    
    # Handle function calls
    if message_type == "FunctionCallRequest":
        await handle_function_call_request(decoded, sts_ws)

async def sts_sender(sts_ws, audio_queue):
    # Send audio chunks from Twilio to Deepgram STS with monitoring
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
    # Receive messages from Deepgram STS and forward to Twilio
    logger.info("STS receiver started")
    streamsid = await streamsid_queue.get()
    
    try:
        async for message in sts_ws:
            if type(message) is str:
                # Handle text (JSON) messages
                try:
                    decoded = json.loads(message)
                    await handle_text_message(decoded, twilio_ws, sts_ws, streamsid)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                continue
            
            # Handle audio messages (mulaw → Twilio)
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

# Global variables for call session
caller_phone = None
called_phone = None

# ---------------Twilio---------------

async def get_phone_numbers_from_call_sid(call_sid: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get caller and called phone numbers from Twilio API using callSid.
    Returns (caller_phone, called_phone) or (None, None) if failed.
    """
    try:
        from twilio.rest import Client

        # Use only the working credentials
        account_sid = os.getenv('TWILIO_MAIN_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_MAIN_ACCOUNT_AUTH_TOKEN')

        if not account_sid or not auth_token:
            logger.error('TWILIO_MAIN_ACCOUNT_SID and TWILIO_MAIN_ACCOUNT_AUTH_TOKEN required')
            return None, None

        # Initialize client and fetch call
        client = Client(account_sid, auth_token)
        call = client.calls(call_sid).fetch()

        # Extract phone numbers (from test: from_ might be None, use from_formatted)
        caller_phone = call.from_formatted
        called_phone = call.to

        logger.info(f'Retrieved from Twilio API: {caller_phone} -> {called_phone}')
        return caller_phone, called_phone

    except Exception as e:
        logger.error(f'Failed to get phone numbers from Twilio API: {e}')
        return None, None


async def twilio_receiver(twilio_ws, audio_queue, streamsid_queue):
    # Receive audio/media events from Twilio and queue them for STS
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
                    # Get Twilio streamSid and start conversation logging
                    start_data = data["start"]
                    streamsid = start_data["streamSid"]
                    
                    # Get callSid and extract phone numbers from Twilio API
                    call_sid = start_data.get("callSid")
                    
                    if call_sid:
                        logger.info(f"Retrieving phone numbers for call: {call_sid}")
                        caller_phone, called_phone = await get_phone_numbers_from_call_sid(call_sid)
                    else:
                        logger.warning("No callSid in start data")
                        caller_phone, called_phone = None, None
                    
                    # Log phone numbers
                    logger.info(f"Call from {caller_phone} to {called_phone}")
                    
                    # Store in conversation logger
                    conversation_logger.start_session(streamsid)
                    # Add phone numbers to session data
                    if conversation_logger.current_session:
                        conversation_logger.current_session["caller_phone"] = caller_phone
                        conversation_logger.current_session["called_phone"] = called_phone
                        conversation_logger.current_session["agent_id"] = called_phone  # Use called number as agent ID
                    
                    streamsid_queue.put_nowait(streamsid)
                    logger.info(f"Call started - StreamSID: {streamsid}")
                    
                elif event == "connected":
                    logger.info("Twilio connection established")
                    
                elif event == "media":
                    # Collect inbound audio
                    media = data["media"]
                    chunk = base64.b64decode(media["payload"])
                    if media["track"] == "inbound":
                        inbuffer.extend(chunk)
                        
                elif event == "stop":
                    logger.info("Call ended")
                    conversation_logger.end_session()
                    break
                
                # Send fixed-size chunks to queue
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
        # Ensure session is closed even if error occurs
        conversation_logger.end_session()
        logger.info("Twilio receiver stopped")


async def twilio_handler(twilio_ws):
    # Main Twilio connection handler (bridges Twilio and Deepgram)
    global call_should_end
    call_should_end = False  # Reset for each call
    
    audio_queue = asyncio.Queue()
    streamsid_queue = asyncio.Queue()
    
    logger.info("New Twilio connection established")
    
    try:
        async with sts_connect() as sts_ws:
            # Send configuration to STS
            config_message = load_config()
            await sts_ws.send(json.dumps(config_message))
            logger.info("Configuration sent to Deepgram")
            
            # Run sender/receiver tasks concurrently
            tasks = [
                asyncio.create_task(sts_sender(sts_ws, audio_queue)),
                asyncio.create_task(sts_receiver(sts_ws, twilio_ws, streamsid_queue)),
                asyncio.create_task(twilio_receiver(twilio_ws, audio_queue, streamsid_queue)),
            ]
            
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                # Cancel remaining tasks
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
    # Start local websocket server for Twilio
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