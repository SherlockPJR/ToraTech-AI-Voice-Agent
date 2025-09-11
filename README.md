# AI Voice Appointment Agent

A real-time voice agent that answers phone calls, understands speech, and books appointments. It bridges Twilio phone audio to a Deepgram Agent (STT/LLM/TTS), executes calendar tools against Google Calendar and Supabase, and streams natural speech back to the caller.

## Flowchart

```
Caller (PSTN)
   |
   v
Twilio Media Stream (mulaw/8kHz)
   |
   |  WebSocket (phone audio)
   v
Voice Agent Server (main.py)
   |\
   | \ 1) Uplink audio  ->
   |  \----------------------->  Deepgram Agent (STS)
   |                             - STT: speech -> text
   |                             - LLM: intent + (optional) tool calls
   |                             - TTS: text -> speech
   |  /<-----------------------
   | / 2) Downlink audio (agent speech)
   |
   | 3) Tool calls (JSON)
   v
Calendar/DB Tools (functions_google_calendar.py)
   |\
   | \-> time parsing (time_parser.py)
   |  -> clinic/customer cache (clinic_cache.py)
   |
   |--> Google Calendar (availability, events)
   |--> Supabase (clinics, customers, hours)

Result -> Deepgram Agent (tool responses)
Agent speech -> main.py -> Twilio -> Caller
```

Why these nodes
- Twilio: Provides the phone call and full-duplex media over WebSocket (mulaw 8kHz).
- Deepgram Agent (STS): One connection that handles STT, LLM decisions, TTS, and orchestrates function calls.
- Google Calendar: Ground-truth scheduling (availability checks and appointment creation).
- Supabase: Stores clinic configuration, hours, and customer records; supports fast lookups.
- main.py: The bridge and orchestrator between Twilio and Deepgram; executes tools and manages logging.

## Repository Structure

- `main.py`: WebSocket server that bridges Twilio <-> Deepgram, handles function calls, barge-in, and session lifecycle.
- `config.json`: Deepgram Agent configuration (audio IO, system prompt, function schemas, greetings).
- `functions_google_calendar.py`: Implements tools used by the agent:
  - `check_availability`, `create_appointment`, `get_available_slots`, `get_clinic_hours`, `end_call`.
- Helpers:
  - `clinic_cache.py`: Loads and caches clinic + day-specific data for fast tool execution.
  - `time_parser.py`: Robust natural language time parsing for dates, ranges, and recurrence.
  - `loggers.py`: ConversationLogger and PerformanceLogger (structured conversation + performance metrics).
- Logs/artifacts:
  - `voice_agent.log`: System logs.
  - `conversations.json`: Full conversation transcripts with function calls.
  - `performance.json`: Per-session performance metrics with per-turn breakdowns.

## Setup

1) Requirements
- Python 3.11+
- Twilio account credentials for main account (to fetch call info):
  - `TWILIO_MAIN_ACCOUNT_SID`
  - `TWILIO_MAIN_ACCOUNT_AUTH_TOKEN`
- Deepgram API key: `DEEPGRAM_API_KEY`
- Supabase: `SUPABASE_URL`, `SUPABASE_KEY`
- Google Calendar OAuth files in project root:
  - `google_calendar_credentials.json` (client secrets)
  - On first run, `token.json` is generated automatically

2) Environment
- Create `.env` in project root with at least:
  - `DEEPGRAM_API_KEY=...`
  - `TWILIO_MAIN_ACCOUNT_SID=...`
  - `TWILIO_MAIN_ACCOUNT_AUTH_TOKEN=...`
  - `SUPABASE_URL=...`
  - `SUPABASE_KEY=...`
  - Optional: `PERFORMANCE_LOGGING_LEVEL=BASIC` (any value other than `OFF` enables logging)

3) Install and run
```
uv sync
uv run python main.py
```
The server listens on `localhost:5000` for Twilio Media Stream WebSocket connections.

Tip: In development, expose `ws(s)://localhost:5000` with a tunnel (e.g., ngrok) and configure your Twilio Voice URL to that public address.

## Configuration (config.json)

`config.json` configures the Deepgram Agent:
- Audio IO: mulaw 8kHz in/out (Twilio compatible)
- STT and TTS voice
- LLM behavior (system prompt, temperature, tools)
- Function definitions (names and JSON schemas) for the calendar tools

The agent will call your functions by sending `FunctionCallRequest` events to `main.py`, which will execute and return `FunctionCallResponse` results.

## Runtime Behavior

- Twilio connects to `main.py` and streams caller audio.
- `main.py` forwards audio to Deepgram STS and relays agent audio back to Twilio.
- When the agent needs data/actions, it calls tools in `functions_google_calendar.py`.
- Barge-in is supported: if the caller starts speaking, the agent audio buffer is cleared to reduce interruption latency.

## Performance Logging

Performance data is written to `performance.json` with per-session and per-turn details. Each turn corresponds to one user utterance and the ensuing agent response.

Per-turn durations (durations)
- `audio_to_audio_ms`: Caller finished speaking → agent reply text ready (pre‑TTS).
- `pre_decision_ms`: Model thinking time before any tool calls.
- `tool_total_ms`: Sum of all tool execution durations within the turn.
- `post_tools_to_history_ms`: After the last tool result → agent reply text ready.
- `tts_ms`: Agent text → first audio frame synthesized.

Per-turn transport (transport)
- `inbound_decode_ms_avg`: Avg base64 decode time for inbound Twilio chunks.
- `uplink_send_ms_avg`: Avg time awaiting send of chunks to Deepgram.
- `tts_encode_ms_first`: Time to base64-encode the first agent audio chunk.
- `twilio_send_ms_first`: Time to send the first agent audio chunk to Twilio.

Session summary (summary)
- Aggregates of the above (e.g., average audio_to_audio, tts, inbound decode, uplink send, etc.), function counts and totals, and call duration.

Notes
- The additive parts are designed so that, for tool-using turns:
  - `audio_to_audio_ms = pre_decision_ms + tool_total_ms + post_tools_to_history_ms`
- For turns without tools:
  - `audio_to_audio_ms = pre_decision_ms`
- `tts_ms` explains text→audio time and complements `audio_to_audio_ms` (which stops at reply text readiness).

## Development

Useful commands
```
uv sync                                # Install deps
uv run python main.py                  # Start the voice agent server
uv run python -m py_compile main.py    # Quick syntax check
uv run python -m py_compile functions_google_calendar.py
```

Common files to prepare
- `.env` with all required secrets and URLs
- `config.json` (agent behavior + functions)
- `google_calendar_credentials.json` in repo root

## Troubleshooting

- No audio or delayed replies
  - Check `voice_agent.log` for connection issues.
  - Inspect `performance.json` per-turn metrics to see where latency accrues (pre_decision, tools, post_tools_to_history, tts, transport).
- Tool failures
  - Confirm `SUPABASE_URL`/`SUPABASE_KEY` and Google Calendar credentials.
  - Verify phone number → clinic mapping exists in Supabase (used by `clinic_cache.py`).
- Twilio connection
  - Ensure your Twilio Voice Webhook targets the tunnel URL that forwards to `ws(s)://localhost:5000`.

## Security

- Keep `.env`, `google_calendar_credentials.json`, and `token.json` private.
- Do not commit secrets; use environment variables and secret managers in production.

## License

Proprietary – internal use unless a license is added.

