import json
import os
import time
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Logs conversation sessions with structured metadata and timing."""

    def __init__(self, log_file="conversations.json"):
        self.log_file = log_file
        self.current_session = None

    def start_session(self, stream_sid):
        """Initialize new conversation session with metadata tracking."""
        # Phone numbers are set later by the caller (main.py) after retrieval
        self.current_session = {
            "session_id": str(uuid.uuid4()),
            "stream_sid": stream_sid,
            "caller_phone": None,
            "called_phone": None,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "messages": [],
            "function_calls": [],
            "metadata": {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "function_calls": 0,
                "call_ended_by_user": False,
            },
        }
        logger.info(
            f"Started conversation session: {self.current_session['session_id']}"
        )

    def log_message(self, role, content, message_type="text"):
        """Record conversation message with role attribution."""
        if not self.current_session:
            return

        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "type": message_type,
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
            "success": "error" not in result if isinstance(result, dict) else True,
        }

        self.current_session["function_calls"].append(function_call)
        self.current_session["metadata"]["function_calls"] += 1

        logger.info(
            f"Function call: {func_name} - Success: {function_call['success']}"
        )

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
                with open(self.log_file, "r") as f:
                    conversations = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                conversations = []

        # Add current session
        conversations.append(self.current_session)

        # Save to file
        with open(self.log_file, "w") as f:
            json.dump(conversations, f, indent=2)

        logger.info(
            "Conversation ended - Duration: %.1fs, Messages: %d, Functions: %d",
            duration_seconds,
            self.current_session["metadata"]["total_messages"],
            self.current_session["metadata"]["function_calls"],
        )

        self.current_session = None


class PerformanceLogger:
    """Tracks critical performance metrics for customer experience optimization."""

    def __init__(self, log_file="performance.json"):
        self.log_file = log_file
        self.current_session = None
        self.enabled = os.getenv("PERFORMANCE_LOGGING_LEVEL", "BASIC").upper() != "OFF"
        self.timing_stack = {}
        # Track per-turn timing breakdowns without changing existing arrays
        self.current_turn = None
        # Transport measurement windows (per user utterance before turn starts)
        self._inbound_decode_ms = None
        self._uplink_send_ms = None

    def start_session(self, stream_sid, caller_phone=None, called_phone=None):
        """Initialize performance tracking session."""
        if not self.enabled:
            return

        self.current_session = {
            "session_id": str(uuid.uuid4()),
            "stream_sid": stream_sid,
            "caller_phone": caller_phone,
            "called_phone": called_phone,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "metrics": {
                "call_setup_time_ms": None,
                "speech_to_response_latencies_ms": [],
                "interruption_response_times_ms": [],
                "function_executions": [],
                # New: per-turn structured breakdowns
                "turns": [],
                "summary": {},
            },
        }
        logger.info(
            f"Performance tracking started: {self.current_session['session_id']}"
        )

    # -------- Per-turn helpers (non-disruptive to existing metrics) --------
    def _ensure_turn(self):
        if not self.enabled or not self.current_session:
            return
        if self.current_turn is None:
            self.current_turn = {
                "turn_id": len(self.current_session["metrics"].get("turns", []))
                + 1,
                "timestamps": {},
                "functions": [],
                "durations": {},
            }

    def begin_turn(self):
        if not self.enabled or not self.current_session:
            return
        # If a previous turn exists and has agent history, finalize it; otherwise keep it open
        if self.current_turn:
            has_history = (
                self.current_turn.get("timestamps", {}).get("assistant_history_ts")
                is not None
            )
            has_durations = bool(self.current_turn.get("durations"))
            if has_history and not has_durations:
                try:
                    self.finalize_turn()
                except Exception:
                    pass
            elif not has_history:
                # Keep the current turn open; do not create a new one yet
                return
        self._ensure_turn()

    def set_turn_timestamp(self, key, iso_ts=None):
        if not self.enabled or not self.current_session:
            return
        # Only set on an existing turn; avoid implicitly creating agent-only turns
        if self.current_turn is None:
            return
        ts_map = self.current_turn["timestamps"]
        if key not in ts_map:
            ts_map[key] = iso_ts or datetime.now().isoformat()

    # -------- Transport metrics (per-turn attachments) --------
    def begin_inbound_window(self):
        if not self.enabled or not self.current_session:
            return
        self._inbound_decode_ms = []

    def add_inbound_decode_ms(self, ms):
        if not self.enabled or not self.current_session:
            return
        if self._inbound_decode_ms is not None:
            self._inbound_decode_ms.append(ms)

    def begin_uplink_window(self):
        if not self.enabled or not self.current_session:
            return
        self._uplink_send_ms = []

    def add_uplink_send_ms(self, ms):
        if not self.enabled or not self.current_session:
            return
        if self._uplink_send_ms is not None:
            self._uplink_send_ms.append(ms)

    def attach_transport_avgs_to_current_turn(self):
        if not self.enabled or not self.current_session:
            return
        if self.current_turn is None:
            return
        transport = self.current_turn.setdefault("transport", {})
        if self._inbound_decode_ms:
            transport["inbound_decode_ms_avg"] = sum(self._inbound_decode_ms) / len(self._inbound_decode_ms)
        else:
            transport["inbound_decode_ms_avg"] = None
        if self._uplink_send_ms:
            transport["uplink_send_ms_avg"] = sum(self._uplink_send_ms) / len(self._uplink_send_ms)
        else:
            transport["uplink_send_ms_avg"] = None
        # Clear windows
        self._inbound_decode_ms = None
        self._uplink_send_ms = None

    def set_turn_transport_metric(self, key, value):
        if not self.enabled or not self.current_session or self.current_turn is None:
            return
        transport = self.current_turn.setdefault("transport", {})
        if key not in transport:
            transport[key] = value

    def add_turn_function(self, name, duration_ms, start_iso=None, end_iso=None):
        if not self.enabled or not self.current_session:
            return
        # Only attach to an existing turn; do not create one implicitly
        if self.current_turn is None:
            return
        self.current_turn["functions"].append(
            {
                "name": name,
                "duration_ms": duration_ms,
                "start_ts": start_iso,
                "end_ts": end_iso,
            }
        )

    def note_tool_result_sent(self):
        if not self.enabled or not self.current_session:
            return
        self.set_turn_timestamp("last_tool_result_sent_ts")

    def _parse_iso(self, s):
        try:
            return datetime.fromisoformat(s) if s else None
        except Exception:
            return None

    def _compute_durations_for_turn(self, turn):
        """Compute additive breakdown for a given turn dict; returns durations dict."""
        ts = turn.get("timestamps", {})
        user_end = self._parse_iso(ts.get("user_speech_end_ts"))
        agent_history = self._parse_iso(ts.get("assistant_history_ts"))
        llm_decision = self._parse_iso(ts.get("llm_decision_ts"))
        last_tool_result = self._parse_iso(ts.get("last_tool_result_sent_ts"))
        first_agent_audio = self._parse_iso(ts.get("first_agent_audio_ts"))
        audio_to_audio_ms = None
        pre_decision_ms = None
        post_tools_to_history_ms = None

        # Only include tool durations that finished before assistant history
        if agent_history:
            tool_total_ms = sum(
                (f.get("duration_ms") or 0)
                for f in turn.get("functions", [])
                if self._parse_iso(f.get("end_ts")) and self._parse_iso(f.get("end_ts")) <= agent_history
            )
        else:
            tool_total_ms = sum((f.get("duration_ms") or 0) for f in turn.get("functions", []))

        if user_end and agent_history:
            audio_to_audio_ms = max((agent_history - user_end).total_seconds() * 1000, 0)

            # Pre-decision: capped to audio_to_audio_ms and non-negative
            if llm_decision and llm_decision <= agent_history:
                pre_decision_ms = max((llm_decision - user_end).total_seconds() * 1000, 0)
            elif not turn.get("functions"):
                pre_decision_ms = audio_to_audio_ms
                tool_total_ms = 0
            else:
                pre_decision_ms = 0

            # Post-tools to history: only if last_tool_result exists before history
            if last_tool_result and agent_history and last_tool_result <= agent_history:
                post_tools_to_history_ms = max(
                    (agent_history - last_tool_result).total_seconds() * 1000, 0
                )
            else:
                post_tools_to_history_ms = 0

            # Ensure additive cap: sum(parts) must not exceed audio_to_audio
            parts_sum = (pre_decision_ms or 0) + (tool_total_ms or 0) + (post_tools_to_history_ms or 0)
            if parts_sum > (audio_to_audio_ms or 0):
                # Reduce pre_decision to fit within the cap
                pre_decision_ms = max((audio_to_audio_ms or 0) - (tool_total_ms or 0) - (post_tools_to_history_ms or 0), 0)

        # TTS time: assistant history to first agent audio (if available)
        tts_ms = None
        if agent_history and first_agent_audio and first_agent_audio >= agent_history:
            tts_ms = (first_agent_audio - agent_history).total_seconds() * 1000

        return {
            "audio_to_audio_ms": audio_to_audio_ms,
            "pre_decision_ms": pre_decision_ms,
            "tool_total_ms": tool_total_ms,
            "post_tools_to_history_ms": post_tools_to_history_ms,
            "tts_ms": tts_ms,
        }

    def finalize_turn(self):
        """Compute additive breakdown that sums to audio_to_audio latency and store it."""
        if not self.enabled or not self.current_session or not self.current_turn:
            return
        durations = self._compute_durations_for_turn(self.current_turn)
        self.current_turn["durations"] = durations
        self.current_session["metrics"]["turns"].append(self.current_turn)
        logger.info(
            "Turn %s latencies - audio_to_audio: %s ms, components: pre_decision=%s, tools=%s, post_tools_to_history=%s",
            self.current_turn["turn_id"],
            durations.get("audio_to_audio_ms"),
            durations.get("pre_decision_ms"),
            durations.get("tool_total_ms"),
            durations.get("post_tools_to_history_ms"),
        )
        self.current_turn = None

    def attach_agent_audio_end(self, iso_ts):
        """Attach agent_audio_end_ts to the most recent prior turn with assistant history and finalize it."""
        if not self.enabled or not self.current_session:
            return
        turns = self.current_session["metrics"].get("turns", [])
        for turn in reversed(turns):
            ts = turn.get("timestamps", {})
            if ts.get("assistant_history_ts") and not ts.get("agent_audio_end_ts"):
                ts["agent_audio_end_ts"] = iso_ts
                # Recompute durations for this turn in-place
                turn["durations"] = self._compute_durations_for_turn(turn)
                logger.info(
                    "Turn %s (retro) finalized - audio_to_audio: %s ms",
                    turn.get("turn_id"),
                    turn.get("durations", {}).get("audio_to_audio_ms"),
                )
                break

    def start_timing(self, metric_name, context=None):
        """Start timing measurement for a metric."""
        if not self.enabled or not self.current_session:
            return

        self.timing_stack[metric_name] = {
            "start_time": time.time(),
            "context": context or {},
        }

    def end_timing(self, metric_name, context=None):
        """End timing measurement and record the metric."""
        if (
            not self.enabled
            or not self.current_session
            or metric_name not in self.timing_stack
        ):
            return

        start_data = self.timing_stack.pop(metric_name)
        duration_ms = (time.time() - start_data["start_time"]) * 1000

        self.log_metric(
            metric_name, duration_ms, "ms", {**start_data["context"], **(context or {})}
        )

    def log_metric(self, metric_name, value, unit="ms", context=None):
        """Record a performance metric."""
        if not self.enabled or not self.current_session:
            return

        try:
            metrics = self.current_session["metrics"]

            if metric_name == "call_setup_time":
                metrics["call_setup_time_ms"] = value
            elif metric_name == "speech_to_response":
                metrics["speech_to_response_latencies_ms"].append(value)
            elif metric_name == "interruption_response":
                metrics["interruption_response_times_ms"].append(value)
            elif metric_name == "function_execution":
                metrics["function_executions"].append(
                    {
                        "name": context.get("function_name", "unknown"),
                        "duration_ms": value,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            # Removed deepgram_processing and raw audio_timing_events to reduce noise

        except Exception as e:
            logger.warning(f"Performance metric recording failed: {e}")

    def end_session(self):
        """Finalize session with summary statistics and save to file."""
        if not self.enabled or not self.current_session:
            return

        try:
            # Finalize any open turn before closing session
            try:
                if self.current_turn and not self.current_turn.get("durations"):
                    self.finalize_turn()
            except Exception:
                pass
            self.current_session["end_time"] = datetime.now().isoformat()

            # Calculate summary statistics
            metrics = self.current_session["metrics"]
            summary = {}

            # Calculate audio-to-audio latencies from per-turn durations
            turns_for_latency = metrics.get("turns", [])
            audio_latencies = [
                t.get("durations", {}).get("audio_to_audio_ms")
                for t in turns_for_latency
                if t.get("durations", {}).get("audio_to_audio_ms") is not None
            ]

            if audio_latencies:
                summary["audio_to_audio_latencies_ms"] = audio_latencies
                summary["avg_audio_latency_ms"] = sum(audio_latencies) / len(
                    audio_latencies
                )
                summary["max_audio_latency_ms"] = max(audio_latencies)
                summary["min_audio_latency_ms"] = min(audio_latencies)

            if metrics["speech_to_response_latencies_ms"]:
                summary["avg_response_latency_ms"] = sum(
                    metrics["speech_to_response_latencies_ms"]
                ) / len(metrics["speech_to_response_latencies_ms"])
                summary["max_response_latency_ms"] = max(
                    metrics["speech_to_response_latencies_ms"]
                )

            if metrics["interruption_response_times_ms"]:
                summary["avg_interruption_response_ms"] = sum(
                    metrics["interruption_response_times_ms"]
                ) / len(metrics["interruption_response_times_ms"])

            if metrics["function_executions"]:
                summary["total_function_time_ms"] = sum(
                    f["duration_ms"] for f in metrics["function_executions"]
                )
                summary["function_count"] = len(metrics["function_executions"])

            # Aggregate new per-turn transport and TTS metrics
            turns = metrics.get("turns", [])
            def _avg(lst):
                return (sum(lst) / len(lst)) if lst else None

            tts_ms_list = [
                t.get("durations", {}).get("tts_ms")
                for t in turns
                if t.get("durations", {}).get("tts_ms") is not None
            ]
            inbound_decode_list = [
                t.get("transport", {}).get("inbound_decode_ms_avg")
                for t in turns
                if t.get("transport", {}).get("inbound_decode_ms_avg") is not None
            ]
            uplink_send_list = [
                t.get("transport", {}).get("uplink_send_ms_avg")
                for t in turns
                if t.get("transport", {}).get("uplink_send_ms_avg") is not None
            ]
            tts_encode_first_list = [
                t.get("transport", {}).get("tts_encode_ms_first")
                for t in turns
                if t.get("transport", {}).get("tts_encode_ms_first") is not None
            ]
            twilio_send_first_list = [
                t.get("transport", {}).get("twilio_send_ms_first")
                for t in turns
                if t.get("transport", {}).get("twilio_send_ms_first") is not None
            ]

            avg_tts_ms = _avg(tts_ms_list)
            avg_inbound_decode_ms = _avg(inbound_decode_list)
            avg_uplink_send_ms = _avg(uplink_send_list)
            avg_tts_encode_ms_first = _avg(tts_encode_first_list)
            avg_twilio_send_ms_first = _avg(twilio_send_first_list)

            if avg_tts_ms is not None:
                summary["avg_tts_ms"] = avg_tts_ms
            if avg_inbound_decode_ms is not None:
                summary["avg_inbound_decode_ms"] = avg_inbound_decode_ms
            if avg_uplink_send_ms is not None:
                summary["avg_uplink_send_ms"] = avg_uplink_send_ms
            if avg_tts_encode_ms_first is not None:
                summary["avg_tts_encode_ms_first"] = avg_tts_encode_ms_first
            if avg_twilio_send_ms_first is not None:
                summary["avg_twilio_send_ms_first"] = avg_twilio_send_ms_first

            # Calculate call duration
            start = datetime.fromisoformat(self.current_session["start_time"])
            end = datetime.fromisoformat(self.current_session["end_time"])
            summary["call_duration_seconds"] = (end - start).total_seconds()

            metrics["summary"] = summary

            # Save to file
            self._save_to_file()

            audio_avg = summary.get("avg_audio_latency_ms", 0)
            logger.info(
                "Performance session completed - True audio latency: %.1fms, Functions: %s",
                audio_avg,
                summary.get("function_count", 0),
            )

        except Exception as e:
            logger.error(f"Performance session finalization failed: {e}")
        finally:
            self.current_session = None

    def _save_to_file(self):
        """Save performance data to JSON file."""
        try:
            # Load existing performance data
            performance_data = []
            if os.path.exists(self.log_file):
                try:
                    with open(self.log_file, "r") as f:
                        performance_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    performance_data = []

            # Add current session
            performance_data.append(self.current_session)

            # Save to file
            with open(self.log_file, "w") as f:
                json.dump(performance_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
