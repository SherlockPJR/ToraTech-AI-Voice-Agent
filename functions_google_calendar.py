import os
import datetime
import asyncio
import re
from dateutil import parser as date_parser
import dateparser
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv

from time_parser import normalise_time_input
from clinic_cache import customer_cache

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from typing import Optional, Dict

load_dotenv()

"""Google Calendar integration with appointment booking and slot management."""

# Google Calendar API scopes
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_credentials():
    """Retrieve stored credentials or trigger OAuth flow if needed."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "google_calendar_credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds


def get_service():
    """Build a Google Calendar API service object."""
    creds = get_credentials()
    return build("calendar", "v3", credentials=creds)


def is_within_working_hours(dt: datetime.datetime, working_hours: Dict, target_day: int = None) -> Tuple[bool, str]:
    """Validate datetime against clinic operating hours with break time handling."""
    weekday = dt.weekday()
    
    # Use day-specific hours if available
    if target_day is not None and weekday in working_hours:
        time_ranges = working_hours[weekday]
    # Fall back to full week format
    elif weekday in working_hours:
        time_ranges = working_hours[weekday]
    else:
        time_ranges = []
    
    if not time_ranges:
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return False, f"We are closed on {weekday_names[weekday]}s"
    
    dt_time = dt.time()
    for start_h, start_m, end_h, end_m in time_ranges:
        start_time = datetime.time(start_h, start_m)
        end_time = datetime.time(end_h, end_m)
        if start_time <= dt_time < end_time:  # Use < for end_time to avoid overlap
            return True, ""
    
    # Build readable hours format for error message
    hours_str = []
    for start_h, start_m, end_h, end_m in time_ranges:
        start_str = f"{start_h}:{start_m:02d}" if start_m else str(start_h)
        end_str = f"{end_h}:{end_m:02d}" if end_m else str(end_h)
        hours_str.append(f"{start_str}-{end_str}")
    
    return False, f"Outside working hours. We are open: {', '.join(hours_str)}"


def get_predefined_slots_for_date(date: datetime.date, 
                                  available_time_slots: List[Dict],
                                  tz: str) -> List[datetime.datetime]:
    """Extract predefined appointment slots from database for specific date."""
    tzinfo = ZoneInfo(tz)
    slots = []
    
    weekday = date.weekday()  # 0=Monday, 6=Sunday
    
    # Extract slots matching target day of week
    for slot_record in available_time_slots:
        if slot_record.get('day_of_week') == weekday and slot_record.get('is_active', True):
            slot_time_str = slot_record.get('slot_time')
            if slot_time_str:
                try:
                    # Convert time string to datetime object
                    hour, minute = map(int, slot_time_str.split(':')[:2])
                    slot_datetime = datetime.datetime.combine(
                        date, 
                        datetime.time(hour, minute), 
                        tzinfo
                    )
                    slots.append(slot_datetime)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Invalid slot time format '{slot_time_str}': {e}")
                    continue
    
    return sorted(slots)


def validate_requested_time_against_slots(requested_dt: datetime.datetime, 
                                        available_time_slots: List[Dict],
                                        target_day: int = None) -> Tuple[bool, str]:
    """Check if requested time matches available predefined appointment slots."""
    weekday = requested_dt.weekday()
    requested_time = requested_dt.time()
    
    # Filter slots by weekday if needed
    relevant_slots = available_time_slots
    if target_day is None:
        # Filter by weekday for full week data
        relevant_slots = [slot for slot in available_time_slots 
                         if slot.get('day_of_week') == weekday and slot.get('is_active', True)]
    
    # Search for exact time match in available slots
    for slot_record in relevant_slots:
        slot_time_str = slot_record.get('slot_time')
        if slot_time_str:
            try:
                hour, minute = map(int, slot_time_str.split(':')[:2])
                slot_time = datetime.time(hour, minute)
                
                if requested_time == slot_time:
                    return True, ""
            except (ValueError, IndexError):
                continue
    
    # No matching slot found - build error message with alternatives
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = weekday_names[weekday]
    
    # Extract available times for suggestions
    available_times = []
    for slot_record in relevant_slots:
        slot_time_str = slot_record.get('slot_time')
        if slot_time_str:
            try:
                hour, minute = map(int, slot_time_str.split(':')[:2])
                available_times.append(f"{hour:02d}:{minute:02d}")
            except (ValueError, IndexError):
                continue
    
    if available_times:
        return False, f"Requested time not available. Available slots on {day_name}: {', '.join(sorted(available_times))}"
    else:
        return False, f"No appointment slots available on {day_name}s"




# ---------------- Internal helpers ---------------- #

def _normalize_text_for_match(value: Optional[str]) -> str:
    """Lowercase a string and collapse whitespace for fuzzy comparisons."""
    if not value:
        return ""
    return " ".join(value.lower().split())


def _looks_time_specific(text: str) -> bool:
    """Heuristically determine if the user provided a specific appointment time."""
    lowered = text.lower()
    if re.search(r"\d{1,2}:\d{2}", lowered):
        return True
    if re.search(r"\d{1,2}\s*(?:am|pm)", lowered):
        return True
    for keyword in ("morning", "afternoon", "evening", "night"):
        if keyword in lowered:
            return True
    return False


def _get_event_start_datetime(event: Dict, tz: str) -> Optional[datetime.datetime]:
    """Extract a timezone-aware start datetime from a calendar event."""
    start_info = event.get("start", {})
    start_str = start_info.get("dateTime") or start_info.get("date")
    if not start_str:
        return None

    try:
        if start_info.get("dateTime"):
            start_dt = date_parser.isoparse(start_str)
        else:
            start_dt = datetime.datetime.fromisoformat(start_str)
            start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    except Exception:
        return None

    tzinfo = ZoneInfo(tz)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=tzinfo)
    return start_dt.astimezone(tzinfo)


def _build_search_window(original_time_input, tz: str, slot_duration: int) -> Dict:
    """Create a time window for locating an existing appointment."""
    tzinfo = ZoneInfo(tz)

    if isinstance(original_time_input, dict):
        start_info = original_time_input.get("start", {})
        start_str = start_info.get("dateTime") or start_info.get("date")
        if not start_str:
            return {"error": "original_time_input missing start time"}

        try:
            if start_info.get("dateTime"):
                start_dt = date_parser.isoparse(start_str)
                time_specific = True
            else:
                start_dt = datetime.datetime.fromisoformat(start_str)
                start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                time_specific = False
        except Exception as exc:
            return {"error": f"Could not interpret original_time_input: {exc}"}

        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=tzinfo)
        target_start = start_dt.astimezone(tzinfo)

    elif isinstance(original_time_input, str):
        time_specific = _looks_time_specific(original_time_input)
        normalized = normalise_time_input(original_time_input, tz, slot_duration)
        if "error" in normalized:
            return normalized

        start_str = normalized.get("start", {}).get("dateTime") or normalized.get("start", {}).get("date")
        if not start_str:
            return {"error": "Could not determine appointment start from input"}

        try:
            if normalized.get("start", {}).get("dateTime"):
                start_dt = date_parser.isoparse(start_str)
            else:
                start_dt = datetime.datetime.fromisoformat(start_str)
                start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        except Exception as exc:
            return {"error": f"Could not parse appointment reference time: {exc}"}

        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=tzinfo)
        target_start = start_dt.astimezone(tzinfo)

    else:
        return {"error": "original_time_input must be str or dict"}

    # Build search window
    if time_specific:
        margin_minutes = max(slot_duration, 15)
        window_start = target_start - datetime.timedelta(minutes=margin_minutes)
        window_end = target_start + datetime.timedelta(minutes=margin_minutes)
    else:
        window_start = target_start.replace(hour=0, minute=0, second=0, microsecond=0)
        window_end = window_start + datetime.timedelta(days=1)

    return {
        "target_start": target_start,
        "window_start": window_start,
        "window_end": window_end,
        "time_specific": time_specific
    }


def _event_matches_name(event: Dict, normalized_name: str) -> bool:
    """Check if the provided name appears in the event summary, description, or attendees."""
    if not normalized_name:
        return False

    candidate_fields = [
        event.get("summary"),
        event.get("description")
    ]

    for field in candidate_fields:
        if normalized_name in _normalize_text_for_match(field):
            return True

    for attendee in event.get("attendees", []):
        if normalized_name in _normalize_text_for_match(attendee.get("displayName")):
            return True

    return False


def _spell_name_for_confirmation(name: str) -> str:
    """Return a string spelling the name for spoken confirmation."""
    parts = []
    for segment in name.split():
        letters = [ch.upper() for ch in segment if ch.isalpha()]
        if letters:
            parts.append("-".join(letters))
    return " / ".join(parts)


def _format_time_for_speech(dt: datetime.datetime) -> str:
    """Format datetime into a conversational time string."""
    return dt.strftime("%I:%M %p").lstrip("0").replace(" 0", " ")


# ---------------- Agent functions ---------------- #

async def check_availability(params):
    """Validate appointment slot against database and Google Calendar availability."""
    
    # Retrieve clinic configuration from cache
    clinic_data = params.get("_clinic_data")
    if not clinic_data:
        error_msg = "CRITICAL: Clinic data not available in cache - check main.py initialization"
        print(f"{error_msg}")
        return {"error": error_msg}
    
    # Extract required parameters
    user_input = params.get("time_input")
    if not user_input:
        return {"error": "time_input is required"}
    
    caller_phone = params.get("caller_phone")
    
    # Extract day-specific clinic configuration
    working_hours = clinic_data["working_hours"]
    slot_duration = clinic_data["slot_duration_minutes"]
    tz = clinic_data["timezone"]
    calendar_id = clinic_data["calendar_id"]
    clinic_name = clinic_data["clinic_name"]
    target_day = clinic_data.get("target_day_of_week")
    
    if target_day is not None:
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = weekday_names[target_day]
        print(f"Using {day_name}-specific data for {clinic_name} - validating against {len(clinic_data['available_time_slots'])} predefined slots")
    else:
        print(f"Using cached data for {clinic_name} - validating against predefined slots")
    
    # Lookup customer info for session context
    customer_info = None
    if caller_phone:
        try:
            customer_info = await customer_cache.get_or_create_customer(caller_phone)
        except Exception as e:
            print(f"Customer lookup failed for {caller_phone}: {e}")
            # Continue without customer data

    service = get_service()
    norm = normalise_time_input(user_input, tz)
    
    # Handle time parsing errors
    if "error" in norm:
        return norm

    # Extract datetime for validation
    start_dt = datetime.datetime.fromisoformat(norm["start"]["dateTime"])
    
    # Primary validation: check against predefined appointment slots
    available_slots = clinic_data["available_time_slots"]
    slot_valid, slot_error = validate_requested_time_against_slots(start_dt, available_slots, target_day)
    if not slot_valid:
        return {
            "status": "invalid_slot",
            "message": slot_error
        }
    
    # Secondary validation: verify working hours compliance
    is_valid, hours_error = is_within_working_hours(start_dt, working_hours, target_day)
    if not is_valid:
        return {
            "status": "outside_hours", 
            "message": hours_error,
            "working_hours": working_hours
        }

    # Calculate appointment end time using clinic slot duration
    slot_end_dt = start_dt + datetime.timedelta(minutes=slot_duration)
    
    # Query Google Calendar for conflicts during appointment window
    body = {
        "timeMin": start_dt.isoformat(),
        "timeMax": slot_end_dt.isoformat(),
        "items": [{"id": calendar_id}],
    }

    try:
        freebusy = service.freebusy().query(body=body).execute()
        busy_times = freebusy["calendars"][calendar_id]["busy"]
        
        if not busy_times:
            return {"status": "available", "message": "Yes, available."}
        
        # Detect scheduling conflicts with existing events
        for busy in busy_times:
            busy_start = datetime.datetime.fromisoformat(busy["start"])
            busy_end = datetime.datetime.fromisoformat(busy["end"])
            
            # Test for time overlap with busy periods
            if (start_dt < busy_end and slot_end_dt > busy_start):
                return {"status": "busy", "message": "Not available.", "busy": busy_times}
        
        return {"status": "available", "message": "Yes, available."}
        
    except Exception as e:
        return {"error": f"Error checking calendar: {str(e)}"}
    

async def get_available_slots(params):
    """Return all free appointment slots for a specific date after checking calendar conflicts."""
    
    # Retrieve clinic configuration from cache
    clinic_data = params.get("_clinic_data")
    if not clinic_data:
        error_msg = "CRITICAL: Clinic data not available in cache - check main.py initialization"
        print(f"{error_msg}")
        return {"error": error_msg}
    
    # Extract required parameters
    date_input = params.get("date_input")
    if not date_input:
        return {"error": "date_input is required"}
    
    caller_phone = params.get("caller_phone")
    
    # Extract day-specific clinic configuration
    working_hours = clinic_data["working_hours"]
    slot_duration = clinic_data["slot_duration_minutes"]
    tz = clinic_data["timezone"]
    calendar_id = clinic_data["calendar_id"]
    clinic_name = clinic_data["clinic_name"]
    target_day = clinic_data.get("target_day_of_week")
    
    if target_day is not None:
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = weekday_names[target_day]
        print(f"Getting available slots for {clinic_name} on {date_input} ({day_name} - {len(clinic_data['available_time_slots'])} predefined slots)")
    else:
        print(f"Getting available slots for {clinic_name} on {date_input}")
    
    # Lookup customer info for session context
    customer_info = None
    if caller_phone:
        try:
            customer_info = await customer_cache.get_or_create_customer(caller_phone)
        except Exception as e:
            print(f"Customer lookup failed for {caller_phone}: {e}")

    service = get_service()

    # Parse target date string
    try:
        if isinstance(date_input, str):
            # Parse natural language date input
            parsed_date = dateparser.parse(date_input, settings={
                "TIMEZONE": tz,
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future"
            })
            if not parsed_date:
                return {"error": f"Could not parse date: '{date_input}'"}
            target_date = parsed_date.date()
        else:
            return {"error": "date_input must be a string"}

    except Exception as e:
        return {"error": f"Error parsing date: {str(e)}"}

    # Extract predefined slots for target date
    available_slots_data = clinic_data["available_time_slots"]
    
    if target_day is not None and target_date.weekday() == target_day:
        # Use day-specific slots
        time_slots = []
        tzinfo = ZoneInfo(tz)
        for slot_record in available_slots_data:
            slot_time_str = slot_record.get('slot_time')
            if slot_time_str:
                try:
                    hour, minute = map(int, slot_time_str.split(':')[:2])
                    slot_datetime = datetime.datetime.combine(target_date, datetime.time(hour, minute), tzinfo)
                    time_slots.append(slot_datetime)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Invalid slot time format '{slot_time_str}': {e}")
        time_slots = sorted(time_slots)
    else:
        # Use full week slot filtering
        time_slots = get_predefined_slots_for_date(target_date, available_slots_data, tz)
    
    if not time_slots:
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = weekday_names[target_date.weekday()]
        return {
            "status": "closed",
            "message": f"No appointment slots available on {day_name}s",
            "available_slots": []
        }

    # Query Google Calendar for day's conflicts
    tzinfo = ZoneInfo(tz)
    available_slots = []
    
    # Single query for entire day's calendar data
    day_start = datetime.datetime.combine(target_date, datetime.time.min, tzinfo)
    day_end = datetime.datetime.combine(target_date, datetime.time.max, tzinfo)
    
    body = {
        "timeMin": day_start.isoformat(),
        "timeMax": day_end.isoformat(),
        "items": [{"id": calendar_id}],
    }
    
    try:
        freebusy = service.freebusy().query(body=body).execute()
        busy_times = freebusy["calendars"][calendar_id]["busy"]
        
        # Parse busy periods for overlap detection
        busy_periods = []
        for busy in busy_times:
            start = datetime.datetime.fromisoformat(busy["start"])
            end = datetime.datetime.fromisoformat(busy["end"])
            busy_periods.append((start, end))
        
        # Test each predefined slot against calendar
        for slot in time_slots:
            slot_end = slot + datetime.timedelta(minutes=slot_duration)
            
            # Test for scheduling conflicts
            is_free = True
            for busy_start, busy_end in busy_periods:
                # Detect time overlap with busy period
                if (slot < busy_end and slot_end > busy_start):
                    is_free = False
                    break
            
            if is_free:
                available_slots.append({
                    "start_time": slot.strftime("%H:%M"),
                    "start_datetime": slot.isoformat(),
                    "duration_minutes": slot_duration
                })
        
        # Build response with available slots
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = weekday_names[target_date.weekday()]
        
        return {
            "status": "success",
            "date": target_date.isoformat(),
            "day_name": day_name,
            "available_slots": available_slots,
            "total_slots": len(available_slots),
            "working_hours": working_hours
        }
        
    except Exception as e:
        return {"error": f"Error checking calendar: {str(e)}"}
    

async def create_appointment(params):
    """Create Google Calendar appointment after validating slot and gathering customer data."""
    
    # Retrieve clinic configuration from cache
    clinic_data = params.get("_clinic_data")
    if not clinic_data:
        error_msg = "CRITICAL: Clinic data not available in cache - check main.py initialization"
        print(f"{error_msg}")
        return {"error": error_msg}
    
    # Extract required parameters
    time_input = params.get("time_input")
    if not time_input:
        return {"error": "time_input is required"}
        
    name = params.get("name", "Unknown")
    caller_phone = params.get("caller_phone")
    description = params.get("description")
    attendees = params.get("attendees", [])
    location = params.get("location")
    send_updates = params.get("send_updates", "none")
    
    # Extract day-specific clinic configuration
    working_hours = clinic_data["working_hours"]
    slot_duration = clinic_data["slot_duration_minutes"]
    tz = clinic_data["timezone"]
    calendar_id = clinic_data["calendar_id"]
    clinic_name = clinic_data["clinic_name"]
    target_day = clinic_data.get("target_day_of_week")
    
    # Use clinic location if not provided
    if not location and clinic_data.get('address'):
        location = clinic_data['address']
    
    if target_day is not None:
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = weekday_names[target_day]
        print(f"Creating {day_name} appointment for {name} at {clinic_name}")
    else:
        print(f"Creating appointment for {name} at {clinic_name}")
    
    # Lookup/create customer record
    customer_info = None
    if caller_phone:
        try:
            customer_info = await customer_cache.get_or_create_customer(caller_phone)
        except Exception as e:
            print(f"Customer lookup failed for {caller_phone}: {e}")

    service = get_service()

    # Populate event with customer information
    if customer_info:
        first_name = customer_info.get('first_name', '')
        last_name = customer_info.get('last_name', '')
        if first_name or last_name:
            name = f"{first_name} {last_name}".strip()
        
        # Include customer email in invitations
        if customer_info.get('email') and customer_info['email'] not in attendees:
            attendees.append(customer_info['email'])

    # Parse time input with clinic's slot duration
    if isinstance(time_input, str):
        norm = normalise_time_input(time_input, tz, slot_duration)
    elif isinstance(time_input, dict):
        norm = time_input
    else:
        return {"error": "time_input must be str or normalised dict"}

    # Handle time parsing errors
    if "error" in norm:
        return norm

    # Extract datetime for validation
    start_dt = datetime.datetime.fromisoformat(norm["start"]["dateTime"])
    
    # Primary validation: check against predefined appointment slots
    available_slots = clinic_data["available_time_slots"]
    slot_valid, slot_error = validate_requested_time_against_slots(start_dt, available_slots, target_day)
    if not slot_valid:
        return {
            "status": "invalid_slot",
            "message": slot_error
        }
    
    # Secondary validation: verify working hours compliance
    is_valid, hours_error = is_within_working_hours(start_dt, working_hours, target_day)
    if not is_valid:
        return {
            "status": "outside_hours", 
            "message": hours_error,
            "working_hours": working_hours
        }

    # Query Google Calendar for conflicts
    body = {
        "timeMin": norm["start"]["dateTime"],
        "timeMax": norm["end"]["dateTime"],
        "items": [{"id": calendar_id}],
    }
    
    try:
        freebusy = service.freebusy().query(body=body).execute()
        busy_times = freebusy["calendars"][calendar_id]["busy"]

        if busy_times:
            return {"status": "conflict", "message": "Slot not available.", "busy": busy_times}

        # Construct calendar event with all details
        event_summary = f"Appointment with {name}"
        if clinic_data.get('clinic_name'):
            event_summary = f"{clinic_data['clinic_name']} - Appointment with {name}"
        
        event_body = {
            "summary": event_summary,
            "start": norm["start"],
            "end": norm["end"],
        }
        
        # Format event description with contact information
        event_description_parts = []
        if description:
            event_description_parts.append(description)
        
        if customer_info:
            customer_details = []
            if customer_info.get('phone_number'):
                customer_details.append(f"Phone: {customer_info['phone_number']}")
            if customer_info.get('email'):
                customer_details.append(f"Email: {customer_info['email']}")
            
            if customer_details:
                event_description_parts.append("Customer Details:\n" + "\n".join(customer_details))
        
        if clinic_data.get('phone_number'):
            event_description_parts.append(f"Clinic Phone: {clinic_data['phone_number']}")
        
        if event_description_parts:
            event_body["description"] = "\n\n".join(event_description_parts)
        
        if location:
            event_body["location"] = location
        
        if attendees:
            event_body["attendees"] = [{"email": e} for e in attendees]

        # Insert event into Google Calendar
        created = service.events().insert(
            calendarId=calendar_id,
            body=event_body,
            sendUpdates=send_updates
        ).execute()

        return {
            "status": "created",
            "event": created,
            "message": f"Appointment booked for {name}",
            "event_link": created.get('htmlLink'),
            "appointment_details": {
                "customer_name": name,
                "clinic_name": clinic_data.get('clinic_name', 'Unknown Clinic'),
                "datetime": start_dt.isoformat(),
                "duration_minutes": (datetime.datetime.fromisoformat(norm["end"]["dateTime"]) - start_dt).seconds // 60
            }
        }
        
    except Exception as e:
        return {"error": f"Error creating appointment: {str(e)}"}


async def cancel_or_reschedule_appointment(params):
    """Cancel an existing appointment and optionally book a new slot."""

    clinic_data = params.get("_clinic_data")
    if not clinic_data:
        error_msg = "CRITICAL: Clinic data not available in cache - check main.py initialization"
        print(error_msg)
        return {"error": error_msg}

    action = (params.get("action") or "cancel").lower()
    new_time_input = params.get("new_time_input")
    if new_time_input and action == "cancel":
        action = "reschedule"

    if action not in {"cancel", "reschedule"}:
        return {"error": "action must be either 'cancel' or 'reschedule'"}

    customer_name = params.get("customer_name") or params.get("name")
    if not customer_name:
        return {"error": "customer_name is required"}

    original_time_input = params.get("original_time_input") or params.get("time_input")
    if not original_time_input:
        return {"error": "original_time_input is required"}

    slot_duration = clinic_data["slot_duration_minutes"]
    tz = clinic_data["timezone"]
    calendar_id = clinic_data["calendar_id"]

    search_window = _build_search_window(original_time_input, tz, slot_duration)
    if "error" in search_window:
        return search_window

    service = get_service()

    window_start_iso = search_window["window_start"].astimezone(ZoneInfo(tz)).isoformat()
    window_end_iso = search_window["window_end"].astimezone(ZoneInfo(tz)).isoformat()

    events: List[Dict] = []
    page_token = None
    while True:
        try:
            response = service.events().list(
                calendarId=calendar_id,
                timeMin=window_start_iso,
                timeMax=window_end_iso,
                singleEvents=True,
                orderBy="startTime",
                pageToken=page_token
            ).execute()
        except Exception as exc:
            return {"error": f"Error looking up existing appointment: {exc}"}

        events.extend(response.get("items", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    normalized_name = _normalize_text_for_match(customer_name)
    matches: List[Tuple[Dict, datetime.datetime]] = []
    for event in events:
        if not _event_matches_name(event, normalized_name):
            continue

        event_start = _get_event_start_datetime(event, tz)
        if not event_start:
            continue

        if search_window["time_specific"]:
            diff = abs((event_start - search_window["target_start"]).total_seconds())
            if diff > max(slot_duration, 15) * 60:
                continue
        else:
            if event_start.date() != search_window["target_start"].date():
                continue

        matches.append((event, event_start))

    if not matches:
        friendly_date = search_window["target_start"].strftime("%A, %B %d")
        return {
            "status": "not_found",
            "message": f"Could not locate an appointment for {customer_name} on {friendly_date}",
            "spelled_name": _spell_name_for_confirmation(customer_name)
        }

    if len(matches) > 1 and not search_window["time_specific"]:
        friendly_date = search_window["target_start"].strftime("%A, %B %d")
        options = []
        for event, start_dt in matches:
            options.append({
                "event_id": event.get("id"),
                "summary": event.get("summary"),
                "start": event.get("start"),
                "link": event.get("htmlLink"),
                "spoken_time": _format_time_for_speech(start_dt)
            })
        spoken_times_list = [option["spoken_time"] for option in options]
        if len(spoken_times_list) > 1:
            times_prefix = ", ".join(spoken_times_list[:-1])
            spoken_times = f"{times_prefix}, and {spoken_times_list[-1]}"
        else:
            spoken_times = spoken_times_list[0]
        return {
            "status": "needs_selection",
            "message": f"I can see {customer_name} is booked on {friendly_date} at {spoken_times}. Which one should I cancel or reschedule?",
            "matches": options,
            "spelled_name": _spell_name_for_confirmation(customer_name)
        }

    matches.sort(key=lambda item: abs((item[1] - search_window["target_start"]).total_seconds()))
    target_event, target_start = matches[0]

    cancelled_event_info = {
        "event_id": target_event.get("id"),
        "summary": target_event.get("summary"),
        "start": target_event.get("start"),
        "end": target_event.get("end"),
        "htmlLink": target_event.get("htmlLink"),
    }

    try:
        service.events().delete(calendarId=calendar_id, eventId=target_event["id"]).execute()
    except Exception as exc:
        return {"error": f"Error cancelling appointment: {exc}"}

    if action == "cancel":
        return {
            "status": "cancelled",
            "message": f"Cancelled appointment for {customer_name}. Would they like to reschedule or book another time with us?",
            "cancelled_event": cancelled_event_info,
            "spelled_name": _spell_name_for_confirmation(customer_name)
        }

    # Action is reschedule
    base_response = {
        "cancelled_event": cancelled_event_info,
        "original_start": target_start.isoformat(),
        "spelled_name": _spell_name_for_confirmation(customer_name)
    }

    if not new_time_input:
        base_response.update({
            "status": "cancelled_pending_reschedule",
            "message": f"Cancelled appointment for {customer_name}. Ask if they'd like to reschedule or book another time.",
        })
        return base_response

    # Reuse create_appointment for booking the new time
    new_params = dict(params)
    new_params["time_input"] = new_time_input
    new_params["name"] = customer_name
    new_params["_clinic_data"] = clinic_data
    new_params.pop("original_time_input", None)

    reschedule_result = await create_appointment(new_params)

    if reschedule_result.get("status") == "created":
        base_response.update({
            "status": "rescheduled",
            "message": f"Rescheduled appointment for {customer_name}.",
            "new_event": reschedule_result.get("event"),
            "appointment_details": reschedule_result.get("appointment_details")
        })
    else:
        base_response.update({
            "status": "cancelled_pending_reschedule",
            "message": f"Cancelled appointment for {customer_name}, but could not book the new time.",
            "reschedule_error": reschedule_result
        })

    return base_response


async def get_clinic_hours(params):
    """Return formatted clinic operating hours for specified date using cached data."""
    
    # Retrieve clinic configuration from cache (no database queries)
    clinic_data = params.get("_clinic_data")
    if not clinic_data:
        error_msg = "CRITICAL: Clinic data not available in cache - check main.py initialization"
        print(f"{error_msg}")
        return {"error": error_msg}
    
    date_input = params.get("date_input", "today")
    clinic_name = clinic_data["clinic_name"]
    target_day = clinic_data.get("target_day_of_week")
    
    if target_day is not None:
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = weekday_names[target_day]
        print(f"Getting {day_name} clinic hours for {clinic_name} (day-specific data)")
    else:
        print(f"Getting clinic hours for {clinic_name} on {date_input}")
    
    try:
        # Extract target day of week from date input
        normalized = normalise_time_input(date_input)
        
        # Parse datetime and extract weekday
        target_date = datetime.datetime.fromisoformat(normalized["start"]["dateTime"])
        
        # Convert to weekday index
        target_day_of_week = target_date.weekday()
            
    except Exception as date_error:
        return {"error": f"Could not parse date input '{date_input}': {str(date_error)}"}
    
    # Extract operating hours for target day
    raw_operating_hours = clinic_data["raw_operating_hours"]
    
    if target_day is not None and len(raw_operating_hours) == 1:
        # Use pre-filtered day data
        day_hours = raw_operating_hours[0]
        if day_hours['day_of_week'] != target_day_of_week:
            print(f"Warning: Day mismatch - cached data for day {day_hours['day_of_week']}, requested day {target_day_of_week}")
    else:
        # Search full week data for target day
        day_hours = None
        for hour_record in raw_operating_hours:
            if hour_record['day_of_week'] == target_day_of_week:
                day_hours = hour_record
                break
    
    if not day_hours:
        return {"error": f"No opening hours found for day {target_day_of_week} at {clinic_name}"}
    
    # Build readable hours format
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = weekday_names[day_hours['day_of_week']]
    
    if day_hours['is_closed']:
        formatted_hours = f"{day_name}: Closed"
    else:
        open_time = day_hours['open_time'][:5] if day_hours['open_time'] else "N/A"  # Format HH:MM
        close_time = day_hours['close_time'][:5] if day_hours['close_time'] else "N/A"
        
        if day_hours['break_start_time'] and day_hours['break_end_time']:
            break_start = day_hours['break_start_time'][:5]
            break_end = day_hours['break_end_time'][:5]
            formatted_hours = f"{day_name}: {open_time}-{break_start}, {break_end}-{close_time}"
        else:
            formatted_hours = f"{day_name}: {open_time}-{close_time}"
    
    return {
        "status": "success",
        "requested_date": date_input,
        "target_date": target_date.strftime("%Y-%m-%d"),
        "day_of_week": target_day_of_week,
        "opening_hours": formatted_hours,
        "raw_data": day_hours
    }


# Function registry for agent execution

FUNCTION_MAP = {
    "check_availability": check_availability,
    "create_appointment": create_appointment,
    "get_available_slots": get_available_slots,
    "cancel_or_reschedule_appointment": cancel_or_reschedule_appointment,
    "get_clinic_hours": get_clinic_hours,
}
