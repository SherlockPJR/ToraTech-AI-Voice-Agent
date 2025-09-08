import os
import datetime
import asyncio
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

# Google Calendar API scopes
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# REMOVED: get_clinic_slot_duration - data now comes from clinic cache

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

# REMOVED: init_supabase - now handled by clinic_cache.py

# REMOVED: get_clinic_by_phone - data now comes from clinic cache

# REMOVED: get_customer_by_phone - now handled by customer_cache

# REMOVED: get_or_create_customer_by_phone - now handled by customer_cache

# REMOVED: get_clinic_working_hours - data now comes from clinic cache


def get_service():
    """Build a Google Calendar API service object."""
    creds = get_credentials()
    return build("calendar", "v3", credentials=creds)


def is_within_working_hours(dt: datetime.datetime, working_hours: Dict, target_day: int = None) -> Tuple[bool, str]:
    """
    Check if datetime falls within working hours.
    
    Args:
        dt: datetime to check
        working_hours: working hours dict (may be day-specific or full week)
        target_day: specific day of week if working_hours is day-specific
    
    Returns:
        (is_valid, error_message)
    """
    weekday = dt.weekday()
    
    # Handle day-specific working hours (new format)
    if target_day is not None and weekday in working_hours:
        time_ranges = working_hours[weekday]
    # Handle full week working hours (fallback)
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
    
    # Format working hours for error message
    hours_str = []
    for start_h, start_m, end_h, end_m in time_ranges:
        start_str = f"{start_h}:{start_m:02d}" if start_m else str(start_h)
        end_str = f"{end_h}:{end_m:02d}" if end_m else str(end_h)
        hours_str.append(f"{start_str}-{end_str}")
    
    return False, f"Outside working hours. We are open: {', '.join(hours_str)}"


def get_predefined_slots_for_date(date: datetime.date, 
                                  available_time_slots: List[Dict],
                                  tz: str) -> List[datetime.datetime]:
    """
    Get predefined appointment slots from database for a specific date.
    
    Args:
        date: The date to get slots for
        available_time_slots: Predefined slots from clinics_available_time_slots table
        tz: Timezone string
        
    Returns:
        List of datetime objects representing predefined time slots for this date
    """
    tzinfo = ZoneInfo(tz)
    slots = []
    
    weekday = date.weekday()  # 0=Monday, 6=Sunday
    
    # Filter slots for the specific day of week
    for slot_record in available_time_slots:
        if slot_record.get('day_of_week') == weekday and slot_record.get('is_active', True):
            slot_time_str = slot_record.get('slot_time')
            if slot_time_str:
                try:
                    # Parse time string like "10:00:00"
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
    
    return sorted(slots)  # Return sorted by time


def validate_requested_time_against_slots(requested_dt: datetime.datetime, 
                                        available_time_slots: List[Dict],
                                        target_day: int = None) -> Tuple[bool, str]:
    """
    Validate that the requested appointment time matches a predefined slot.
    
    Args:
        requested_dt: The requested appointment datetime
        available_time_slots: Predefined slots from database (may be day-filtered)
        target_day: specific day of week if slots are already filtered
        
    Returns:
        (is_valid, error_message)
    """
    weekday = requested_dt.weekday()
    requested_time = requested_dt.time()
    
    # If slots are already day-filtered, check all slots. Otherwise filter by weekday
    relevant_slots = available_time_slots
    if target_day is None:
        # Full week slots - filter by weekday
        relevant_slots = [slot for slot in available_time_slots 
                         if slot.get('day_of_week') == weekday and slot.get('is_active', True)]
    
    # Find matching slots for this time
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
    
    # If we get here, no matching slot was found
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = weekday_names[weekday]
    
    # Find available slots for this day to suggest alternatives
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




# ---------------- Agent functions ---------------- #

async def check_availability(params):
    """Check if a time slot is free in Google Calendar with predefined slot validation."""
    
    # Get cached clinic data
    clinic_data = params.get("_clinic_data")
    if not clinic_data:
        error_msg = "CRITICAL: Clinic data not available in cache - check main.py initialization"
        print(f"{error_msg}")
        return {"error": error_msg}
    
    # Get required parameters
    user_input = params.get("time_input")
    if not user_input:
        return {"error": "time_input is required"}
    
    caller_phone = params.get("caller_phone")
    
    # Use day-specific cached data
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
    
    # Handle customer data if needed (uses customer cache)
    customer_info = None
    if caller_phone:
        try:
            customer_info = await customer_cache.get_or_create_customer(caller_phone)
        except Exception as e:
            print(f"Customer lookup failed for {caller_phone}: {e}")
            # Continue without customer info - not critical for availability check

    service = get_service()
    norm = normalise_time_input(user_input, tz)
    
    # Check if parsing failed
    if "error" in norm:
        return norm

    # Parse the start datetime for validation
    start_dt = datetime.datetime.fromisoformat(norm["start"]["dateTime"])
    
    # Step 1: Validate against predefined database slots FIRST (day-specific)
    available_slots = clinic_data["available_time_slots"]
    slot_valid, slot_error = validate_requested_time_against_slots(start_dt, available_slots, target_day)
    if not slot_valid:
        return {
            "status": "invalid_slot",
            "message": slot_error
        }
    
    # Step 2: Validate working hours (redundant check but kept for consistency)
    is_valid, hours_error = is_within_working_hours(start_dt, working_hours, target_day)
    if not is_valid:
        return {
            "status": "outside_hours", 
            "message": hours_error,
            "working_hours": working_hours
        }

    # Calculate the actual end time based on slot duration
    slot_end_dt = start_dt + datetime.timedelta(minutes=slot_duration)
    
    # Check Google Calendar availability for the full slot duration
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
        
        # Check for conflicts with proper overlap detection
        for busy in busy_times:
            busy_start = datetime.datetime.fromisoformat(busy["start"])
            busy_end = datetime.datetime.fromisoformat(busy["end"])
            
            # Check if the requested slot overlaps with any busy period
            if (start_dt < busy_end and slot_end_dt > busy_start):
                return {"status": "busy", "message": "Not available.", "busy": busy_times}
        
        return {"status": "available", "message": "Yes, available."}
        
    except Exception as e:
        return {"error": f"Error checking calendar: {str(e)}"}
    

async def get_available_slots(params):
    """Get all available appointment slots for a specified date."""
    
    # Get cached clinic data
    clinic_data = params.get("_clinic_data")
    if not clinic_data:
        error_msg = "CRITICAL: Clinic data not available in cache - check main.py initialization"
        print(f"{error_msg}")
        return {"error": error_msg}
    
    # Get required parameters
    date_input = params.get("date_input")
    if not date_input:
        return {"error": "date_input is required"}
    
    caller_phone = params.get("caller_phone")
    
    # Use day-specific cached data
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
    
    # Handle customer data if needed (uses customer cache)
    customer_info = None
    if caller_phone:
        try:
            customer_info = await customer_cache.get_or_create_customer(caller_phone)
        except Exception as e:
            print(f"Customer lookup failed for {caller_phone}: {e}")

    service = get_service()

    # Parse the date input
    try:
        if isinstance(date_input, str):
            # Try to parse the date
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

    # Get predefined time slots for the date from database (already day-filtered if target_day exists)
    available_slots_data = clinic_data["available_time_slots"]
    
    if target_day is not None and target_date.weekday() == target_day:
        # Slots are already filtered for the target day
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
        # Fallback to original method
        time_slots = get_predefined_slots_for_date(target_date, available_slots_data, tz)
    
    if not time_slots:
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = weekday_names[target_date.weekday()]
        return {
            "status": "closed",
            "message": f"No appointment slots available on {day_name}s",
            "available_slots": []
        }

    # Check each slot against Google Calendar
    tzinfo = ZoneInfo(tz)
    available_slots = []
    
    # Batch the freebusy query for efficiency
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
        
        # Convert busy times to datetime objects for comparison
        busy_periods = []
        for busy in busy_times:
            start = datetime.datetime.fromisoformat(busy["start"])
            end = datetime.datetime.fromisoformat(busy["end"])
            busy_periods.append((start, end))
        
        # Check each time slot
        for slot in time_slots:
            slot_end = slot + datetime.timedelta(minutes=slot_duration)
            
            # Check if this slot conflicts with any busy period
            is_free = True
            for busy_start, busy_end in busy_periods:
                # Check for overlap
                if (slot < busy_end and slot_end > busy_start):
                    is_free = False
                    break
            
            if is_free:
                available_slots.append({
                    "start_time": slot.strftime("%H:%M"),
                    "start_datetime": slot.isoformat(),
                    "duration_minutes": slot_duration
                })
        
        # Format response
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
    """Create a calendar event if the slot is free and within working hours."""
    
    # Get cached clinic data
    clinic_data = params.get("_clinic_data")
    if not clinic_data:
        error_msg = "CRITICAL: Clinic data not available in cache - check main.py initialization"
        print(f"{error_msg}")
        return {"error": error_msg}
    
    # Get required parameters
    time_input = params.get("time_input")
    if not time_input:
        return {"error": "time_input is required"}
        
    name = params.get("name", "Unknown")
    caller_phone = params.get("caller_phone")
    description = params.get("description")
    attendees = params.get("attendees", [])
    location = params.get("location")
    send_updates = params.get("send_updates", "none")
    
    # Use day-specific cached data
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
    
    # Handle customer data (uses customer cache)
    customer_info = None
    if caller_phone:
        try:
            customer_info = await customer_cache.get_or_create_customer(caller_phone)
        except Exception as e:
            print(f"Customer lookup failed for {caller_phone}: {e}")

    service = get_service()

    # Use customer info for event details if available
    if customer_info:
        first_name = customer_info.get('first_name', '')
        last_name = customer_info.get('last_name', '')
        if first_name or last_name:
            name = f"{first_name} {last_name}".strip()
        
        # Add customer email to attendees if available
        if customer_info.get('email') and customer_info['email'] not in attendees:
            attendees.append(customer_info['email'])

    # Normalise time input using clinic-specific slot duration
    if isinstance(time_input, str):
        norm = normalise_time_input(time_input, tz, slot_duration)
    elif isinstance(time_input, dict):
        norm = time_input
    else:
        return {"error": "time_input must be str or normalised dict"}

    # Check if parsing failed
    if "error" in norm:
        return norm

    # Parse the start datetime for validation
    start_dt = datetime.datetime.fromisoformat(norm["start"]["dateTime"])
    
    # Step 1: Validate against predefined database slots FIRST (day-specific)
    available_slots = clinic_data["available_time_slots"]
    slot_valid, slot_error = validate_requested_time_against_slots(start_dt, available_slots, target_day)
    if not slot_valid:
        return {
            "status": "invalid_slot",
            "message": slot_error
        }
    
    # Step 2: Validate working hours (redundant check but kept for consistency)
    is_valid, hours_error = is_within_working_hours(start_dt, working_hours, target_day)
    if not is_valid:
        return {
            "status": "outside_hours", 
            "message": hours_error,
            "working_hours": working_hours
        }

    # Check availability
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

        # Build event with clinic and customer info
        event_summary = f"Appointment with {name}"
        if clinic_data.get('clinic_name'):
            event_summary = f"{clinic_data['clinic_name']} - Appointment with {name}"
        
        event_body = {
            "summary": event_summary,
            "start": norm["start"],
            "end": norm["end"],
        }
        
        # Build description with clinic and customer details
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

        # Create the calendar event
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


async def get_clinic_hours(params):
    """Get clinic opening hours for a specific date from cached data."""
    
    # MANDATORY: Get cached clinic data (NO DB CALLS)
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
        # Parse date input to get target day of week
        normalized = normalise_time_input(date_input)
        
        # Extract the start datetime and convert to day of week
        target_date = datetime.datetime.fromisoformat(normalized["start"]["dateTime"])
        
        # Convert to day of week (0=Monday, 1=Tuesday... 6=Sunday)
        target_day_of_week = target_date.weekday()
            
    except Exception as date_error:
        return {"error": f"Could not parse date input '{date_input}': {str(date_error)}"}
    
    # Get specific day's hours from cached data (may already be day-filtered)
    raw_operating_hours = clinic_data["raw_operating_hours"]
    
    if target_day is not None and len(raw_operating_hours) == 1:
        # Day-specific data - use the single record
        day_hours = raw_operating_hours[0]
        if day_hours['day_of_week'] != target_day_of_week:
            print(f"Warning: Day mismatch - cached data for day {day_hours['day_of_week']}, requested day {target_day_of_week}")
    else:
        # Full week data - find the specific day's data
        day_hours = None
        for hour_record in raw_operating_hours:
            if hour_record['day_of_week'] == target_day_of_week:
                day_hours = hour_record
                break
    
    if not day_hours:
        return {"error": f"No opening hours found for day {target_day_of_week} at {clinic_name}"}
    
    # Format the opening hours for display
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


# ---------------- Function Map ---------------- #

FUNCTION_MAP = {
    "check_availability": check_availability,
    "create_appointment": create_appointment,
    "get_available_slots": get_available_slots,
    "get_clinic_hours": get_clinic_hours,
}