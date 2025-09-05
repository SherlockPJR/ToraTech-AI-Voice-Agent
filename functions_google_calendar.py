import os
import datetime
import asyncio
import re
from dateutil import parser as date_parser
import dateparser
from zoneinfo import ZoneInfo
from recurrent.event_parser import RecurringEvent
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from supabase import create_client, Client
from typing import Optional, Dict

load_dotenv()

# Google Calendar API scopes
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Working hours configuration - [Monday=0 to Sunday=6]
# Format: weekday: [(start_hour, start_min, end_hour, end_min), ...]
WORKING_HOURS = {
    0: [(10, 0, 14, 0), (14, 45, 19, 0)],  # Monday: 10:00-14:00 & 14:45-19:00
    1: [(10, 0, 14, 0), (14, 45, 19, 0)],  # Tuesday
    2: [(10, 0, 14, 0), (14, 45, 19, 0)],  # Wednesday
    3: [(10, 0, 14, 0), (14, 45, 19, 0)],  # Thursday
    4: [(10, 0, 14, 0), (14, 45, 19, 0)],  # Friday
    5: [(10, 0, 16, 0)],                   # Saturday: 10:00-16:00
    6: []                                  # Sunday: closed
}

async def get_clinic_slot_duration(clinic_id: str) -> int:
    """Get clinic-specific slot duration from Supabase, fallback to 60 minutes"""
    try:
        supabase = init_supabase()
        result = supabase.table("clinics_available_time_slots").select("slot_duration_minutes").eq("clinic_id", clinic_id).limit(1).execute()
        
        if result.data and result.data[0].get('slot_duration_minutes'):
            return result.data[0]['slot_duration_minutes']
        
        # Fallback to default 60 minutes if not found
        return 60
        
    except Exception:
        # Fallback to default on error
        return 60

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


def init_supabase() -> Client:
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url:
        raise Exception("SUPABASE_URL not found.")
    
    if not supabase_key:
        raise Exception("SUPABASE_URL not found.")
    
    return create_client(supabase_url, supabase_key)

async def get_clinic_by_phone(phone_number: str) -> Optional[Dict]:
    """Get clinic information from Supabase by voice_agent_phone_number"""
    try:
        supabase = init_supabase()
        
        # Use LIKE pattern to handle whitespace issues (like trailing newlines)
        result = supabase.table("clinics").select("*").like("voice_agent_phone_number", f"%{phone_number}%").execute()
        
        # If that doesn't work, try exact match
        if not result.data:
            result = supabase.table("clinics").select("*").eq("voice_agent_phone_number", phone_number).execute()
        
        return result.data[0] if result.data else None
    except Exception:
        return None

async def get_customer_by_phone(phone_number: str) -> Optional[Dict]:
    """Get customer information from Supabase by phone_number"""
    try:
        supabase = init_supabase()
        result = supabase.table("customers").select("*").eq("phone_number", phone_number).execute()
        return result.data[0] if result.data else None
    except Exception:
        return None

async def get_or_create_customer_by_phone(phone_number: str) -> Optional[Dict]:
    """Get customer by phone, create if doesn't exist"""
    try:
        # First try to get existing customer
        customer = await get_customer_by_phone(phone_number)
        if customer:
            return customer
        
        # Create new customer if not found
        supabase = init_supabase()
        new_customer_data = {
            "phone_number": phone_number,
            "created_at": datetime.now().isoformat()
        }
        
        result = supabase.table("customers").insert(new_customer_data).execute()
        return result.data[0] if result.data else None
        
    except Exception as e:
        # Log error but don't fail the entire operation
        print(f"Error creating customer: {e}")
        return None

async def get_clinic_working_hours(clinic_id: str) -> Dict:
    """Get clinic operating hours from Supabase"""
    try:
        supabase = init_supabase()
        result = supabase.table("clinics_operating_hours").select("*").eq("clinic_id", clinic_id).execute()
        
        # Convert to the WORKING_HOURS format
        working_hours = {}
        for row in result.data:
            day = row['day_of_week']
            if row['is_closed']:
                working_hours[day] = []
            else:
                # Convert time objects to (hour, minute, hour, minute) tuples
                open_time = row['open_time']
                close_time = row['close_time']
                
                if open_time and close_time:
                    # Parse time strings like "10:00:00"
                    open_h, open_m = map(int, open_time.split(':')[:2])
                    close_h, close_m = map(int, close_time.split(':')[:2])
                    
                    # Handle break times if they exist
                    time_slots = []
                    if row['break_start_time'] and row['break_end_time']:
                        break_start_h, break_start_m = map(int, row['break_start_time'].split(':')[:2])
                        break_end_h, break_end_m = map(int, row['break_end_time'].split(':')[:2])
                        
                        # Split into before and after break
                        time_slots.append((open_h, open_m, break_start_h, break_start_m))
                        time_slots.append((break_end_h, break_end_m, close_h, close_m))
                    else:
                        time_slots.append((open_h, open_m, close_h, close_m))
                    
                    working_hours[day] = time_slots
                else:
                    working_hours[day] = []
        
        return working_hours
    except Exception:
        # Return default working hours if database query fails
        return WORKING_HOURS


def get_service():
    """Build a Google Calendar API service object."""
    creds = get_credentials()
    return build("calendar", "v3", credentials=creds)


def is_within_working_hours(dt: datetime.datetime, working_hours: Dict) -> Tuple[bool, str]:
    """
    Check if datetime falls within working hours.
    
    Returns:
        (is_valid, error_message)
    """
    weekday = dt.weekday()
    time_ranges = working_hours.get(weekday, [])
    
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


def generate_time_slots(date: datetime.date, 
                       working_hours: Dict,
                       slot_duration_minutes: int,
                       tz: str = "America/Chicago") -> List[datetime.datetime]:
    """
    Generate all possible appointment slots for a given date within working hours.
    
    Args:
        date: The date to generate slots for
        working_hours: Working hours configuration
        slot_duration_minutes: Duration of each slot in minutes
        tz: Timezone string
        
    Returns:
        List of datetime objects representing available time slots
    """
    tzinfo = ZoneInfo(tz)
    slots = []
    
    weekday = date.weekday()
    time_ranges = working_hours.get(weekday, [])
    
    if not time_ranges:
        return slots  # Closed day
    
    for start_h, start_m, end_h, end_m in time_ranges:
        # Create datetime objects for the start and end of this working period
        period_start = datetime.datetime.combine(date, datetime.time(start_h, start_m), tzinfo)
        period_end = datetime.datetime.combine(date, datetime.time(end_h, end_m), tzinfo)
        
        # Generate slots within this period
        current_slot = period_start
        while current_slot + datetime.timedelta(minutes=slot_duration_minutes) <= period_end:
            slots.append(current_slot)
            current_slot += datetime.timedelta(minutes=slot_duration_minutes)
    
    return slots


def normalise_time_input(user_input: str,
                        tz: str = "America/Chicago",
                        default_duration_minutes: int = 60,
                        now: Optional[datetime.datetime] = None) -> Dict:
    """
    Comprehensive time normalisation function for Google Calendar API.
    
    Handles:
    - Single times: "Sept 1 at 3pm", "tomorrow at 2pm", "next Monday at 9am"
    - Time ranges: "from 2pm to 4pm", "between Sept 1 and Sept 5"
    - Relative ranges: "from next Monday to Friday", "next week"
    - Time of day: "Monday morning", "Friday afternoon"
    - Recurring events: "every Tuesday at 2pm", "daily at 9am"
    - Weekend expressions: "this weekend", "next weekend"
    
    Returns:
    {
        "start": {"dateTime": ISO_STRING, "timeZone": TIMEZONE},
        "end": {"dateTime": ISO_STRING, "timeZone": TIMEZONE},
        "recurrence": ["RRULE_STRING"],  # Optional, for recurring events
        "type": "single|range|recurring|time_of_day",
        "error": "ERROR_MESSAGE",  # If parsing failed
        "suggestion": "HELPFUL_SUGGESTION"  # If parsing failed
    }
    """
    
    # Initialize timezone and current time
    tzinfo = ZoneInfo(tz)
    now = now or datetime.datetime.now(tzinfo)
    raw = user_input.strip()
    low = raw.lower().strip()
    
    # Define connector patterns for range detection
    CONNECTORS_PATTERN = r'\s+(?:to|and|–|—|-|through|thru|until|till)\s+'
    
    # Helper functions
    def to_iso(dt: datetime.datetime) -> str:
        """Convert datetime to timezone-aware ISO string."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        return dt.astimezone(tzinfo).isoformat()
    
    def get_weekday_number(day_str: str) -> Optional[int]:
        """Get weekday number from day name (0=Monday, 6=Sunday)."""
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6,
            'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6
        }
        day_lower = day_str.lower().strip()
        for day_name, day_num in weekdays.items():
            if day_name in day_lower:
                return day_num
        return None
    
    def get_next_weekday(base_date: datetime.datetime, target_weekday: int, 
                        force_next_week: bool = False) -> datetime.datetime:
        """Get next occurrence of target weekday."""
        days_ahead = target_weekday - base_date.weekday()
        if days_ahead < 0 or (days_ahead == 0 and force_next_week):
            days_ahead += 7
        elif days_ahead == 0 and base_date.hour >= 12:
            # If it's the same day but past noon, assume next week
            days_ahead += 7
        return base_date + datetime.timedelta(days=days_ahead)
    
    def parse_time_of_day(text: str, base_date: datetime.datetime) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime]]:
        """Parse time-of-day expressions like 'morning', 'afternoon'."""
        text_lower = text.lower()
        time_ranges = {
            'morning': (9, 0, 12, 0),      # 9:00 AM to 12:00 PM
            'afternoon': (12, 0, 17, 0),   # 12:00 PM to 5:00 PM
            'evening': (17, 0, 21, 0),     # 5:00 PM to 9:00 PM
            'night': (21, 0, 23, 59),      # 9:00 PM to 11:59 PM
        }
        
        for period, (start_h, start_m, end_h, end_m) in time_ranges.items():
            if period in text_lower:
                start_dt = base_date.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
                end_dt = base_date.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
                return start_dt, end_dt
        return None, None
    
    def parse_time_only(text: str, base_date: datetime.datetime) -> Optional[datetime.datetime]:
        """Parse time-only expressions like '2pm', '14:30'."""
        text = text.strip()
        
        # Pattern for times like "2pm", "2:30pm", "14:30"
        time_patterns = [
            (r'^(\d{1,2}):(\d{2})\s*(am|pm)?$', lambda m: (int(m[1]), int(m[2]), m[3])),
            (r'^(\d{1,2})\s*(am|pm)$', lambda m: (int(m[1]), 0, m[2])),
            (r'^(\d{1,2})\.(\d{2})\s*(am|pm)?$', lambda m: (int(m[1]), int(m[2]), m[3])),
        ]
        
        for pattern, extractor in time_patterns:
            match = re.match(pattern, text.lower())
            if match:
                hour, minute, period = extractor(match)
                
                # Handle AM/PM
                if period:
                    if period == 'pm' and hour != 12:
                        hour += 12
                    elif period == 'am' and hour == 12:
                        hour = 0
                
                # Validate hour and minute
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return None
    
    def parse_relative_weekday(text: str, base: datetime.datetime) -> Optional[datetime.datetime]:
        """Parse expressions like 'next Monday', 'this Friday'."""
        text_lower = text.lower()
        
        # Check for weekday
        weekday_num = get_weekday_number(text_lower)
        if weekday_num is None:
            return None
        
        # Check for relative modifiers
        if 'next' in text_lower:
            # "next Monday" - always next week's occurrence
            result = get_next_weekday(base, weekday_num, force_next_week=True)
        elif 'this' in text_lower:
            # "this Monday" - this week's occurrence (or next if already passed)
            result = get_next_weekday(base, weekday_num, force_next_week=False)
        else:
            # Just "Monday" - next occurrence
            result = get_next_weekday(base, weekday_num, force_next_week=False)
        
        # Check for time of day
        time_match = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))', text_lower)
        if time_match:
            time_part = parse_time_only(time_match.group(1), result)
            if time_part:
                result = time_part
        else:
            # Check for morning/afternoon/evening
            start_dt, end_dt = parse_time_of_day(text_lower, result)
            if start_dt:
                return start_dt
        
        # Check if we need to extract a specific time from the text
        at_match = re.search(r'at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', text_lower)
        if at_match:
            time_part = parse_time_only(at_match.group(1), result)
            if time_part:
                result = time_part
        
        return result
    
    def parse_weekend(text: str, base: datetime.datetime) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime]]:
        """Parse weekend expressions."""
        text_lower = text.lower()
        
        if 'weekend' in text_lower:
            # Determine which weekend
            if 'next' in text_lower:
                # Next weekend
                days_to_saturday = (5 - base.weekday()) % 7
                if days_to_saturday == 0:  # Today is Saturday
                    days_to_saturday = 7
                saturday = base + datetime.timedelta(days=days_to_saturday)
            elif 'this' in text_lower or 'weekend' in text_lower:
                # This weekend (or just "weekend")
                days_to_saturday = (5 - base.weekday()) % 7
                if days_to_saturday == 0 and base.weekday() == 6:  # Today is Sunday
                    days_to_saturday = 6  # Next Saturday
                saturday = base + datetime.timedelta(days=days_to_saturday)
            else:
                return None, None
            
            # Weekend is Saturday to Sunday
            start_dt = saturday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = (saturday + datetime.timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
            return start_dt, end_dt
        
        return None, None
    
    def safe_dateparser_parse(text: str, base: datetime.datetime) -> Optional[datetime.datetime]:
        """Safely parse date with dateparser, handling common failures."""
        try:
            # First try our custom parsers for better accuracy
            
            # Try relative weekday parsing
            result = parse_relative_weekday(text, base)
            if result:
                return result
            
            # Try time-only parsing
            result = parse_time_only(text, base)
            if result:
                return result
            
            # Fall back to dateparser
            result = dateparser.parse(text, settings={
                "RELATIVE_BASE": base,
                "RETURN_AS_TIMEZONE_AWARE": True,
                "TIMEZONE": tz,
                "PREFER_DATES_FROM": "future",
                "PREFER_DAY_OF_MONTH": "first"
            })
            
            return result
        except Exception:
            return None
    
    # Error response helper
    def error_response(message: str, suggestion: str = "") -> Dict:
        return {
            "error": message,
            "suggestion": suggestion or "Try formats like: 'tomorrow at 2pm', 'next Monday 9am to 5pm', 'September 1st at 3pm'"
        }
    
    try:
        # 1: Handle recurring expressions first
        try:
            r = RecurringEvent(now_date=now)
            rec_parsed = r.parse(raw)
            if rec_parsed and r.is_recurring:
                rrule_text = r.get_RFC_rrule()
                
                # Try to extract a better start time
                # Look for time patterns in the original input
                time_match = re.search(r'at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))', low)
                if time_match:
                    start_dt = parse_time_only(time_match.group(1), now) or now
                else:
                    start_dt = now
                
                end_dt = start_dt + datetime.timedelta(minutes=default_duration_minutes)
                
                return {
                    "start": {"dateTime": to_iso(start_dt), "timeZone": tz},
                    "end": {"dateTime": to_iso(end_dt), "timeZone": tz},
                    "recurrence": [rrule_text],
                    "type": "recurring"
                }
        except Exception:
            pass  # Continue to other parsing methods
        
        # 2: Handle weekend expressions
        weekend_start, weekend_end = parse_weekend(low, now)
        if weekend_start and weekend_end:
            return {
                "start": {"dateTime": to_iso(weekend_start), "timeZone": tz},
                "end": {"dateTime": to_iso(weekend_end), "timeZone": tz},
                "type": "weekend"
            }
        
        # 3: Handle explicit date ranges
        range_keywords = ["between", "from", " to ", " and ", "–", "—", "-", " through ", " thru ", " until ", " till "]
        has_range = any(kw in low for kw in range_keywords)
        
        if has_range:
            # Clean input and split into parts
            cleaned = re.sub(r"^\s*(?:between|from)\s+", "", raw, flags=re.I)
            parts = re.split(CONNECTORS_PATTERN, cleaned, maxsplit=1, flags=re.I)
            
            if len(parts) == 2:
                start_str, end_str = [part.strip() for part in parts]
                
                # Parse start time
                start_dt = safe_dateparser_parse(start_str, now)
                if not start_dt:
                    return error_response(f"Could not understand start time: '{start_str}'")
                
                # Parse end time with logic
                end_dt = None
                
                # Method 1: Try parsing end as time-only (for same-day ranges)
                end_time_only = parse_time_only(end_str, start_dt)
                if end_time_only:
                    end_dt = end_time_only
                
                # Method 2: Try parsing end relative to start_dt
                if not end_dt:
                    end_dt = safe_dateparser_parse(end_str, start_dt)
                
                # Method 3: If it's a weekday, use weekday logic
                if not end_dt or end_dt <= start_dt:
                    end_weekday = get_weekday_number(end_str)
                    if end_weekday is not None:
                        if "next" in end_str.lower():
                            # Explicitly next week
                            end_dt = get_next_weekday(start_dt, end_weekday, force_next_week=True)
                        else:
                            # Same week if possible
                            if end_weekday >= start_dt.weekday():
                                days_to_add = end_weekday - start_dt.weekday()
                                end_dt = start_dt + datetime.timedelta(days=days_to_add)
                            else:
                                # Next week
                                end_dt = get_next_weekday(start_dt, end_weekday)
                
                # Method 4: Parse relative to 'now' as last resort
                if not end_dt or end_dt <= start_dt:
                    end_dt = safe_dateparser_parse(end_str, now)
                
                # Validate we have a valid end time
                if not end_dt:
                    return error_response(f"Could not understand end time: '{end_str}'",
                                        "Try 'from Monday to Friday' or 'from 2pm to 4pm'")
                
                if end_dt <= start_dt:
                    # For same-day time ranges, adjust the end date
                    if end_dt.time() < start_dt.time():
                        end_dt = end_dt.replace(
                            year=start_dt.year,
                            month=start_dt.month,
                            day=start_dt.day
                        )
                        # If still invalid, might be next day
                        if end_dt <= start_dt:
                            end_dt = end_dt + datetime.timedelta(days=1)
                
                # Determine if this is a date range (no specific times)
                has_specific_times = any(indicator in low for indicator in [":", "am", "pm", "hour", "minute"])
                if not has_specific_times and (end_dt - start_dt).days > 0:
                    # Treat as full day range
                    start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=0)
                
                return {
                    "start": {"dateTime": to_iso(start_dt), "timeZone": tz},
                    "end": {"dateTime": to_iso(end_dt), "timeZone": tz},
                    "type": "range"
                }
        
        # 4: Handle combined expressions (weekday + time of day)
        # Check for patterns like "Monday morning", "Friday afternoon"
        weekday_match = re.search(r'((?:next|this)?\s*(?:mon|tue|wed|thu|fri|sat|sun)\w*)\s*(morning|afternoon|evening|night)', low)
        if weekday_match:
            # Parse the weekday part
            weekday_dt = parse_relative_weekday(weekday_match.group(1), now)
            if weekday_dt:
                # Apply time of day
                start_dt, end_dt = parse_time_of_day(weekday_match.group(2), weekday_dt)
                if start_dt and end_dt:
                    return {
                        "start": {"dateTime": to_iso(start_dt), "timeZone": tz},
                        "end": {"dateTime": to_iso(end_dt), "timeZone": tz},
                        "type": "time_of_day"
                    }
        
        # 5: Handle single time expressions
        single_dt = safe_dateparser_parse(raw, now)
        if single_dt:
            # Check for time-of-day expressions without a specific time
            if any(period in low for period in ['morning', 'afternoon', 'evening', 'night']):
                start_dt, end_dt = parse_time_of_day(low, single_dt)
                if start_dt and end_dt:
                    return {
                        "start": {"dateTime": to_iso(start_dt), "timeZone": tz},
                        "end": {"dateTime": to_iso(end_dt), "timeZone": tz},
                        "type": "time_of_day"
                    }
            
            # Handle "next week" as special case
            if "next week" in low:
                # Get Monday of next week
                days_to_next_monday = (7 - now.weekday()) % 7 or 7
                week_start = (now + datetime.timedelta(days=days_to_next_monday)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                week_end = (week_start + datetime.timedelta(days=6)).replace(
                    hour=23, minute=59, second=59, microsecond=0
                )
                return {
                    "start": {"dateTime": to_iso(week_start), "timeZone": tz},
                    "end": {"dateTime": to_iso(week_end), "timeZone": tz},
                    "type": "week_range"
                }
            
            # Default: single time point with default duration
            end_dt = single_dt + datetime.timedelta(minutes=default_duration_minutes)
            return {
                "start": {"dateTime": to_iso(single_dt), "timeZone": tz},
                "end": {"dateTime": to_iso(end_dt), "timeZone": tz},
                "type": "single"
            }
        
        # 6: If all parsing failed
        return error_response(f"Could not parse time expression: '{raw}'")
        
    except Exception as e:
        return error_response(f"Unexpected error parsing time: '{raw}'", f"Error details: {str(e)}")


# ---------------- Agent functions ---------------- #

async def check_availability(params):
    """Check if a time slot is free in Google Calendar with working hours validation."""
    service = get_service()
    user_input = params.get("time_input")
    caller_phone = params.get("caller_phone")
    called_phone = params.get("called_phone")
    calendar_id = params.get("calendar_id", "primary")
    tz = params.get("tz", "America/Chicago")

    if not user_input:
        return {"error": "time_input is required"}
    
    if not called_phone:
        return {"error": "called_phone is required"}
    
    # Look up clinic by called phone number
    clinic_info = await get_clinic_by_phone(called_phone)
    if not clinic_info:
        return {"error": f"No clinic found for phone number: {called_phone}"}
    
    clinic_id = clinic_info.get('clinic_id')
    
    # Get clinic-specific slot duration from database
    slot_duration = await get_clinic_slot_duration(clinic_id)
    
    # Look up or create customer by caller phone number (if provided)
    customer_info = None
    if caller_phone:
        customer_info = await get_or_create_customer_by_phone(caller_phone)
    
    # Get clinic-specific working hours, fallback to default
    working_hours = await get_clinic_working_hours(clinic_id)
    
    # Use clinic timezone if available
    if clinic_info.get('timezone'):
        tz = clinic_info['timezone']
    
    # Use clinic calendar if available
    if clinic_info.get('calendar_id'):
        calendar_id = clinic_info['calendar_id']

    norm = normalise_time_input(user_input, tz)
    
    # Check if parsing failed
    if "error" in norm:
        return norm

    # Parse the start datetime for working hours validation
    start_dt = datetime.datetime.fromisoformat(norm["start"]["dateTime"])
    
    # Validate working hours using clinic-specific hours
    is_valid, hours_error = is_within_working_hours(start_dt, working_hours)
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
    service = get_service()
    date_input = params.get("date_input")
    caller_phone = params.get("caller_phone")
    called_phone = params.get("called_phone")
    calendar_id = params.get("calendar_id", "primary")
    tz = params.get("tz", "America/Chicago")

    if not date_input:
        return {"error": "date_input is required"}
    
    if not called_phone:
        return {"error": "called_phone is required"}
    
    # Look up clinic by called phone number
    clinic_info = await get_clinic_by_phone(called_phone)
    if not clinic_info:
        return {"error": f"No clinic found for phone number: {called_phone}"}
    
    clinic_id = clinic_info.get('clinic_id')
    
    # Get clinic-specific slot duration from database
    slot_duration = await get_clinic_slot_duration(clinic_id)
    
    # Look up or create customer by caller phone number (if provided)
    customer_info = None
    if caller_phone:
        customer_info = await get_or_create_customer_by_phone(caller_phone)
    
    # Get clinic-specific working hours
    working_hours = await get_clinic_working_hours(clinic_id)
    
    # Use clinic timezone if available
    if clinic_info.get('timezone'):
        tz = clinic_info['timezone']
    
    # Use clinic calendar if available
    if clinic_info.get('calendar_id'):
        calendar_id = clinic_info['calendar_id']

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

    # Generate all possible time slots for the date using clinic-specific working hours
    time_slots = generate_time_slots(target_date, working_hours, slot_duration, tz)
    
    if not time_slots:
        return {
            "status": "closed",
            "message": "We are closed on this day",
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
    service = get_service()
    time_input = params.get("time_input")
    name = params.get("name", "Unknown")
    caller_phone = params.get("caller_phone")
    called_phone = params.get("called_phone")
    calendar_id = params.get("calendar_id", "primary")
    tz = params.get("tz", "America/Chicago")
    description = params.get("description")
    attendees = params.get("attendees", [])
    location = params.get("location")
    send_updates = params.get("send_updates", "none")

    if not time_input:
        return {"error": "time_input is required"}
    
    if not called_phone:
        return {"error": "called_phone is required"}
    
    # Look up clinic by called phone number
    clinic_info = await get_clinic_by_phone(called_phone)
    if not clinic_info:
        return {"error": f"No clinic found for phone number: {called_phone}"}
    
    clinic_id = clinic_info.get('clinic_id')
    
    # Look up or create customer by caller phone number (if provided)
    customer_info = None
    if caller_phone:
        customer_info = await get_or_create_customer_by_phone(caller_phone)
    working_hours = await get_clinic_working_hours(clinic_id)
    
    # Use clinic timezone if available
    if clinic_info.get('timezone'):
        tz = clinic_info['timezone']
    
    # Use clinic calendar if available
    if clinic_info.get('calendar_id'):
        calendar_id = clinic_info['calendar_id']
    
    # Use clinic location if not provided and available
    if not location and clinic_info.get('address'):
        location = clinic_info['address']

    # Use customer info for event details if available
    if customer_info:
        first_name = customer_info.get('first_name', '')
        last_name = customer_info.get('last_name', '')
        if first_name or last_name:
            name = f"{first_name} {last_name}".strip()
        
        # Add customer email to attendees if available
        if customer_info.get('email') and customer_info['email'] not in attendees:
            attendees.append(customer_info['email'])

    # Normalise time input
    if isinstance(time_input, str):
        norm = normalise_time_input(time_input, tz)
    elif isinstance(time_input, dict):
        norm = time_input
    else:
        return {"error": "time_input must be str or normalised dict"}

    # Check if parsing failed
    if "error" in norm:
        return norm

    # Parse the start datetime for working hours validation
    start_dt = datetime.datetime.fromisoformat(norm["start"]["dateTime"])
    
    # Validate working hours using clinic-specific hours
    is_valid, hours_error = is_within_working_hours(start_dt, working_hours)
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
        if clinic_info.get('clinic_name'):
            event_summary = f"{clinic_info['clinic_name']} - Appointment with {name}"
        
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
        
        if clinic_info.get('phone_number'):
            event_description_parts.append(f"Clinic Phone: {clinic_info['phone_number']}")
        
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
                "clinic_name": clinic_info.get('clinic_name', 'Unknown Clinic'),
                "datetime": start_dt.isoformat(),
                "duration_minutes": (datetime.datetime.fromisoformat(norm["end"]["dateTime"]) - start_dt).seconds // 60
            }
        }
        
    except Exception as e:
        return {"error": f"Error creating appointment: {str(e)}"}


async def get_clinic_hours(params):
    """Get clinic opening hours for a specific date from Supabase."""
    caller_phone = params.get("caller_phone")
    called_phone = params.get("called_phone")
    date_input = params.get("date_input", "today")  # Default to today if not specified
    
    if not called_phone:
        return {"error": "called_phone is required"}
    
    try:
        # Look up clinic by called phone number
        clinic_info = await get_clinic_by_phone(called_phone)
        if not clinic_info:
            return {"error": f"No clinic found for phone number: {called_phone}"}
        
        clinic_id = clinic_info.get('clinic_id')
        
        # Parse date input to get target day of week using normalise_time_input
        try:
            # Use the existing normalise_time_input function for consistent date parsing
            normalized = normalise_time_input(date_input)
            
            # Extract the start datetime and convert to day of week
            import datetime
            target_date = datetime.datetime.fromisoformat(normalized["start"]["dateTime"])
            
            # Convert to day of week (0=Monday, 1=Tuesday... 6=Sunday)
            target_day_of_week = target_date.weekday()
            
        except Exception as date_error:
            return {"error": f"Could not parse date input '{date_input}': {str(date_error)}"}
        
        # Get specific day's hours from Supabase
        supabase = init_supabase()
        result = supabase.table("clinics_operating_hours").select("*").eq("clinic_id", clinic_id).eq("day_of_week", target_day_of_week).execute()
        
        if not result.data:
            return {"error": f"No opening hours found for this day at clinic"}
        
        # Get the specific day's data
        day_hours = result.data[0]  # Should only be one record per day per clinic
        
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
        
    except Exception as e:
        return {"error": f"Error retrieving clinic hours: {str(e)}"}


# ---------------- Function Map ---------------- #

FUNCTION_MAP = {
    "check_availability": check_availability,
    "create_appointment": create_appointment,
    "get_available_slots": get_available_slots,
    "get_clinic_hours": get_clinic_hours,
}