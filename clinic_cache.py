import asyncio
import datetime
import dateparser
from typing import Dict, Optional, Any
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv()

class ClinicCache:
    """
    Thread-safe clinic data cache that loads all clinic-related information once per call session.
    Uses clinic_id as primary key for efficient related data retrieval.
    NO DEFAULTS - fails explicitly if required data is missing.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def init_supabase(self) -> Client:
        """Initialize Supabase client"""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url:
            raise Exception("SUPABASE_URL not found in environment variables.")
        
        if not supabase_key:
            raise Exception("SUPABASE_KEY not found in environment variables.")
        
        return create_client(supabase_url, supabase_key)
    
    async def load_clinic_data(self, called_phone: str) -> Dict[str, Any]:
        """
        Load all clinic-related data for a phone number.
        Uses clinic_id as the primary key for efficient data retrieval from related tables.
        
        Args:
            called_phone: The phone number that was called (clinic's phone)
            
        Returns:
            Dict containing all clinic data
            
        Raises:
            Exception: If clinic not found or any required data is missing
        """
        async with self._lock:
            # Check if already cached
            if called_phone in self._cache:
                return self._cache[called_phone]
            
            supabase = self.init_supabase()
            
            # Step 1: Get clinic basic info by phone number to obtain clinic_id
            print(f"Loading clinic data for phone: {called_phone}")
            
            clinic_result = supabase.table("clinics").select("*").like("voice_agent_phone_number", f"%{called_phone}%").execute()
            
            if not clinic_result.data:
                # Try exact match as fallback
                clinic_result = supabase.table("clinics").select("*").eq("voice_agent_phone_number", called_phone).execute()
            
            if not clinic_result.data:
                raise Exception(f"No clinic found for phone number: {called_phone}")
            
            clinic_info = clinic_result.data[0]
            clinic_id = clinic_info.get('clinic_id')
            
            if not clinic_id:
                raise Exception(f"Clinic found but missing clinic_id for phone: {called_phone}")
            
            print(f"Found clinic_id: {clinic_id}")
            
            # Step 2: Use clinic_id as foreign key to get all related data efficiently
            
            # Get clinic operating hours
            hours_result = supabase.table("clinics_operating_hours").select("*").eq("clinic_id", clinic_id).execute()
            
            if not hours_result.data:
                raise Exception(f"No operating hours found for clinic_id: {clinic_id}")
            
            print(f"Found {len(hours_result.data)} operating hour records")
            
            # Get ALL clinic available time slots (not just duration)
            slots_result = supabase.table("clinics_available_time_slots").select("*").eq("clinic_id", clinic_id).execute()
            
            if not slots_result.data:
                raise Exception(f"No available time slots found for clinic_id: {clinic_id}")
            
            # Extract slot duration (should be consistent across all slots)
            slot_duration_minutes = None
            for slot in slots_result.data:
                if slot.get('slot_duration_minutes'):
                    slot_duration_minutes = slot['slot_duration_minutes']
                    break
            
            if slot_duration_minutes is None:
                raise Exception(f"No slot duration found in available time slots for clinic_id: {clinic_id}")
            
            print(f"Found {len(slots_result.data)} predefined time slots with {slot_duration_minutes}min duration")
            
            # Step 3: Validate required clinic fields (NO DEFAULTS)
            required_fields = {
                'clinic_name': 'Clinic name is required',
                'timezone': 'Timezone is required', 
                'calendar_id': 'Calendar ID is required'
            }
            
            for field, error_msg in required_fields.items():
                if not clinic_info.get(field):
                    raise Exception(f"{error_msg} for clinic_id: {clinic_id}")
            
            # Step 4: Process operating hours into the expected format
            working_hours = {}
            
            for row in hours_result.data:
                day = row['day_of_week']
                
                if row['is_closed']:
                    working_hours[day] = []
                else:
                    open_time = row['open_time']
                    close_time = row['close_time']
                    
                    if not open_time or not close_time:
                        raise Exception(f"Invalid operating hours for clinic_id {clinic_id}, day {day}: missing open_time or close_time")
                    
                    try:
                        # Parse time strings like "10:00:00"
                        open_h, open_m = map(int, open_time.split(':')[:2])
                        close_h, close_m = map(int, close_time.split(':')[:2])
                    except (ValueError, IndexError) as e:
                        raise Exception(f"Invalid time format for clinic_id {clinic_id}, day {day}: {e}")
                    
                    # Handle break times if they exist
                    time_slots = []
                    if row['break_start_time'] and row['break_end_time']:
                        try:
                            break_start_h, break_start_m = map(int, row['break_start_time'].split(':')[:2])
                            break_end_h, break_end_m = map(int, row['break_end_time'].split(':')[:2])
                            
                            # Split into before and after break
                            time_slots.append((open_h, open_m, break_start_h, break_start_m))
                            time_slots.append((break_end_h, break_end_m, close_h, close_m))
                        except (ValueError, IndexError) as e:
                            raise Exception(f"Invalid break time format for clinic_id {clinic_id}, day {day}: {e}")
                    else:
                        time_slots.append((open_h, open_m, close_h, close_m))
                    
                    working_hours[day] = time_slots
            
            # Ensure we have operating hours for all 7 days (0=Monday to 6=Sunday)
            for day in range(7):
                if day not in working_hours:
                    raise Exception(f"Missing operating hours for day {day} for clinic_id: {clinic_id}")
            
            # Step 5: Build comprehensive clinic data
            clinic_data = {
                # Basic clinic info (validated)
                'clinic_id': clinic_id,
                'clinic_name': clinic_info['clinic_name'],
                'phone_number': clinic_info.get('phone_number'),
                'voice_agent_phone_number': clinic_info.get('voice_agent_phone_number'),
                'address': clinic_info.get('address'),
                'timezone': clinic_info['timezone'],
                'calendar_id': clinic_info['calendar_id'],
                
                # Operating data (validated)
                'working_hours': working_hours,
                'slot_duration_minutes': slot_duration_minutes,
                'available_time_slots': slots_result.data,  # Predefined slots from DB
                
                # Raw data for reference
                'raw_clinic_info': clinic_info,
                'raw_operating_hours': hours_result.data,
                'raw_slot_info': slots_result.data
            }
            
            print(f"Successfully loaded clinic data for {clinic_info['clinic_name']}")
            
            # Cache the data
            self._cache[called_phone] = clinic_data
            return clinic_data
    
    def get_cached_data(self, called_phone: str) -> Optional[Dict[str, Any]]:
        """Get cached clinic data without async operations"""
        return self._cache.get(called_phone)
    
    def extract_day_from_time_input(self, time_input: str, timezone: str) -> Optional[int]:
        """Extract day of week (0-6, Monday=0) from time input string."""
        try:
            # Try to parse the time input to get the target date
            parsed_date = dateparser.parse(time_input, settings={
                "TIMEZONE": timezone,
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future"
            })
            
            if parsed_date:
                return parsed_date.weekday()  # 0=Monday, 6=Sunday
            else:
                return None
        except Exception as e:
            print(f"Warning: Could not extract day from '{time_input}': {e}")
            return None
    
    def get_day_specific_data(self, called_phone: str, time_input: str) -> Optional[Dict[str, Any]]:
        """Get clinic data filtered for a specific day based on time input."""
        clinic_data = self.get_cached_data(called_phone)
        if not clinic_data:
            return None
        
        timezone = clinic_data.get("timezone")
        if not timezone:
            print("Warning: No timezone found in clinic data")
            return None
        
        # Extract day from time input
        target_day = self.extract_day_from_time_input(time_input, timezone)
        if target_day is None:
            print(f"Warning: Could not determine day from time input: '{time_input}'")
            return None
        
        # Filter working hours for the specific day
        day_working_hours = {}
        if target_day in clinic_data["working_hours"]:
            day_working_hours[target_day] = clinic_data["working_hours"][target_day]
        
        # Filter available time slots for the specific day
        day_slots = [
            slot for slot in clinic_data["available_time_slots"]
            if slot.get("day_of_week") == target_day and slot.get("is_active", True)
        ]
        
        # Filter raw operating hours for the specific day
        day_raw_hours = [
            hour_record for hour_record in clinic_data["raw_operating_hours"]
            if hour_record.get("day_of_week") == target_day
        ]
        
        # Create day-specific clinic data
        day_specific_data = {
            # Basic clinic info (unchanged)
            'clinic_id': clinic_data['clinic_id'],
            'clinic_name': clinic_data['clinic_name'],
            'phone_number': clinic_data.get('phone_number'),
            'voice_agent_phone_number': clinic_data.get('voice_agent_phone_number'),
            'address': clinic_data.get('address'),
            'timezone': clinic_data['timezone'],
            'calendar_id': clinic_data['calendar_id'],
            'slot_duration_minutes': clinic_data['slot_duration_minutes'],
            
            # Day-specific data (filtered)
            'target_day_of_week': target_day,
            'working_hours': day_working_hours,
            'available_time_slots': day_slots,
            'raw_operating_hours': day_raw_hours,
            
            # Original data for reference (if needed)
            'full_clinic_data': clinic_data
        }
        
        return day_specific_data
    
    async def clear_cache(self, called_phone: str = None):
        """Clear cache for specific phone number or all entries"""
        async with self._lock:
            if called_phone:
                self._cache.pop(called_phone, None)
            else:
                self._cache.clear()
    
    async def preload_all_clinics(self):
        """
        Preload all clinic data for faster access.
        Useful for warming up the cache on server start.
        """
        supabase = self.init_supabase()
        
        # Get all clinic phone numbers
        clinics_result = supabase.table("clinics").select("voice_agent_phone_number").execute()
        
        if not clinics_result.data:
            print("No clinics found for preloading")
            return
        
        # Load data for each clinic
        tasks = []
        for clinic in clinics_result.data:
            phone = clinic.get('voice_agent_phone_number')
            if phone:
                tasks.append(self.load_clinic_data(phone.strip()))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful = sum(1 for result in results if not isinstance(result, Exception))
            failed = len(results) - successful
            
            print(f"Preload completed: {successful} successful, {failed} failed")
            
            # Log any failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    phone = clinics_result.data[i].get('voice_agent_phone_number')
                    print(f"Failed to preload clinic {phone}: {result}")

# Global clinic cache instance
clinic_cache = ClinicCache()

# Customer cache for frequently accessed customers
class CustomerCache:
    """Simple cache for customer data to reduce repeat lookups within a call session"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def init_supabase(self) -> Client:
        """Initialize Supabase client"""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise Exception("Supabase credentials not found in environment variables.")
        
        return create_client(supabase_url, supabase_key)
    
    async def get_or_create_customer(self, phone_number: str) -> Dict[str, Any]:
        """
        Get customer by phone, create if doesn't exist, with caching.
        
        Args:
            phone_number: Customer's phone number
            
        Returns:
            Dict containing customer data
            
        Raises:
            Exception: If database operation fails
        """
        async with self._lock:
            # Check cache first
            if phone_number in self._cache:
                return self._cache[phone_number]
            
            supabase = self.init_supabase()
            
            # Try to get existing customer
            result = supabase.table("customers").select("*").eq("phone_number", phone_number).execute()
            
            if result.data:
                customer = result.data[0]
            else:
                # Create new customer
                import datetime
                new_customer_data = {
                    "phone_number": phone_number,
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                create_result = supabase.table("customers").insert(new_customer_data).execute()
                
                if not create_result.data:
                    raise Exception(f"Failed to create customer for phone: {phone_number}")
                
                customer = create_result.data[0]
                print(f"Created new customer for phone: {phone_number}")
            
            # Cache the customer data
            self._cache[phone_number] = customer
            return customer
    
    async def clear_cache(self, phone_number: str = None):
        """Clear cache for specific phone number or all entries"""
        async with self._lock:
            if phone_number:
                self._cache.pop(phone_number, None)
            else:
                self._cache.clear()

# Global customer cache instance
customer_cache = CustomerCache()