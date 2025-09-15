import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()

class FAQCache:
    """Thread-safe cache for clinic FAQ data with intelligent keyword matching."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def init_supabase(self) -> Client:
        """Create authenticated Supabase client from environment variables."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url:
            raise Exception("SUPABASE_URL not found in environment variables.")
        
        if not supabase_key:
            raise Exception("SUPABASE_KEY not found in environment variables.")
        
        return create_client(supabase_url, supabase_key)
    
    async def load_clinic_faqs(self, clinic_id: str) -> Dict[str, Any]:
        """Load and cache all FAQ data for a specific clinic."""
        async with self._lock:
            # Return cached data if available
            if clinic_id in self._cache:
                return self._cache[clinic_id]
            
            supabase = self.init_supabase()
            
            print(f"Loading FAQ data for clinic_id: {clinic_id}")
            
            # Fetch FAQ categories for this clinic
            categories_result = supabase.table("clinics_faq_categories").select("*").eq("clinic_id", clinic_id).eq("is_active", True).order("display_order").execute()
            
            # Fetch all active FAQs for this clinic
            faqs_result = supabase.table("clinics_faqs").select("*").eq("clinic_id", clinic_id).eq("is_active", True).order("priority", desc=True).execute()
            
            if not faqs_result.data:
                print(f"No FAQs found for clinic_id: {clinic_id}")
                # Return empty structure for clinics without FAQs
                faq_data = {
                    'clinic_id': clinic_id,
                    'categories': [],
                    'faqs': [],
                    'keyword_index': {},
                    'question_index': {}
                }
            else:
                # Build keyword index for fast searching
                keyword_index = {}
                question_index = {}
                
                for faq in faqs_result.data:
                    faq_id = faq['id']
                    
                    # Index by keywords if available
                    if faq.get('keywords'):
                        for keyword in faq['keywords']:
                            keyword_lower = keyword.lower()
                            if keyword_lower not in keyword_index:
                                keyword_index[keyword_lower] = []
                            keyword_index[keyword_lower].append(faq_id)
                    
                    # Index by question words for fuzzy matching
                    question_words = self._extract_keywords_from_text(faq['question'])
                    for word in question_words:
                        if word not in question_index:
                            question_index[word] = []
                        question_index[word].append(faq_id)
                
                faq_data = {
                    'clinic_id': clinic_id,
                    'categories': categories_result.data,
                    'faqs': faqs_result.data,
                    'keyword_index': keyword_index,
                    'question_index': question_index
                }
                
                print(f"Successfully loaded {len(faqs_result.data)} FAQs for clinic {clinic_id}")
            
            # Store in cache for future requests
            self._cache[clinic_id] = faq_data
            return faq_data
    
    def get_cached_faqs(self, clinic_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached FAQ data synchronously."""
        return self._cache.get(clinic_id)
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract meaningful keywords from text for indexing."""
        # Remove common stop words and extract meaningful terms
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                     'to', 'was', 'with', 'you', 'your', 'do', 'does', 'can', 'will',
                     'should', 'would', 'could', 'i', 'we', 'they', 'my', 'our'}
        
        # Clean and split text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        return keywords
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def find_best_faq_match(self, question_text: str, clinic_id: str) -> Optional[Dict[str, Any]]:
        """Find the best matching FAQ for a given question."""
        faq_data = self.get_cached_faqs(clinic_id)
        if not faq_data or not faq_data['faqs']:
            return None
        
        question_lower = question_text.lower()
        question_keywords = self._extract_keywords_from_text(question_text)
        
        # Check for obviously unrelated keywords that should never match
        unrelated_keywords = [
            'computer', 'password', 'weather', 'car', 'internet',
            'website', 'email', 'login', 'software', 'hardware', 'technology',
            'restaurant', 'food', 'movie', 'movies', 'music', 'television', 'sports'
        ]
        
        # Removed 'fix' and 'repair' as they might apply to health ("fix my back pain")
        
        # If question contains unrelated keywords as whole words, return no match
        import re
        for unrelated in unrelated_keywords:
            # Use word boundaries to match whole words only
            if re.search(r'\b' + re.escape(unrelated) + r'\b', question_lower):
                return None
        
        matches = []
        
        # Score each FAQ based on multiple criteria
        for faq in faq_data['faqs']:
            score = 0
            faq_id = faq['id']
            
            # 1. Exact keyword matches (highest priority)
            if faq.get('keywords'):
                for keyword in faq['keywords']:
                    if keyword.lower() in question_lower:
                        score += 10
            
            # 2. Question word matches
            faq_keywords = self._extract_keywords_from_text(faq['question'])
            common_keywords = set(question_keywords) & set(faq_keywords)
            score += len(common_keywords) * 3
            
            # 3. Answer content matches
            answer_keywords = self._extract_keywords_from_text(faq['answer'])
            answer_matches = set(question_keywords) & set(answer_keywords)
            score += len(answer_matches) * 2
            
            # 4. Question similarity
            question_similarity = self._calculate_similarity(question_text, faq['question'])
            if question_similarity > 0.3:
                score += question_similarity * 5
            
            # 5. Priority boost
            score += faq.get('priority', 0) * 0.5
            
            if score > 0:
                matches.append({
                    'faq': faq,
                    'score': score,
                    'similarity': question_similarity
                })
        
        # Sort by score and return best match
        if matches:
            matches.sort(key=lambda x: x['score'], reverse=True)
            best_match = matches[0]
            
            # Only return if score is above threshold
            # With unrelated keyword filtering, we can be more lenient with scoring
            if best_match['score'] >= 5:
                return best_match['faq']
        
        return None
    
    async def update_faq_usage(self, faq_id: str):
        """Update usage count for an FAQ (for analytics)."""
        try:
            supabase = self.init_supabase()
            # Get current usage count first
            current = supabase.table("clinics_faqs").select("usage_count").eq("id", faq_id).execute()
            if current.data:
                current_count = current.data[0].get('usage_count', 0) or 0
                # Increment usage count
                supabase.table("clinics_faqs").update({
                    "usage_count": current_count + 1
                }).eq("id", faq_id).execute()
        except Exception as e:
            print(f"Warning: Could not update FAQ usage count: {e}")
    
    async def clear_cache(self, clinic_id: str = None):
        """Clear cached FAQ data for specific clinic or all entries."""
        async with self._lock:
            if clinic_id:
                self._cache.pop(clinic_id, None)
            else:
                self._cache.clear()

# Global FAQ cache instance
faq_cache = FAQCache()

async def get_faq_answer(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find and return the best FAQ answer for a given question.
    
    Args:
        params: Dictionary containing:
            - question: The caller's question text
            - caller_phone: Phone number of the caller  
            - called_phone: Phone number that was called (clinic's phone)
    
    Returns:
        Dict containing answer, confidence, and metadata
    """
    try:
        # Extract parameters
        question = params.get("question")
        caller_phone = params.get("caller_phone")
        called_phone = params.get("called_phone")
        
        if not question:
            return {
                "success": False,
                "error": "Question parameter is required",
                "answer": "I need a question to help you with. Is there anything specific you'd like to know?"
            }
        
        # Import clinic_cache to get clinic_id
        from clinic_cache import clinic_cache
        
        # Get clinic data from existing cache
        clinic_data = clinic_cache.get_cached_data(called_phone)
        if not clinic_data:
            return {
                "success": False,
                "error": "Clinic information not available",
                "answer": "I don't have access to our FAQ information right now. Please call our clinic directly for assistance."
            }
        
        clinic_id = clinic_data['clinic_id']
        
        # Ensure FAQ data is loaded for this clinic
        faq_data = faq_cache.get_cached_faqs(clinic_id)
        if not faq_data:
            faq_data = await faq_cache.load_clinic_faqs(clinic_id)
        
        # Find best matching FAQ
        best_faq = faq_cache.find_best_faq_match(question, clinic_id)
        
        if best_faq:
            # Update usage count asynchronously
            asyncio.create_task(faq_cache.update_faq_usage(best_faq['id']))
            
            # Substitute clinic variables in the answer
            answer = best_faq['answer']
            if clinic_data:
                # Replace common variables that might appear in FAQ answers
                clinic_name = clinic_data.get('clinic_name', 'our clinic')
                clinic_phone = clinic_data.get('phone_number', 'our main number')
                
                answer = answer.replace('{CLINIC_NAME}', clinic_name)
                answer = answer.replace('{CLINIC_PHONE}', clinic_phone)
                # Add more variable substitutions as needed
            
            return {
                "success": True,
                "answer": answer,
                "question": best_faq['question'],
                "category": best_faq.get('category_id'),
                "confidence": "high",
                "faq_id": best_faq['id']
            }
        else:
            return {
                "success": False,
                "error": "No matching FAQ found",
                "answer": "I don't have specific information about that. Would you like me to help you schedule an appointment so you can discuss this with our team?"
            }
    
    except Exception as e:
        print(f"Error in get_faq_answer: {e}")
        return {
            "success": False,
            "error": f"FAQ lookup failed: {str(e)}",
            "answer": "I'm having trouble accessing our FAQ information right now. Is there anything else I can help you with, like scheduling an appointment?"
        }