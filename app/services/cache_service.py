"""
Response Cache Service for offline/low-connectivity mode.
Caches common responses and provides fallback when API fails.
"""

import hashlib
import re
import logging
from typing import Optional, List, Tuple

from app.database.connection import get_db_context
from app.database import crud

logger = logging.getLogger(__name__)


class ResponseCacheService:
    """Service for caching and retrieving common responses."""
    
    # Common healthcare queries that should be cached (multi-language)
    COMMON_QUERIES = {
        "en": [
            "fever symptoms",
            "pregnancy care",
            "immunization schedule",
            "breastfeeding tips",
            "diarrhea treatment",
            "malaria symptoms",
            "anemia signs",
            "high blood pressure",
            "diabetes symptoms",
            "newborn care",
        ],
        "hi": [
            "बुखार के लक्षण",
            "गर्भावस्था देखभाल",
            "टीकाकरण अनुसूची",
            "स्तनपान युक्तियाँ",
            "दस्त का इलाज",
            "मलेरिया के लक्षण",
            "एनीमिया के लक्षण",
            "उच्च रक्तचाप",
            "मधुमेह के लक्षण",
            "नवजात शिशु की देखभाल",
        ],
        "ta": [
            "காய்ச்சல் அறிகுறிகள்",
            "கர்ப்ப பராமரிப்பு",
            "தடுப்பூசி அட்டவணை",
            "தாய்ப்பால் குறிப்புகள்",
            "வயிற்றுப்போக்கு சிகிச்சை",
            "மலேரியா அறிகுறிகள்",
            "இரத்தசோகை அறிகுறிகள்",
            "உயர் இரத்த அழுத்தம்",
            "சர்க்கரை நோய் அறிகுறிகள்",
            "புதிதாகப் பிறந்த குழந்தை பராமரிப்பு",
        ],
        "ml": [
            "പനിയുടെ ലക്ഷണങ്ങൾ",
            "ഗർഭകാല പരിചരണം",
            "വാക്സിനേഷൻ ഷെഡ്യൂൾ",
            "മുലയൂട്ടൽ നുറുങ്ങുകൾ",
            "വയറിളക്കം ചികിത്സ",
            "മലേറിയ ലക്ഷണങ്ങൾ",
            "അനീമിയ ലക്ഷണങ്ങൾ",
            "ഉയർന്ന രക്തസമ്മർദ്ദം",
            "പ്രമേഹ ലക്ഷണങ്ങൾ",
            "നവജാത ശിശു പരിചരണം",
        ]
    }
    
    # Stop words to remove when normalizing queries
    STOP_WORDS = {
        "en": {"what", "is", "are", "how", "to", "the", "a", "an", "for", "of", "in", "on", "can", "do", "does", "tell", "me", "about", "please"},
        "hi": {"क्या", "है", "हैं", "कैसे", "का", "की", "के", "में", "पर", "एक", "मुझे", "बताएं", "कृपया"},
        "ta": {"என்ன", "எப்படி", "ஒரு", "இந்த", "அந்த", "என்னை", "கூறு", "தயவுசெய்து"},
        "ml": {"എന്താണ്", "എങ്ങനെ", "ഒരു", "ഈ", "ആ", "എനിക്ക്", "പറയൂ", "ദയവായി"},
    }
    
    def __init__(self):
        """Initialize cache service."""
        self._local_cache = {}  # In-memory cache for very fast access
        logger.info("Response cache service initialized")
    
    def normalize_query(self, query: str, language: str = "en") -> str:
        """Normalize query for matching by removing stop words and extra spaces."""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove stop words
        stop_words = self.STOP_WORDS.get(language, self.STOP_WORDS["en"])
        words = normalized.split()
        words = [w for w in words if w not in stop_words and len(w) > 1]
        
        # Join and remove extra spaces
        normalized = ' '.join(words)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def extract_keywords(self, query: str, language: str = "en") -> List[str]:
        """Extract keywords from query for matching.
        
        IMPROVED: Filters out common stop words and prioritizes medical terms.
        """
        normalized = self.normalize_query(query, language)
        keywords = normalized.split()
        
        # Stop words to exclude (common words that don't help with matching)
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'shall',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where',
            'what', 'which', 'who', 'whom', 'how', 'why',
            'this', 'that', 'these', 'those', 'there', 'here',
            'from', 'with', 'for', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below',
            'some', 'any', 'all', 'each', 'every', 'both',
            'one', 'two', 'three', 'four', 'five',
            'also', 'very', 'just', 'only', 'more', 'most', 'other',
            'now', 'today', 'yesterday', 'days', 'day', 'last',
            'please', 'thank', 'thanks', 'help', 'need',
            # Health-specific stop words
            'patient', 'visited', 'visit', 'check', 'checked', 'report', 'reports'
        }
        
        # Medical priority keywords (boost these)
        medical_keywords = {
            'fever', 'cough', 'cold', 'pain', 'headache', 'vomiting', 'diarrhea',
            'diabetes', 'diabetic', 'hypertension', 'pregnant', 'pregnancy', 'anemia',
            'blood', 'pressure', 'sugar', 'temperature', 'weight', 'pulse',
            'medication', 'medicine', 'tablet', 'syrup', 'injection',
            'symptoms', 'treatment', 'diagnosis', 'emergency',
            'child', 'children', 'infant', 'baby', 'newborn', 'mother', 'maternal'
        }
        
        # Keep only meaningful words (3+ chars for English, 2+ for others)
        min_len = 3 if language == "en" else 2
        keywords = [k for k in keywords if len(k) >= min_len]
        
        # Filter out stop words
        keywords = [k for k in keywords if k.lower() not in stop_words]
        
        # Sort: medical keywords first, then by length (longer = more specific)
        keywords.sort(key=lambda k: (0 if k.lower() in medical_keywords else 1, -len(k)))
        
        return keywords[:10]  # Limit to 10 keywords
    
    def compute_hash(self, query: str, language: str = "en") -> str:
        """Compute hash for normalized query."""
        normalized = self.normalize_query(query, language)
        content = f"{language}:{normalized}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_cached_response(
        self,
        query: str,
        language: str = "en"
    ) -> Optional[Tuple[str, float]]:
        """
        Get cached response for a query.
        Returns (response, confidence_score) or None.
        """
        query_hash = self.compute_hash(query, language)
        
        # Check in-memory cache first (very fast)
        cache_key = f"{language}:{query_hash}"
        if cache_key in self._local_cache:
            cached = self._local_cache[cache_key]
            logger.info(f"Local cache hit for: {query[:30]}...")
            return (cached["response"], cached["confidence"])
        
        # Check database cache
        try:
            with get_db_context() as db:
                cache = crud.get_cached_response(db, query_hash)
                if cache:
                    # Store in local cache for faster subsequent access
                    self._local_cache[cache_key] = {
                        "response": cache.response_text,
                        "confidence": cache.confidence_score
                    }
                    return (cache.response_text, cache.confidence_score)
                
                # Try fuzzy matching with keywords
                keywords = self.extract_keywords(query, language)
                if keywords:
                    similar = crud.search_similar_cached_responses(db, keywords, language, limit=1)
                    if similar:
                        cache = similar[0]
                        logger.info(f"Fuzzy cache match for: {query[:30]}...")
                        return (cache.response_text, cache.confidence_score * 0.8)  # Lower confidence for fuzzy
        except Exception as e:
            logger.error(f"Error getting cached response: {e}")
        
        return None
    
    def cache_response(
        self,
        query: str,
        response: str,
        language: str = "en",
        conversation_id: Optional[int] = None
    ) -> bool:
        """Cache a response for future use."""
        try:
            query_hash = self.compute_hash(query, language)
            normalized = self.normalize_query(query, language)
            keywords = self.extract_keywords(query, language)
            
            with get_db_context() as db:
                crud.create_or_update_cache(
                    db=db,
                    query_hash=query_hash,
                    query_normalized=normalized,
                    response_text=response,
                    language=language,
                    query_keywords=keywords,
                    source_conversation_id=conversation_id
                )
            
            # Also update local cache
            cache_key = f"{language}:{query_hash}"
            self._local_cache[cache_key] = {
                "response": response,
                "confidence": 1.0
            }
            
            logger.info(f"Cached response for: {query[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False
    
    def update_feedback(self, query: str, language: str, is_positive: bool) -> bool:
        """Update cache feedback metrics."""
        try:
            query_hash = self.compute_hash(query, language)
            
            with get_db_context() as db:
                crud.update_cache_feedback(db, query_hash, is_positive)
            
            # Invalidate local cache if negative feedback
            if not is_positive:
                cache_key = f"{language}:{query_hash}"
                self._local_cache.pop(cache_key, None)
            
            return True
        except Exception as e:
            logger.error(f"Error updating cache feedback: {e}")
            return False
    
    def should_cache_query(self, query: str, language: str = "en") -> bool:
        """Determine if a query should be cached (common healthcare queries)."""
        normalized = self.normalize_query(query, language).lower()
        
        # Check against common queries
        common = self.COMMON_QUERIES.get(language, self.COMMON_QUERIES["en"])
        for common_query in common:
            common_normalized = self.normalize_query(common_query, language).lower()
            # Check if query contains common query keywords
            if common_normalized in normalized or normalized in common_normalized:
                return True
            # Check for significant keyword overlap
            query_words = set(normalized.split())
            common_words = set(common_normalized.split())
            overlap = len(query_words & common_words)
            if overlap >= 2 or (overlap >= 1 and len(common_words) <= 2):
                return True
        
        return False
    
    def get_offline_fallback(self, language: str = "en") -> str:
        """Get a generic fallback message for offline mode."""
        messages = {
            "en": "⚠️ I'm having trouble connecting right now. Please check your internet connection and try again. For emergencies, please contact your nearest health center immediately.",
            "hi": "⚠️ मुझे अभी कनेक्ट करने में समस्या हो रही है। कृपया अपना इंटरनेट कनेक्शन जांचें और पुनः प्रयास करें। आपातकाल के लिए, कृपया तुरंत अपने निकटतम स्वास्थ्य केंद्र से संपर्क करें।",
            "ta": "⚠️ இப்போது இணைப்பதில் சிக்கல் உள்ளது. உங்கள் இணைய இணைப்பைச் சரிபார்த்து மீண்டும் முயற்சிக்கவும். அவசரநிலைகளுக்கு, உடனடியாக உங்கள் அருகிலுள்ள சுகாதார மையத்தைத் தொடர்பு கொள்ளுங்கள்.",
            "ml": "⚠️ ഇപ്പോൾ കണക്റ്റ് ചെയ്യുന്നതിൽ പ്രശ്നമുണ്ട്. നിങ്ങളുടെ ഇന്റർനെറ്റ് കണക്ഷൻ പരിശോധിച്ച് വീണ്ടും ശ്രമിക്കുക. അടിയന്തിര സാഹചര്യങ്ങളിൽ, ഉടൻ തന്നെ നിങ്ങളുടെ അടുത്തുള്ള ആരോഗ്യ കേന്ദ്രവുമായി ബന്ധപ്പെടുക."
        }
        return messages.get(language, messages["en"])
    
    def clear_local_cache(self):
        """Clear the in-memory cache."""
        self._local_cache.clear()
        logger.info("Local response cache cleared")


# Global instance
_cache_service = None

def get_cache_service() -> ResponseCacheService:
    """Get or create the cache service singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = ResponseCacheService()
    return _cache_service
