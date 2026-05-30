"""
Structured Data Extraction Service for ASHA AI Assistant.
Handles extraction and validation of patient data from conversations.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from enum import Enum

from app.services.azure_openai_service import get_openai_service
from app.database.models import VisitType

logger = logging.getLogger(__name__)


class ExtractedVitals(BaseModel):
    """Extracted vital signs from conversation."""
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    pulse_rate: Optional[int] = None
    temperature: Optional[float] = None  # Celsius
    weight: Optional[float] = None  # kg
    height: Optional[float] = None  # cm
    blood_sugar: Optional[float] = None
    oxygen_saturation: Optional[int] = None
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None:
            # Convert Fahrenheit to Celsius if needed
            if v > 50:  # Likely Fahrenheit
                v = (v - 32) * 5/9
            if not 30 <= v <= 45:
                return None
        return v
    
    @validator('blood_pressure_systolic', 'blood_pressure_diastolic')
    def validate_bp(cls, v):
        if v is not None and not 40 <= v <= 250:
            return None
        return v


class ExtractedPatientData(BaseModel):
    """Complete extracted patient data from conversation."""
    
    # Patient identification
    patient_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    
    # Patient contact/location
    village: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    
    # Health profile
    is_pregnant: bool = False
    pregnancy_months: Optional[int] = None
    is_child: bool = False
    known_conditions: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    
    # Visit details
    visit_type: str = "home_visit"
    symptoms: List[str] = Field(default_factory=list)
    observations: Optional[str] = None
    
    # Vitals
    vitals: Optional[ExtractedVitals] = None
    
    # Actions
    medications_given: List[str] = Field(default_factory=list)
    advice_given: Optional[str] = None
    referral_needed: bool = False
    referral_reason: Optional[str] = None
    
    # Follow-up
    follow_up_required: bool = False
    follow_up_days: Optional[int] = None
    
    # Emergency
    is_emergency: bool = False
    emergency_reason: Optional[str] = None
    
    # Raw data
    raw_extraction: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = 0.0


class ExtractionService:
    """Service for extracting structured data from conversations."""
    
    def __init__(self):
        """Initialize extraction service."""
        self.openai_service = get_openai_service()
        
        # Common symptom keywords for validation
        self.symptom_keywords = {
            "fever", "headache", "pain", "cough", "cold", "vomiting",
            "diarrhea", "weakness", "dizziness", "nausea", "fatigue",
            "swelling", "bleeding", "rash", "itching", "burning",
            "difficulty breathing", "chest pain", "abdominal pain"
        }
        
        # Visit type mappings
        self.visit_type_keywords = {
            "home": VisitType.HOME_VISIT,
            "clinic": VisitType.CLINIC_VISIT,
            "emergency": VisitType.EMERGENCY,
            "follow": VisitType.FOLLOW_UP,
            "routine": VisitType.ROUTINE_CHECKUP,
            "checkup": VisitType.ROUTINE_CHECKUP,
            "immunization": VisitType.IMMUNIZATION,
            "vaccine": VisitType.IMMUNIZATION,
            "antenatal": VisitType.ANTENATAL,
            "prenatal": VisitType.ANTENATAL,
            "postnatal": VisitType.POSTNATAL,
            "postpartum": VisitType.POSTNATAL
        }
    
    async def extract_patient_data(
        self,
        text: str,
        patient_context: Optional[Dict] = None
    ) -> ExtractedPatientData:
        """
        Extract structured patient data from conversational text.
        
        Args:
            text: Conversational health update
            patient_context: Known patient information
        
        Returns:
            ExtractedPatientData with structured information
        """
        try:
            # Get AI extraction
            raw_data = await self.openai_service.extract_structured_data(
                text=text,
                patient_context=patient_context
            )
            
            logger.info(f"Raw extraction result: {raw_data}")
        except Exception as api_error:
            # OFFLINE FALLBACK: Use local regex-based extraction
            logger.warning(f"Azure OpenAI unavailable, using local extraction: {api_error}")
            raw_data = self._local_extract_fallback(text)
            logger.info(f"Local extraction result: {raw_data}")
        
        # Parse vitals (runs for BOTH online and offline)
        vitals = self._parse_vitals(raw_data)
        
        # Parse visit type (handle None gracefully)
        visit_type_str = raw_data.get("visit_type") or "home_visit"
        visit_type = self._parse_visit_type(visit_type_str)
        
        # Safely extract values - use 'or' to handle None values from GPT
        patient_name = raw_data.get("patient_name")
        symptoms = raw_data.get("symptoms") or []
        medications = raw_data.get("medications_given") or []
        referral_needed = raw_data.get("referral_needed") or False
        follow_up_required = raw_data.get("follow_up_required") or False
        is_emergency = raw_data.get("is_emergency") or False
        is_pregnant = raw_data.get("is_pregnant") or False
        known_conditions = raw_data.get("known_conditions") or []
        allergies = raw_data.get("allergies") or []
        
        # Infer is_pregnant from visit_type or pregnancy_months
        if visit_type in (VisitType.ANTENATAL,) or raw_data.get("pregnancy_months"):
            is_pregnant = True
        
        # Infer is_child: from GPT flag, age, age_in_days, or visit type
        gpt_is_child = raw_data.get("is_child") or False
        age_value = raw_data.get("age")
        age_in_days = raw_data.get("age_in_days")
        child_visit_types = (VisitType.IMMUNIZATION, VisitType.POSTNATAL)
        is_child = bool(
            gpt_is_child
            or (age_in_days is not None)  # any age given in days = infant
            or (age_value is not None and age_value < 18)
            or visit_type in child_visit_types
        )
        
        # Create structured data object
        extracted = ExtractedPatientData(
            patient_name=patient_name,
            age=raw_data.get("age"),
            gender=raw_data.get("gender"),
            village=raw_data.get("village"),
            phone=raw_data.get("phone"),
            address=raw_data.get("address"),
            is_pregnant=bool(is_pregnant),
            pregnancy_months=raw_data.get("pregnancy_months"),
            is_child=is_child,
            known_conditions=known_conditions if isinstance(known_conditions, list) else [],
            allergies=allergies if isinstance(allergies, list) else [],
            notes=raw_data.get("notes"),
            visit_type=visit_type.value,
            symptoms=symptoms if isinstance(symptoms, list) else [],
            observations=raw_data.get("observations"),
            vitals=vitals,
            medications_given=medications if isinstance(medications, list) else [],
            referral_needed=bool(referral_needed),
            referral_reason=raw_data.get("referral_reason"),
            follow_up_required=bool(follow_up_required),
            follow_up_days=raw_data.get("follow_up_days"),
            is_emergency=bool(is_emergency),
            raw_extraction=raw_data,
            confidence_score=self._calculate_confidence(raw_data)
        )
        
        # Post-process: rescue known_conditions from notes/observations if GPT missed them
        # Use original text + all GPT fields as the search corpus
        fallback_text = " ".join(filter(None, [
            text,
            raw_data.get("observations"),
            raw_data.get("notes"),
            raw_data.get("symptoms") and ", ".join(raw_data["symptoms"]) if isinstance(raw_data.get("symptoms"), list) else None,
        ]))
        if not extracted.known_conditions:
            extracted.known_conditions = self._extract_conditions_from_text(fallback_text)
        
        # Post-process: rescue village from original text + GPT notes/observations
        if not extracted.village:
            extracted.village = self._extract_village_from_text(fallback_text)
        
        # Check for emergency conditions
        if not extracted.is_emergency:
            extracted.is_emergency = self._check_emergency_indicators(extracted)
        
        logger.info(f"Extracted patient data: name={extracted.patient_name}, confidence={extracted.confidence_score:.2f}")
        
        return extracted
    
    def _extract_conditions_from_text(self, text: str) -> List[str]:
        """
        Fallback: scan notes/observations text for known chronic conditions.
        Handles English terms and common transliterations.
        """
        text_lower = text.lower()
        condition_keywords = {
            "diabetes": ["diabetes", "diabetic", "blood sugar", "sugar", "prameha", "madhumeh",
                         "പ്രമേഹ", "மதுமேஹம்", "मधुमेह", "डायबिटीज"],
            "hypertension": ["hypertension", "high bp", "high blood pressure", "bp high",
                             "രക்தसमর്ദ", "ഉയർന്ന രക്തസമ്മർദ്ദം", "உயர் இரத்த அழுத்தம்", "उच्च रक्तचाप"],
            "asthma": ["asthma", "breathing problem", "shortness of breath chronic",
                       "ആസ്ത്മ", "ஆஸ்துமா", "अस्थमा"],
            "tuberculosis": ["tuberculosis", "tb", "kshaya", "ക്ഷയം", "క్షయం", "क्षय"],
            "anemia": ["anemia", "anaemia", "low hemoglobin", "low haemoglobin",
                       "വിളർച്ച", "இரத்த சோகை", "रक्ताल्पता"],
            "heart disease": ["heart disease", "cardiac", "heart problem", "heart attack history",
                              "ഹൃദ്രോഗ", "இதய நோய்", "हृदय रोग"],
            "kidney disease": ["kidney disease", "renal", "kidney problem",
                               "വൃക്കരോഗ", "சிறுநீரக நோய்", "गुर्दे की बीमारी"],
            "thyroid": ["thyroid", "hypothyroid", "hyperthyroid",
                        "തൈറോയ്ഡ്", "தைராய்டு", "थायरॉयड"],
        }
        found = []
        for condition, keywords in condition_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found.append(condition)
        return found
    
    def _extract_village_from_text(self, text: str) -> Optional[str]:
        """
        Scan any free-form text for a village/location name.
        Handles: 'Lives in X', 'from X village', 'X village', 'residing in X', etc.
        Strips generic suffixes (village, town, city, nagar, puram, etc.).
        """
        import re
        # Suffixes to strip from the end of a matched location name
        strip_suffixes = [
            r'\s+village$', r'\s+town$', r'\s+city$', r'\s+district$',
            r'\s+nagar$', r'\s+puram$', r'\s+gram$', r'\s+taluk$',
        ]
        # Words that are never a village name
        skip_words = {
            "home", "clinic", "hospital", "the", "a", "an", "patient",
            "visit", "here", "there", "now", "him", "her", "his",
            "no", "yes", "previous", "diagnosis", "area", "local",
        }
        # Patterns — most specific first
        patterns = [
            # "lives in Pollachi village" / "residing in X" / "lives at X"
            r'(?:lives?\s+in|lives?\s+at|resides?\s+in|residing\s+in|stays?\s+in|staying\s+in)\s+([A-Za-z][A-Za-z\s]{1,40}?)(?:\s+village|\s+town|\s+city)?(?:\s*[,.]|$)',
            # "visited X in <location>" / "from <location>"
            r'\bfrom\s+([A-Z][a-zA-Z\s]{1,30}?)(?:\s+village|\s+town|\s+city)?(?:\s*[,.]|$)',
            # "in X village" / "at X nagar" — capture multi-word names before suffix
            r'\bin\s+([A-Za-z][a-zA-Z\s]{1,30}?)(?:\s+village|\s+town|\s+city|\s+nagar|\s+puram)(?:\s*[,.]|$)',
            r'\bat\s+([A-Za-z][a-zA-Z\s]{1,30}?)(?:\s+village|\s+town|\s+city|\s+nagar|\s+puram)(?:\s*[,.]|$)',
            # "in Saravanampatti," — capitalised single/multi word after 'in'
            r'\bin\s+([A-Z][a-zA-Z\s]{1,30}?)(?:\s*[,.]|$)',
            # Bare "X village" / "X nagar" — capture full name including nagar/puram
            r'([A-Z][a-zA-Z\s]{2,30}?)\s+(?:village|nagar|puram|gram|taluk|town)(?:\s*[,.]|$)',
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                location = m.group(1).strip()
                # Strip trailing suffixes
                for suffix_pat in strip_suffixes:
                    location = re.sub(suffix_pat, '', location, flags=re.IGNORECASE).strip()
                # Filter generic/skip words
                if location.lower() not in skip_words and len(location) >= 3:
                    # Title-case the result
                    return location.strip().title()
        return None

    # Keep old name as alias for backward compatibility
    def _extract_village_from_notes(self, text: str) -> Optional[str]:
        return self._extract_village_from_text(text)
    
    def _local_extract_fallback(self, text: str) -> Dict[str, Any]:
        """
        OFFLINE FALLBACK: Extract basic patient data using regex when Azure OpenAI is unavailable.
        This provides limited but functional extraction without internet.
        """
        import re
        result = {}
        text_lower = text.lower()
        
        # Extract patient name (common patterns)
        name_patterns = [
            r'(?:visited|met|saw|checked)\s+(?:patient\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'(?:patient|name)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'^([A-Z][a-z]+),?\s+(?:age|aged|\d+)',
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["patient_name"] = match.group(1).strip().title()
                break
        
        # Extract age
        age_patterns = [
            r'(?:age|aged)[:\s]*(\d{1,3})(?:\s*(?:years?|yrs?|y\.?o\.?))?',
            r'(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y\.?o\.?)',
            r'(\d{1,2})\s*(?:months?\s*old|mo\s*old)',  # Babies
            r'(\d{1,3})\s*(?:days?\s*old)',  # Newborns
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age_val = int(match.group(1))
                if 'month' in pattern:
                    result["age"] = 0
                    result["age_in_days"] = age_val * 30
                elif 'day' in pattern:
                    result["age"] = 0
                    result["age_in_days"] = age_val
                else:
                    result["age"] = age_val
                break
        
        # Extract gender
        if re.search(r'\b(she|her|female|woman|girl|lady|mother)\b', text_lower):
            result["gender"] = "female"
        elif re.search(r'\b(he|his|male|man|boy)\b', text_lower):
            result["gender"] = "male"
        
        # Extract common symptoms
        symptom_keywords = [
            "fever", "cough", "cold", "headache", "pain", "vomiting", "diarrhea",
            "weakness", "dizziness", "nausea", "fatigue", "swelling", "bleeding",
            "rash", "itching", "breathing difficulty", "chest pain", "abdominal pain"
        ]
        found_symptoms = [s for s in symptom_keywords if s in text_lower]
        if found_symptoms:
            result["symptoms"] = found_symptoms
        
        # Extract village
        village = self._extract_village_from_text(text)
        if village:
            result["village"] = village
        
        # Extract known conditions
        conditions = self._extract_conditions_from_text(text)
        if conditions:
            result["known_conditions"] = conditions
        
        # Detect visit type from keywords
        if any(kw in text_lower for kw in ["vaccine", "vaccination", "immunization"]):
            result["visit_type"] = "immunization"
        elif any(kw in text_lower for kw in ["pregnant", "antenatal", "prenatal"]):
            result["visit_type"] = "antenatal"
            result["is_pregnant"] = True
        elif any(kw in text_lower for kw in ["postnatal", "postpartum", "after delivery"]):
            result["visit_type"] = "postnatal"
        elif any(kw in text_lower for kw in ["emergency", "urgent", "critical"]):
            result["visit_type"] = "emergency"
            result["is_emergency"] = True
        else:
            result["visit_type"] = "home_visit"
        
        # Detect if child
        if result.get("age_in_days") or (result.get("age") is not None and result.get("age", 99) < 18):
            result["is_child"] = True
        if any(kw in text_lower for kw in ["baby", "infant", "newborn", "child"]):
            result["is_child"] = True
        
        # Store original text as notes
        result["notes"] = text[:500]
        
        return result

    def _parse_vitals(self, data: Dict) -> ExtractedVitals:
        """Parse vital signs from extracted data."""
        vitals = ExtractedVitals()
        
        # Parse blood pressure
        bp = data.get("blood_pressure")
        if bp and isinstance(bp, str) and "/" in bp:
            try:
                systolic, diastolic = bp.split("/")
                vitals.blood_pressure_systolic = int(systolic.strip())
                vitals.blood_pressure_diastolic = int(diastolic.strip())
            except ValueError:
                pass
        
        # Parse other vitals
        vitals.pulse_rate = data.get("pulse_rate")
        vitals.temperature = data.get("temperature")
        vitals.weight = data.get("weight")
        vitals.blood_sugar = data.get("blood_sugar")
        
        return vitals
    
    def _parse_visit_type(self, visit_type_str: str) -> VisitType:
        """Parse visit type from string."""
        if not visit_type_str:
            return VisitType.HOME_VISIT
        
        visit_type_str = str(visit_type_str).lower()
        
        for keyword, vtype in self.visit_type_keywords.items():
            if keyword in visit_type_str:
                return vtype
        
        return VisitType.HOME_VISIT
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate confidence score based on extracted data completeness."""
        required_fields = ["patient_name"]
        optional_fields = ["symptoms", "blood_pressure", "visit_type", "observations"]
        
        score = 0.0
        
        # Required fields weight more
        for field in required_fields:
            if data.get(field):
                score += 0.4
        
        # Optional fields
        for field in optional_fields:
            value = data.get(field)
            if value:
                if isinstance(value, list) and len(value) > 0:
                    score += 0.15
                elif isinstance(value, str) and len(value) > 0:
                    score += 0.15
        
        return min(score, 1.0)
    
    def _check_emergency_indicators(self, data: ExtractedPatientData) -> bool:
        """Check for emergency indicators in extracted data."""
        emergency_symptoms = {
            "severe chest pain", "difficulty breathing", "unconscious",
            "seizure", "heavy bleeding", "high fever", "stroke",
            "heart attack", "severe pain"
        }
        
        # Check symptoms
        for symptom in data.symptoms:
            symptom_lower = symptom.lower()
            for emergency in emergency_symptoms:
                if emergency in symptom_lower:
                    return True
        
        # Check vitals
        if data.vitals:
            # High fever
            if data.vitals.temperature and data.vitals.temperature > 39.5:
                return True
            
            # Very high BP
            if data.vitals.blood_pressure_systolic and data.vitals.blood_pressure_systolic > 180:
                return True
            
            # Very low BP
            if data.vitals.blood_pressure_systolic and data.vitals.blood_pressure_systolic < 80:
                return True
            
            # Low oxygen
            if data.vitals.oxygen_saturation and data.vitals.oxygen_saturation < 90:
                return True
        
        return False
    
    def format_for_display(self, data: ExtractedPatientData, language: str = "en") -> str:
        """
        Format extracted data for display to user.
        
        Args:
            data: Extracted patient data
            language: Display language
        
        Returns:
            Formatted string for display
        """
        lines = []
        
        # Translations for labels
        labels = {
            "en": {
                "header": "📋 *Extracted Data:*",
                "patient": "Patient",
                "age": "Age",
                "gender": "Gender",
                "bp": "BP",
                "temp": "Temp",
                "pulse": "Pulse",
                "weight": "Weight",
                "symptoms": "Symptoms",
                "notes": "Notes",
                "visit": "Visit",
                "referral": "Referral needed",
                "followup": "Follow-up",
                "in_days": "in {days} days",
                "emergency": "EMERGENCY - Seek immediate medical help!"
            },
            "hi": {
                "header": "📋 *निकाला गया डेटा:*",
                "patient": "मरीज़",
                "age": "उम्र",
                "gender": "लिंग",
                "bp": "रक्तचाप",
                "temp": "तापमान",
                "pulse": "नाड़ी",
                "weight": "वजन",
                "symptoms": "लक्षण",
                "notes": "टिप्पणी",
                "visit": "विज़िट",
                "referral": "रेफरल आवश्यक",
                "followup": "फॉलो-अप",
                "in_days": "{days} दिनों में",
                "emergency": "आपातकाल - तुरंत चिकित्सा सहायता लें!"
            },
            "ta": {
                "header": "📋 *பிரித்தெடுக்கப்பட்ட தரவு:*",
                "patient": "நோயாளி",
                "age": "வயது",
                "gender": "பாலினம்",
                "bp": "இரத்த அழுத்தம்",
                "temp": "வெப்பநிலை",
                "pulse": "நாடி",
                "weight": "எடை",
                "symptoms": "அறிகுறிகள்",
                "notes": "குறிப்புகள்",
                "visit": "வருகை",
                "referral": "பரிந்துரை தேவை",
                "followup": "பின்தொடர்தல்",
                "in_days": "{days} நாட்களில்",
                "emergency": "அவசரநிலை - உடனடியாக மருத்துவ உதவி பெறுங்கள்!"
            },
            "ml": {
                "header": "📋 *എക്സ്ട്രാക്റ്റ് ചെയ്ത ഡാറ്റ:*",
                "patient": "രോഗി",
                "age": "പ്രായം",
                "gender": "ലിംഗഭേദം",
                "bp": "രക്തസമ്മർദ്ദം",
                "temp": "താപനില",
                "pulse": "നാഡി",
                "weight": "ഭാരം",
                "symptoms": "ലക്ഷണങ്ങൾ",
                "notes": "കുറിപ്പുകൾ",
                "visit": "സന്ദർശനം",
                "referral": "റഫറൽ ആവശ്യം",
                "followup": "ഫോളോ-അപ്പ്",
                "in_days": "{days} ദിവസത്തിനുള്ളിൽ",
                "emergency": "അടിയന്തരാവസ്ഥ - ഉടൻ വൈദ്യസഹായം തേടുക!"
            }
        }
        
        l = labels.get(language, labels["en"])
        
        # Header
        lines.append(l["header"])
        
        # Patient name
        if data.patient_name:
            lines.append(f"👤 {l['patient']}: {data.patient_name}")
            if data.age:
                lines.append(f"   {l['age']}: {data.age}")
            if data.gender:
                lines.append(f"   {l['gender']}: {data.gender}")
            if data.village:
                lines.append(f"   🏘️ Village: {data.village}")
            if data.phone:
                lines.append(f"   📞 Phone: {data.phone}")
            if data.is_pregnant:
                months = f" ({data.pregnancy_months} months)" if data.pregnancy_months else ""
                lines.append(f"   🤰 Pregnant{months}")
            if data.is_child:
                lines.append(f"   👶 Child (under 6 yrs)")
            if data.known_conditions:
                lines.append(f"   🏥 Conditions: {', '.join(data.known_conditions)}")
            if data.allergies:
                lines.append(f"   ⚠️ Allergies: {', '.join(data.allergies)}")
        
        # Vitals
        if data.vitals:
            if data.vitals.blood_pressure_systolic:
                lines.append(f"🩺 {l['bp']}: {data.vitals.blood_pressure_systolic}/{data.vitals.blood_pressure_diastolic} mmHg")
            if data.vitals.temperature:
                lines.append(f"🌡️ {l['temp']}: {data.vitals.temperature:.1f}°C")
            if data.vitals.pulse_rate:
                lines.append(f"💓 {l['pulse']}: {data.vitals.pulse_rate} bpm")
            if data.vitals.weight:
                lines.append(f"⚖️ {l['weight']}: {data.vitals.weight} kg")
        
        # Symptoms
        if data.symptoms:
            symptoms_str = ", ".join(data.symptoms)
            lines.append(f"🤒 {l['symptoms']}: {symptoms_str}")
        
        # Observations / Notes
        if data.observations:
            lines.append(f"📝 {l['notes']}: {data.observations}")
        elif data.notes:
            lines.append(f"📝 {l['notes']}: {data.notes}")
        
        # Visit type
        lines.append(f"📍 {l['visit']}: {data.visit_type.replace('_', ' ').title()}")
        
        # Flags
        if data.referral_needed:
            lines.append(f"⚠️ {l['referral']}: {data.referral_reason or 'Yes'}")
        
        if data.follow_up_required:
            days = data.follow_up_days or "soon"
            lines.append(f"📅 {l['followup']}: {l['in_days'].format(days=days)}")
        
        if data.is_emergency:
            lines.append(f"🚨 *{l['emergency']}*")
        
        return "\n".join(lines)
    
    def _determine_followup_requirements(self, extracted: ExtractedPatientData) -> tuple:
        """
        Automatically determine if follow-up is required based on clinical risk criteria.
        
        Returns:
            tuple: (follow_up_required: bool, follow_up_days: int or None)
        """
        follow_up_required = extracted.follow_up_required
        follow_up_days = extracted.follow_up_days
        
        # High-risk symptom keywords that warrant follow-up
        high_risk_symptoms = {
            "chest pain", "difficulty breathing", "breathlessness", "severe bleeding",
            "unconscious", "seizure", "convulsion", "severe headache", "blurred vision",
            "swelling", "edema", "reduced fetal movement", "severe abdominal pain"
        }
        
        # Check symptoms (case-insensitive)
        symptoms_lower = [s.lower() for s in (extracted.symptoms or [])]
        has_high_risk_symptom = any(
            any(risk in symptom for risk in high_risk_symptoms) 
            for symptom in symptoms_lower
        )
        
        # Check if pregnant (visit type or keywords in symptoms/observations)
        is_pregnant = (
            extracted.visit_type in ("antenatal", "prenatal") or
            any(kw in " ".join(symptoms_lower) for kw in ["pregnant", "pregnancy", "antenatal"])
        )
        
        # Get BP values if available
        bp_systolic = extracted.vitals.blood_pressure_systolic if extracted.vitals else None
        bp_diastolic = extracted.vitals.blood_pressure_diastolic if extracted.vitals else None
        temperature = extracted.vitals.temperature if extracted.vitals else None
        
        # Risk-based follow-up rules
        # Rule 1: Pregnant woman with hypertension (preeclampsia risk) - URGENT
        if is_pregnant and bp_systolic and bp_systolic >= 140:
            follow_up_required = True
            follow_up_days = 1  # Next day follow-up for preeclampsia monitoring
        
        # Rule 2: Pregnant woman with warning symptoms
        elif is_pregnant and has_high_risk_symptom:
            follow_up_required = True
            follow_up_days = min(follow_up_days or 999, 1)
        
        # Rule 3: Severe hypertension (any patient)
        elif bp_systolic and bp_systolic >= 160:
            follow_up_required = True
            follow_up_days = min(follow_up_days or 999, 1)
        
        # Rule 4: Moderate hypertension
        elif bp_systolic and bp_systolic >= 140:
            follow_up_required = True
            follow_up_days = min(follow_up_days or 999, 3)
        
        # Rule 5: High fever (≥ 39°C / 102.2°F)
        elif temperature and temperature >= 39.0:
            follow_up_required = True
            follow_up_days = min(follow_up_days or 999, 2)
        
        # Rule 6: Emergency visit type
        elif extracted.visit_type == "emergency" or extracted.is_emergency:
            follow_up_required = True
            follow_up_days = min(follow_up_days or 999, 2)
        
        # Rule 7: High-risk symptoms without other triggers
        elif has_high_risk_symptom:
            follow_up_required = True
            follow_up_days = min(follow_up_days or 999, 2)
        
        # Default: if already marked for follow-up but no days, set 3 days
        if follow_up_required and not follow_up_days:
            follow_up_days = 3
        
        return follow_up_required, follow_up_days
    
    def create_visit_data(self, extracted: ExtractedPatientData) -> Dict[str, Any]:
        """
        Convert extracted data to visit record format.
        
        Args:
            extracted: Extracted patient data
        
        Returns:
            Dictionary suitable for creating Visit record
        """
        # Automatically determine follow-up requirements based on clinical risk
        follow_up_required, follow_up_days = self._determine_followup_requirements(extracted)
        
        visit_data = {
            "visit_type": VisitType(extracted.visit_type),
            "symptoms": extracted.symptoms,
            "observations": extracted.observations,
            "referral_needed": extracted.referral_needed,
            "referral_facility": extracted.referral_reason,
            "follow_up_required": follow_up_required,
            "extracted_data": extracted.raw_extraction
        }
        
        if extracted.vitals:
            visit_data.update({
                "blood_pressure_systolic": extracted.vitals.blood_pressure_systolic,
                "blood_pressure_diastolic": extracted.vitals.blood_pressure_diastolic,
                "pulse_rate": extracted.vitals.pulse_rate,
                "temperature": extracted.vitals.temperature,
                "weight": extracted.vitals.weight,
                "blood_sugar": extracted.vitals.blood_sugar,
                "oxygen_saturation": extracted.vitals.oxygen_saturation
            })
        
        # Set follow-up date based on determined days
        if follow_up_days:
            visit_data["follow_up_date"] = datetime.utcnow() + timedelta(days=follow_up_days)
        
        if extracted.medications_given:
            visit_data["medications_given"] = extracted.medications_given
        
        return visit_data


# Singleton instance
_extraction_service: Optional[ExtractionService] = None


def get_extraction_service() -> ExtractionService:
    """Get singleton extraction service instance."""
    global _extraction_service
    if _extraction_service is None:
        _extraction_service = ExtractionService()
    return _extraction_service
