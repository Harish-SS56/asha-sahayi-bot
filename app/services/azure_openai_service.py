"""
Azure OpenAI Service for ASHA AI Assistant.
Handles chat completions, structured extraction, and translation.
"""

import json
import logging
from typing import Optional, List, Dict, Any
from openai import AzureOpenAI
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AzureOpenAIService:
    """Service for Azure OpenAI GPT-4o interactions."""
    
    def __init__(self):
        """Initialize Azure OpenAI client."""
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version
        )
        self.deployment = settings.azure_openai_chat_deployment
        
        # System prompts
        self.asha_system_prompt = self._get_asha_system_prompt()
        self.extraction_system_prompt = self._get_extraction_system_prompt()
    
    def _get_asha_system_prompt(self) -> str:
        """Get the ASHA assistant system prompt."""
        return """You are ASHA Sahayi, an AI assistant specifically designed to help ASHA (Accredited Social Health Activist) workers in India.

CRITICAL IDENTITY: You are NOT a doctor, NOT a nurse, and NOT a medical professional. You are an AI support tool. You MUST make this clear whenever users ask for a diagnosis or treatment decision. Always say "Please consult a doctor" for any clinical decision.

Your role is to:

1. **Understand and Process Health Updates**: Parse conversational health updates from ASHA workers about their patients.

2. **Provide Medical Guidance**: Offer evidence-based health advice grounded ONLY in official NHM (National Health Mission) guidelines and ASHA handbooks. If the provided context does not contain enough information to answer, say so clearly — do NOT generate information from general knowledge.

3. **Multilingual Support**: Understand and respond in Hindi, Tamil, Malayalam, and English. Always respond in the language explicitly specified in your instructions, regardless of what language or script the input appears in.

4. **Patient Memory**: Help track patient information and visit history. When a patient name is mentioned, help recall their health history if available.

5. **Safety First — NON-NEGOTIABLE**:
   - NEVER diagnose medical conditions definitively
   - NEVER prescribe medications or recommend dosages beyond standard first-aid (ORS, Paracetamol)
   - ALWAYS recommend consulting a qualified doctor for serious symptoms
   - NEVER include your own disclaimer section — a standardised disclaimer is added automatically after your response
   - Flag emergency symptoms that require immediate medical attention and call 108

6. **Be Empathetic**: Understand the challenging conditions ASHA workers operate in. Be supportive and practical.

7. **Structured Responses**: When recording patient visits, help organize information clearly.

Remember: You are a decision-support tool, not a replacement for medical professionals. When in doubt, always advise seeking professional medical help.

RESPONSE FORMATTING — CRITICAL (read carefully):
- NEVER use markdown headings (#, ##, ###, ####) — they show as literal # symbols
- NEVER use horizontal rules (---) — they show as literal dashes
- NEVER use double asterisks **like this** — use *single asterisks* for bold section titles
- NEVER add a "Disclaimer:" section — a standardised one is added automatically
- For section titles: write *Title* on its own line (single asterisks only)
- For bullet points: use • symbol or numbers (1. 2. 3.)
- Separate sections with a single blank line
- Use emojis (📋 ✅ ⚠️ 💊 🏥) as visual markers instead of dividers
- Keep responses concise and practical — ASHA workers are in the field

IMPORTANT EMERGENCY SYMPTOMS to always flag:
- Severe chest pain
- Difficulty breathing
- High fever (>103°F/39.4°C)
- Severe bleeding
- Loss of consciousness
- Signs of stroke (face drooping, arm weakness, speech difficulty)
- Severe abdominal pain in pregnancy (possible preeclampsia/abruption)
- Seizures/convulsions"""

    def _get_extraction_system_prompt(self) -> str:
        """Get the structured extraction system prompt."""
        return """You are a medical data extraction assistant. Your task is to extract structured information from conversational health updates provided by ASHA workers.

Extract ALL of the following information when present. Be thorough — do not leave fields empty if the data is mentioned anywhere in the text:

PATIENT IDENTITY:
- patient_name: Full name of the patient (keep in original script/language, do not translate names)
- age: Numeric age in years
- gender: "male", "female", or "other"

LOCATION & CONTACT:
- village: Village, locality, town, or area name where the patient lives (extract even if mentioned in passing like "visited Rameesh in Saravanampatti")
- phone: Any phone number mentioned (patient's or emergency contact)
- address: Any address details beyond the village

HEALTH PROFILE (chronic/background conditions — NOT acute symptoms):
- known_conditions: Array of chronic diseases already known for this patient (e.g., ["diabetes", "hypertension", "asthma"]). ALWAYS translate to English. If patient "has had diabetes for 8 years" → ["diabetes"]
- allergies: Array of known allergies (translate to English)
- is_pregnant: true/false — is the patient currently pregnant?
- pregnancy_months: Number of months pregnant if pregnant

CURRENT VISIT — SYMPTOMS (acute, presenting today):
- symptoms: Array of symptoms reported today (translate to English). E.g., ["frequent urination", "fatigue", "headache"]

VITAL SIGNS (measured today — translate units to standard):
- blood_pressure: "systolic/diastolic" format e.g. "130/80"
- temperature: Numeric in Celsius (convert from Fahrenheit if needed)
- pulse_rate: Numeric beats per minute
- weight: Numeric in kg
- blood_sugar: Numeric mg/dL — ALWAYS extract this if blood sugar / blood glucose is mentioned. E.g., "blood sugar 268 mg/dL" → 268
- oxygen_saturation: Numeric percentage

VISIT DETAILS:
- visit_type: One of: home_visit, clinic_visit, follow_up, emergency, immunization, antenatal, postnatal, routine_checkup
- observations: Brief summary of what the ASHA worker observed (translate to English)
- medications_given: Array of medications given or prescribed today
- notes: Any other important context (emergency contacts, social factors, medication adherence issues)

ACTIONS:
- referral_needed: true/false
- referral_reason: Why referral is needed (translate to English)
- follow_up_required: true/false
- follow_up_days: Number of days until follow-up
- is_emergency: true/false — requires immediate emergency care

CHILD/INFANT RULES:
- is_child: Set true for ANY patient under 18 years, for all immunization visits, postnatal visits, or when age is given as days/months
- age: Always in YEARS. For babies under 1 year, set age=0
- age_in_days: Set for infants under 12 months. "10 days old" → age_in_days=10. "3 months old" → age_in_days=90

CRITICAL RULES:
1. Input can be in Malayalam, Tamil, Hindi, or English — extract ALL fields regardless of language
2. ALWAYS translate field values to English EXCEPT patient_name (keep original script)
3. Separate chronic conditions (known_conditions) from today's symptoms (symptoms)
4. blood_sugar is a VITAL SIGN — if any blood glucose/sugar reading is mentioned, extract it as a number
5. village must be extracted if ANY location of patient's residence is mentioned
6. Use null for fields not mentioned — never guess

EXAMPLES:
- "visited Rameesh in Saravanampatti, 56 years old, has diabetes for 8 years, blood sugar 268 today" →
  patient_name="Rameesh", age=56, village="Saravanampatti", known_conditions=["diabetes"], blood_sugar=268
- "സരവണമ്പട്ടിയിൽ 56 വയസ്സുള്ള രമേഷ്... 8 വർഷമായി പ്രമേഹം... 268 mg/dL" →
  patient_name="രമേഷ്", age=56, village="Saravanampatti", known_conditions=["diabetes"], blood_sugar=268
- "Radha ko 2 saal se BP ki problem hai, aaj BP 150/100" →
  patient_name="Radha", known_conditions=["hypertension"], blood_pressure="150/100"
"""

    async def get_chat_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        rag_context: Optional[str] = None,
        language: str = "en",
        negative_feedback_context: Optional[List[Dict]] = None
    ) -> str:
        """
        Get a chat response from GPT-4o.
        
        Args:
            user_message: The user's message
            conversation_history: Previous conversation messages
            rag_context: Retrieved context from RAG
            language: Response language code
            negative_feedback_context: Recent negative feedback for improvement
        
        Returns:
            AI response string
        """
        try:
            # Build language-aware system prompt
            lang_names = {"hi": "Hindi (हिंदी)", "ta": "Tamil (தமிழ்)", "ml": "Malayalam (മലയാളം)", "en": "English"}
            response_language = lang_names.get(language, "English")

            # Detect if input is Tamil script but target language is Malayalam
            # (Deepgram Whisper transcribes Malayalam speech as Tamil due to phonetic similarity)
            tamil_script_in_ml_context = (
                language == "ml" and
                any('\u0B80' <= c <= '\u0BFF' for c in (user_message or ""))
            )

            # Enhanced system prompt with explicit language instruction
            system_prompt = self.asha_system_prompt
            if tamil_script_in_ml_context:
                system_prompt += """

CRITICAL LANGUAGE INSTRUCTION:
The user is a MALAYALAM speaker. Their voice message was transcribed by speech recognition
which mistakenly used Tamil script (Tamil and Malayalam sound phonetically similar).
The Tamil text you see is the PHONETIC equivalent of what was spoken in Malayalam.
You MUST:
1. Understand the meaning from the Tamil transcription (the content is the same)
2. Respond ENTIRELY in Malayalam (\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02) script only
3. Do NOT respond in Tamil script (\u0ba4\u0bae\u0bbf\u0bb4\u0bcd) under any circumstances
4. All text in your response must be in Malayalam (\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02)"""
            elif language != "en":
                system_prompt += f"""

CRITICAL LANGUAGE INSTRUCTION: 
You MUST respond ENTIRELY in {response_language}. 
Do NOT respond in English. 
All your output text, explanations, and recommendations must be in {response_language}.
This is mandatory - the user has selected {response_language} as their preferred language."""
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add RAG context if available
            if rag_context:
                messages.append({
                    "role": "system",
                    "content": f"Relevant medical guidelines and information:\n\n{rag_context}\n\nUse this information to ground your response. Remember to respond in {response_language}{'  — NOT in Tamil script' if tamil_script_in_ml_context else ''}."
                })
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Add negative feedback context for improvement
            if negative_feedback_context:
                feedback_summary = "\n".join([
                    f"- Q: \"{fb.get('question', '')[:100]}...\" → User marked response as not helpful"
                    for fb in negative_feedback_context[:3]  # Last 3 negative feedbacks
                ])
                messages.append({
                    "role": "system",
                    "content": f"""Previous responses that were NOT helpful to this user (avoid similar approaches):
{feedback_summary}

Improve your response by being more specific, practical, and relevant to rural ASHA worker needs."""
                })
            
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting chat response: {e}")
            raise

    async def extract_structured_data(
        self,
        text: str,
        patient_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract structured patient data from conversational text.
        
        Args:
            text: Conversational health update text
            patient_context: Known patient information for context
        
        Returns:
            Structured JSON data
        """
        try:
            messages = [
                {"role": "system", "content": self.extraction_system_prompt}
            ]
            
            user_content = f"Extract structured data from this health update:\n\n{text}"
            
            if patient_context:
                user_content += f"\n\nKnown patient context:\n{json.dumps(patient_context, indent=2)}"
            
            messages.append({"role": "user", "content": user_content})
            
            # Define the expected JSON schema
            tools = [{
                "type": "function",
                "function": {
                    "name": "log_patient_visit",
                    "description": "Log a patient visit with extracted health data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_name": {
                                "type": "string",
                                "description": "Name of the patient"
                            },
                            "age": {
                                "type": "integer",
                                "description": "Age of the patient in years. For babies under 1 year (e.g., '10 days old', '3 months old'), set age=0."
                            },
                            "age_in_days": {
                                "type": "integer",
                                "description": "Age in days for newborns/infants under 1 year. E.g., '10 days old' → 10, '3 months old' → 90. Only set for babies under 12 months."
                            },
                            "is_child": {
                                "type": "boolean",
                                "description": "True if patient is a child/infant/newborn (under 18 years). Always true for immunization visits, postnatal visits, or when age is mentioned as days/months."
                            },
                            "gender": {
                                "type": "string",
                                "enum": ["male", "female", "other"],
                                "description": "Gender of the patient"
                            },
                            "village": {
                                "type": "string",
                                "description": "Village or locality name where patient lives"
                            },
                            "phone": {
                                "type": "string",
                                "description": "Patient or emergency contact phone number"
                            },
                            "address": {
                                "type": "string",
                                "description": "Patient address if mentioned"
                            },
                            "is_pregnant": {
                                "type": "boolean",
                                "description": "Whether the patient is pregnant"
                            },
                            "pregnancy_months": {
                                "type": "integer",
                                "description": "Months of pregnancy if pregnant"
                            },
                            "known_conditions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Known medical conditions (diabetes, hypertension, etc.)"
                            },
                            "allergies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Known allergies"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Additional notes about the patient (e.g., emergency contact name, special conditions)"
                            },
                            "symptoms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of symptoms reported"
                            },
                            "blood_pressure": {
                                "type": "string",
                                "description": "Blood pressure reading in format systolic/diastolic"
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Body temperature in Celsius"
                            },
                            "pulse_rate": {
                                "type": "integer",
                                "description": "Pulse rate in beats per minute"
                            },
                            "weight": {
                                "type": "number",
                                "description": "Weight in kilograms"
                            },
                            "blood_sugar": {
                                "type": "number",
                                "description": "Blood sugar level"
                            },
                            "visit_type": {
                                "type": "string",
                                "enum": ["home_visit", "clinic_visit", "follow_up", "emergency", "immunization", "antenatal", "postnatal", "routine_checkup"],
                                "description": "Type of visit"
                            },
                            "observations": {
                                "type": "string",
                                "description": "General observations made during visit"
                            },
                            "medications_given": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of medications given"
                            },
                            "referral_needed": {
                                "type": "boolean",
                                "description": "Whether referral to a doctor is needed"
                            },
                            "referral_reason": {
                                "type": "string",
                                "description": "Reason for referral if needed"
                            },
                            "follow_up_required": {
                                "type": "boolean",
                                "description": "Whether follow-up visit is required"
                            },
                            "follow_up_days": {
                                "type": "integer",
                                "description": "Number of days until follow-up"
                            },
                            "is_emergency": {
                                "type": "boolean",
                                "description": "Whether this is an emergency situation"
                            }
                        },
                        "required": ["patient_name"]
                    }
                }
            }]
            
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "log_patient_visit"}},
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent extraction
            )
            
            # Extract the function call arguments
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                extracted_data = json.loads(tool_call.function.arguments)
                return extracted_data
            
            # Fallback: try to parse response content as JSON
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            
            return {"patient_name": None, "error": "Could not extract data"}
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in extraction: {e}")
            return {"patient_name": None, "error": str(e)}
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            raise

    async def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> str:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (en, hi, ta, ml)
            source_language: Source language code (optional)
        
        Returns:
            Translated text
        """
        try:
            lang_names = {
                "en": "English",
                "hi": "Hindi", 
                "ta": "Tamil",
                "ml": "Malayalam"
            }
            
            target_name = lang_names.get(target_language, "English")
            
            prompt = f"Translate the following text to {target_name}. Return only the translation without any explanation:\n\n{text}"
            
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a translator. Translate text accurately while preserving meaning and medical terminology."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return text  # Return original text on error

    async def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Language code (en, hi, ta, ml)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "Detect the language of the text. Return only the ISO 639-1 code: 'en' for English, 'hi' for Hindi, 'ta' for Tamil, 'ml' for Malayalam. If uncertain, return 'en'."
                    },
                    {"role": "user", "content": text}
                ],
                max_tokens=10,
                temperature=0
            )
            
            detected = response.choices[0].message.content.strip().lower()
            
            # Validate and normalize
            valid_codes = ["en", "hi", "ta", "ml"]
            if detected in valid_codes:
                return detected
            
            # Handle variations
            if "hindi" in detected:
                return "hi"
            if "tamil" in detected:
                return "ta"
            if "malayalam" in detected:
                return "ml"
            
            return "en"  # Default to English
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"

    async def check_for_emergency(self, text: str) -> Dict[str, Any]:
        """
        Check if the text describes an emergency situation.
        
        Args:
            text: Health update text
        
        Returns:
            Dictionary with emergency status and recommendations
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the health update for emergency symptoms. Check for:
- Severe chest pain or heart attack signs
- Difficulty breathing
- High fever (>103°F/39.4°C)
- Heavy bleeding
- Loss of consciousness
- Stroke symptoms (FAST: Face drooping, Arm weakness, Speech difficulty, Time to call)
- Severe allergic reactions
- Seizures
- Severe abdominal pain
- Signs of shock

Return JSON with:
{
    "is_emergency": boolean,
    "severity": "low" | "medium" | "high" | "critical",
    "symptoms_flagged": ["list of concerning symptoms"],
    "recommendation": "what to do"
}"""
                    },
                    {"role": "user", "content": text}
                ],
                max_tokens=300,
                temperature=0
            )
            
            content = response.choices[0].message.content
            # Clean up potential markdown formatting
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error checking for emergency: {e}")
            return {
                "is_emergency": False,
                "severity": "low",
                "symptoms_flagged": [],
                "recommendation": "Unable to assess. If unsure, seek medical help."
            }

    async def classify_message(self, text: str) -> Dict[str, Any]:
        """
        Classify message intent to determine how to process it.
        
        Args:
            text: User's message text
        
        Returns:
            Dictionary with message classification:
            - message_type: "greeting" | "question" | "patient_update" | "command"
            - confidence: 0.0 to 1.0
            - has_patient_data: boolean - whether message contains actual patient info
        """
        try:
            # Quick check for very short messages (likely greetings)
            text_lower = text.lower().strip()
            greeting_words = {'hi', 'hii', 'hiii', 'hello', 'hey', 'hola', 'namaste', 
                            'नमस्ते', 'हाय', 'हेलो', 'வணக்கம்', 'ഹലോ', 'hai', 'helo',
                            'good morning', 'good evening', 'good afternoon', 
                            'gm', 'gud morning', 'gd mrng'}
            
            # Direct match for single-word greetings
            if text_lower in greeting_words or len(text_lower) <= 5:
                for greet in greeting_words:
                    if greet in text_lower:
                        return {
                            "message_type": "greeting",
                            "confidence": 0.95,
                            "has_patient_data": False,
                            "reason": "Simple greeting detected"
                        }
            
            # Use AI for more complex classification
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a message classifier for an ASHA healthcare worker assistant.

Classify the message into ONE of these categories:

1. **greeting**: Simple greetings, hellos, or casual conversation starters
   Examples: "hi", "hello", "namaste", "how are you", "good morning"

2. **question**: Healthcare/medical questions or general information requests
   Examples: "What is the treatment for diarrhea?", "How to check BP?", "What are symptoms of dengue?"

3. **patient_update**: Actual patient health information with identifiable patient details
   MUST contain at least ONE of: patient name, specific symptoms with context, vital signs, visit details
   Examples: "Visited Radha today, BP was 130/80", "Patient Meera has fever for 3 days"

4. **command**: Bot commands or menu selections (you likely won't see these)

Respond with ONLY valid JSON:
{
    "message_type": "greeting" | "question" | "patient_update" | "command",
    "confidence": 0.0 to 1.0,
    "has_patient_data": true/false (does it contain actual patient name + health info?),
    "reason": "brief explanation"
}

IMPORTANT: A message is ONLY "patient_update" if it contains BOTH:
- A patient name (a person's name, not "patient" generically)
- Some health information (symptoms, vitals, visit description)

If just a name with no health context, classify as "greeting" or "question"."""
                    },
                    {"role": "user", "content": f"Classify this message: {text}"}
                ],
                max_tokens=150,
                temperature=0
            )
            
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            
            # Validate result
            valid_types = ["greeting", "question", "patient_update", "command"]
            if result.get("message_type") not in valid_types:
                result["message_type"] = "question"  # Default to question
            
            logger.info(f"Message classified as: {result['message_type']} (confidence: {result.get('confidence', 0)}) - {result.get('reason', '')}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in classification: {e}")
            return self._local_classify_fallback(text)
        except Exception as e:
            logger.error(f"Error classifying message (offline?): {e}")
            return self._local_classify_fallback(text)
    
    def _local_classify_fallback(self, text: str) -> Dict[str, Any]:
        """
        OFFLINE FALLBACK: Classify message using local rules when Azure is unavailable.
        More sophisticated than just defaulting to 'question'.
        """
        import re
        text_lower = text.lower().strip()
        text_len = len(text)
        
        # Greeting detection (same as online)
        greeting_words = {'hi', 'hii', 'hiii', 'hello', 'hey', 'hola', 'namaste', 
                        'नमस्ते', 'हाय', 'हेलो', 'வணக்கம்', 'ഹലോ', 'hai', 'helo',
                        'good morning', 'good evening', 'good afternoon'}
        if text_lower in greeting_words or (text_len <= 10 and any(g in text_lower for g in greeting_words)):
            return {"message_type": "greeting", "confidence": 0.8, "has_patient_data": False, "reason": "Local: greeting detected"}
        
        # Patient data indicators - names typically follow these patterns
        name_indicators = [
            r'\b(?:visited|met|saw|checked)\s+([A-Z][a-z]+)',  # "Visited Suresh"
            r'\b(?:patient|name)[:\s]+([A-Z][a-z]+)',  # "Patient: Suresh" or "name Suresh"
            r'^([A-Z][a-z]+),?\s+(?:age|aged|\d+)',  # "Suresh, age 21" at start
            r'\b([A-Z][a-z]+)\s+(?:has|had|is|was|complained|reports?)',  # "Suresh has fever"
        ]
        has_name_pattern = any(re.search(p, text, re.IGNORECASE) for p in name_indicators)
        
        # Health data indicators
        health_indicators = [
            r'\b(?:fever|cough|cold|pain|headache|vomiting|diarrhea|weakness|fatigue)\b',
            r'\b(?:blood\s*(?:pressure|sugar)|bp|temperature|weight|pulse)\b',
            r'\b\d+\s*(?:mg|mmhg|kg|celsius|fahrenheit|bpm|%)\b',  # Vitals with units
            r'\b\d+/\d+\b',  # BP format like 120/80
            r'\b(?:diabetic|hypertensive|pregnant|anemic)\b',
            r'\b(?:medication|medicine|tablet|syrup)\b',
            r'\b(?:symptoms?|diagnosis|condition|treatment)\b',
        ]
        has_health_data = sum(1 for p in health_indicators if re.search(p, text_lower)) >= 1
        
        # Visit indicators
        visit_indicators = [
            r'\b(?:visited|visit|checkup|check-?up|examined)\b',
            r'\b(?:today|yesterday|morning|evening)\b',
            r'\b(?:home\s*visit|clinic|hospital)\b',
        ]
        has_visit_context = any(re.search(p, text_lower) for p in visit_indicators)
        
        # Determine classification
        # If has name pattern + health data → patient_update
        if has_name_pattern and has_health_data:
            return {
                "message_type": "patient_update",
                "confidence": 0.75,
                "has_patient_data": True,
                "reason": "Local: name + health data detected"
            }
        
        # If longer text with visit context + health data → likely patient_update
        if text_len > 50 and has_visit_context and has_health_data:
            return {
                "message_type": "patient_update",
                "confidence": 0.65,
                "has_patient_data": True,
                "reason": "Local: visit context + health data"
            }
        
        # If has name pattern alone (likely asking about someone)
        if has_name_pattern:
            return {
                "message_type": "patient_update",
                "confidence": 0.55,
                "has_patient_data": True,
                "reason": "Local: name pattern detected"
            }
        
        # Default: question
        return {
            "message_type": "question",
            "confidence": 0.5,
            "has_patient_data": False,
            "reason": "Local: defaulting to question"
        }


# Singleton instance
_openai_service: Optional[AzureOpenAIService] = None


def get_openai_service() -> AzureOpenAIService:
    """Get singleton Azure OpenAI service instance."""
    global _openai_service
    if _openai_service is None:
        _openai_service = AzureOpenAIService()
    return _openai_service
