"""
Telegram Bot Message Handlers for ASHA AI Assistant.
Handles all user interactions including text, voice, and commands.
"""

import logging
import re
import time
from typing import Optional
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ConversationHandler
)
from telegram.constants import ParseMode, ChatAction

from app.config import get_settings, LANGUAGE_NAMES
from app.database.connection import get_db_context
from app.database import crud
from app.services.azure_openai_service import get_openai_service
from app.services.deepgram_service import get_deepgram_service
from app.services.rag_service import get_rag_service
from app.services.extraction_service import get_extraction_service
from app.services.tts_service import get_tts_service
from app.services.cache_service import get_cache_service

import os
import io

logger = logging.getLogger(__name__)
settings = get_settings()


# ============== Formatting Helpers ==============

def clean_ai_response(text: str) -> str:
    """
    Sanitize AI response for Telegram Markdown display.
    Converts common GPT markdown to Telegram-compatible format and
    removes symbols that TTS would read aloud as "hash" or "dash dash".
    Also strips any disclaimer the AI includes in its response (we add our own).
    """
    # Strip AI-generated disclaimer sections (we append our own standardized one)
    # Matches "### Disclaimer:", "**Disclaimer:**", "Disclaimer:", etc. to end of text
    text = re.sub(
        r'(?i)\n{0,2}#{0,3}\s*\*{0,2}disclaimer\*{0,2}:?.+',
        '',
        text,
        flags=re.DOTALL
    )
    # Strip AI-generated "Emergency Alert" headers — we already send our own hard-coded one
    # Matches: "⚠️ Emergency Alert", "🚨 Emergency Alert", "Emergency Alert", etc. at start
    # Handle emojis explicitly since character classes don't work well with multi-byte emojis
    text = re.sub(r'^[\s\n]*[⚠️🚨]+[\s\n]*', '', text)  # Remove leading emojis
    text = re.sub(
        r'^\**emergency\s*alert\**[:\s\n]*',
        '',
        text,
        flags=re.IGNORECASE
    )
    # Convert ### Heading / ## Heading / # Heading → *Heading* (Telegram bold)
    text = re.sub(r'#{1,6}\s+(.+)', r'*\1*', text)
    # Remove --- / ---- horizontal rules
    text = re.sub(r'(?m)^-{2,}\s*$', '', text)
    # Convert **bold** → *bold*  (Telegram uses single asterisks)
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def strip_for_tts(text: str) -> str:
    """Strip all formatting symbols before sending to TTS."""
    text = re.sub(r'#{1,6}\s*', '', text)        # remove # headings
    text = re.sub(r'-{2,}', '', text)             # remove --- dividers
    text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text)  # remove *bold*
    text = re.sub(r'_(.+?)_', r'\1', text)        # remove _italic_
    text = re.sub(r'`(.+?)`', r'\1', text)        # remove `code`
    text = re.sub(r'\n{2,}', ' ', text)           # collapse newlines to space
    return text.strip()


# ============== Helper Functions ==============

def get_language_keyboard():
    """Create language selection keyboard."""
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("English"), KeyboardButton("हिन्दी (Hindi)")],
            [KeyboardButton("தமிழ் (Tamil)"), KeyboardButton("മലയാളം (Malayalam)")]
        ],
        resize_keyboard=True,
        one_time_keyboard=True
    )


def get_main_menu_keyboard(language: str = "en"):
    """Create main menu keyboard based on language."""
    if language == "hi":
        return ReplyKeyboardMarkup(
            [
                [KeyboardButton("📋 मेरे मरीज़"), KeyboardButton("📊 आँकड़े")],
                [KeyboardButton("📅 फॉलो-अप"), KeyboardButton("❓ मदद")],
                [KeyboardButton("🌐 भाषा बदलें")]
            ],
            resize_keyboard=True
        )
    elif language == "ta":
        return ReplyKeyboardMarkup(
            [
                [KeyboardButton("📋 என் நோயாளிகள்"), KeyboardButton("📊 புள்ளிவிவரங்கள்")],
                [KeyboardButton("📅 பின்தொடர்தல்"), KeyboardButton("❓ உதவி")],
                [KeyboardButton("🌐 மொழி மாற்று")]
            ],
            resize_keyboard=True
        )
    elif language == "ml":
        return ReplyKeyboardMarkup(
            [
                [KeyboardButton("📋 എന്റെ രോഗികൾ"), KeyboardButton("📊 സ്ഥിതിവിവരക്കണക്കുകൾ")],
                [KeyboardButton("📅 ഫോളോ-അപ്പ്"), KeyboardButton("❓ സഹായം")],
                [KeyboardButton("🌐 ഭാഷ മാറ്റുക")]
            ],
            resize_keyboard=True
        )
    else:
        return ReplyKeyboardMarkup(
            [
                [KeyboardButton("📋 My Patients"), KeyboardButton("📊 Stats")],
                [KeyboardButton("📅 Follow-ups"), KeyboardButton("❓ Help")],
                [KeyboardButton("🌐 Change Language")]
            ],
            resize_keyboard=True
        )


def create_feedback_buttons(
    feedback_key: str,
    language: str = "en",
    include_tts: bool = True,
    tts_key: Optional[str] = None
) -> InlineKeyboardMarkup:
    """Create inline keyboard with feedback buttons and optional TTS button."""
    labels = {
        "en": {"helpful": "👍 Helpful", "not_helpful": "👎 Not helpful", "speak": "🔊 Speak"},
        "hi": {"helpful": "👍 सहायक", "not_helpful": "👎 सहायक नहीं", "speak": "🔊 सुनें"},
        "ta": {"helpful": "👍 உதவியாக", "not_helpful": "👎 உதவியாக இல்லை", "speak": "🔊 கேட்க"},
        "ml": {"helpful": "👍 സഹായകരം", "not_helpful": "👎 സഹായകരമല്ല", "speak": "🔊 കേൾക്കാൻ"},
    }
    
    lang_labels = labels.get(language, labels["en"])
    
    buttons = []
    
    # Row 1: Feedback buttons
    buttons.append([
        InlineKeyboardButton(lang_labels["helpful"], callback_data=f"fb_yes_{feedback_key}"),
        InlineKeyboardButton(lang_labels["not_helpful"], callback_data=f"fb_no_{feedback_key}")
    ])
    
    # Row 2: TTS button (optional)
    if include_tts and tts_key:
        buttons.append([InlineKeyboardButton(lang_labels["speak"], callback_data=tts_key)])
    
    return InlineKeyboardMarkup(buttons)


async def send_typing_action(update: Update):
    """Send typing indicator to user. Silently ignores transient network errors."""
    try:
        await update.effective_chat.send_action(ChatAction.TYPING)
    except Exception:
        pass  # Non-critical — typing indicator failure should not crash handlers


async def send_voice_response(update: Update, text: str, language: str = "en"):
    """
    Generate and send a voice message using ElevenLabs TTS.
    
    Args:
        update: Telegram update object
        text: Text to convert to speech
        language: Language code for TTS
    """
    tts_service = get_tts_service()
    
    if not tts_service.is_enabled():
        logger.debug("TTS disabled, skipping voice response")
        return
    
    # Skip if text is too short or just symbols
    clean_text = strip_for_tts(text)
    if len(clean_text) < 20:
        return
    
    # Extract just the AI guidance part (skip confirmation messages)
    # Look for the main guidance text after patient save confirmations
    lines = clean_text.split('\n')
    guidance_lines = []
    for line in lines:
        # Skip save confirmations and disclaimers
        if '✅' in line or '⚠️' in line or 'ID:' in line:
            continue
        if 'DISCLAIMER' in line or 'अस्वीकरण' in line or 'மறுப்பு' in line or 'നിരാകരണം' in line:
            continue
        if line.strip():
            guidance_lines.append(line.strip())
    
    tts_text = ' '.join(guidance_lines[:10])  # Limit to first 10 meaningful lines
    
    if len(tts_text) < 30:
        return
    
    try:
        # Indicate we're generating audio
        await update.effective_chat.send_action(ChatAction.RECORD_VOICE)
        
        # Generate TTS audio
        audio_bytes = await tts_service.text_to_speech(tts_text, language)
        
        if audio_bytes:
            # Send voice message
            await update.message.reply_voice(
                voice=io.BytesIO(audio_bytes),
                caption="🔊 Voice guidance" if language == "en" else "🔊 ध्वनि मार्गदर्शन" if language == "hi" else "🔊 குரல் வழிகாட்டுதல்" if language == "ta" else "🔊 ശബ്ദ മാർഗ്ഗനിർദ്ദേശം"
            )
            logger.info(f"Sent TTS voice response ({len(audio_bytes)} bytes)")
    except Exception as e:
        logger.warning(f"Failed to send voice response: {e}")


# ============== Command Handlers ==============

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    
    # Create or get ASHA worker profile
    with get_db_context() as db:
        worker = crud.get_or_create_asha_worker(
            db=db,
            telegram_user_id=str(user.id),
            telegram_username=user.username,
            name=user.full_name
        )
        language = worker.preferred_language
    
    # Welcome message
    welcome_messages = {
        "en": f"""
🙏 *Namaste, {user.first_name}!*

Welcome to *ASHA AI Assistant* - your healthcare support companion.

I can help you with:
• 📝 Recording patient visits (voice or text)
• 📋 Tracking patient health records
• 🏥 Providing medical guidance from official NHM guidelines
• 🗣️ Multilingual support (Hindi, Tamil, Malayalam, English)

*How to use:*
1. Send a voice note or type about a patient visit
2. I'll extract the important information
3. Ask me any healthcare questions

⚠️ *Important:* I am an AI assistant, NOT a doctor. All responses are for guidance only — always consult a qualified medical professional for diagnosis or treatment.

🔒 *Data & Consent Notice:*
By using this bot, you confirm that:
• You are an authorised ASHA worker
• You have informed your patients that their health data will be recorded digitally for care management
• Data is used only for patient care tracking and is never shared with third parties

Select your preferred language below or just start chatting!
""",
        "hi": f"""
🙏 *नमस्ते, {user.first_name}!*

*आशा AI सहायक* में आपका स्वागत है - आपका स्वास्थ्य सेवा साथी।

मैं आपकी मदद कर सकता/सकती हूं:
• 📝 मरीज़ों की विज़िट रिकॉर्ड करना (वॉइस या टेक्स्ट)
• 📋 मरीज़ों के स्वास्थ्य रिकॉर्ड ट्रैक करना
• 🏥 NHM दिशानिर्देशों से चिकित्सा मार्गदर्शन
• 🗣️ बहुभाषी सहायता

*कैसे उपयोग करें:*
1. मरीज़ की विज़िट के बारे में वॉइस नोट भेजें या टाइप करें
2. मैं महत्वपूर्ण जानकारी निकालूंगा/निकालूंगी
3. कोई भी स्वास्थ्य सवाल पूछें

⚠️ *महत्वपूर्ण:* मैं एक AI सहायक हूं, डॉक्टर नहीं। गंभीर स्थितियों के लिए हमेशा डॉक्टर से सलाह लें।

🔒 *डेटा सहमति:* इस बॉट का उपयोग करके आप पुष्टि करते हैं कि आप एक अधिकृत आशा कार्यकर्ता हैं और आपने मरीज़ों को डिजिटल रिकॉर्ड के बारे में सूचित किया है।
"""
    }
    
    message = welcome_messages.get(language, welcome_messages["en"])
    
    await update.message.reply_text(
        message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_language_keyboard()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    user = update.effective_user
    
    with get_db_context() as db:
        worker = crud.get_or_create_asha_worker(db, str(user.id))
        language = worker.preferred_language
    
    help_messages = {
        "en": """
*ASHA AI Assistant - Help Guide*

*Commands:*
/start - Start the bot
/help - Show this help
/language - Change language
/patients - View your patients
/stats - View your statistics
/followups - View pending follow-ups

*Recording Patient Visits:*
Simply send a voice note or text message describing the visit. Example:
_"Visited Radha today, her BP was 130/80, she complained of headache"_

I will extract:
• Patient name
• Vital signs (BP, temperature, etc.)
• Symptoms
• Visit details

*Asking Questions:*
Ask any healthcare question. I will answer based on official NHM guidelines and ASHA handbook.

*Voice Notes:*
Send voice messages in Hindi, Tamil, Malayalam, or English. I will transcribe and process them.

*Safety:*
⚠️ For emergencies, always call 108 or visit the nearest hospital.
This bot is a support tool, not a replacement for doctors.
""",
        "hi": """
*आशा AI सहायक - मदद गाइड*

*कमांड्स:*
/start - बॉट शुरू करें
/help - यह मदद दिखाएं
/language - भाषा बदलें
/patients - अपने मरीज़ देखें
/stats - अपने आँकड़े देखें

*मरीज़ विज़िट रिकॉर्ड करना:*
बस विज़िट का वर्णन करते हुए एक वॉइस नोट या टेक्स्ट संदेश भेजें।
उदाहरण: _"आज राधा से मिली, उनका BP 130/80 था, सिरदर्द की शिकायत थी"_

*सवाल पूछना:*
कोई भी स्वास्थ्य सवाल पूछें। मैं आधिकारिक NHM दिशानिर्देशों के आधार पर जवाब दूंगा/दूंगी।

⚠️ आपातकाल के लिए, हमेशा 108 पर कॉल करें।
"""
    }
    
    message = help_messages.get(language, help_messages["en"])
    
    await update.message.reply_text(
        message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_menu_keyboard(language)
    )


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /language command."""
    await update.message.reply_text(
        "Please select your preferred language:\nकृपया अपनी भाषा चुनें:",
        reply_markup=get_language_keyboard()
    )


async def patients_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /patients command."""
    user = update.effective_user
    await send_typing_action(update)
    
    with get_db_context() as db:
        worker = crud.get_or_create_asha_worker(db, str(user.id))
        patients = crud.get_patients_for_worker(db, worker.id, limit=10)
        language = worker.preferred_language
        logger.info(f"Fetching patients for worker {worker.id}: found {len(patients)} patients")
    
    if not patients:
        messages = {
            "en": "📋 You don't have any patients recorded yet.\n\nSend a voice note or text mentioning a patient's name and health details to start recording patients.\n\nExample: \"Visited Lakshmi, 45 years old, she has fever and headache\"",
            "hi": "📋 आपके पास अभी तक कोई मरीज़ रिकॉर्ड नहीं है।\n\nमरीज़ का नाम और स्वास्थ्य विवरण बताते हुए वॉइस नोट या टेक्स्ट भेजें।\n\nउदाहरण: \"लक्ष्मी से मिला, 45 साल, उसे बुखार और सिरदर्द है\"",
            "ta": "📋 உங்களிடம் இன்னும் எந்த நோயாளியும் பதிவு செய்யப்படவில்லை.\n\nநோயாளியின் பெயர் மற்றும் சுகாதார விவரங்களைக் குறிப்பிட்டு குரல் அல்லது உரை அனுப்பவும்.\n\nஉதாரணம்: \"லக்ஷ்மியைப் பார்த்தேன், 45 வயது, காய்ச்சல் தலைவலி\"",
            "ml": "📋 നിങ്ങൾക്ക് ഇതുവരെ രോഗികളൊന്നും രേഖപ്പെടുത്തിയിട്ടില്ല.\n\nരോഗിയുടെ പേരും ആരോഗ്യ വിശദാംശങ്ങളും ഉള്ള വോയ്‌സ് നോട്ട് അല്ലെങ്കിൽ ടെക്സ്റ്റ് അയയ്ക്കുക.\n\nഉദാഹരണം: \"ലക്ഷ്മിയെ സന്ദർശിച്ചു, 45 വയസ്സ്, പനിയും തലവേദനയും\""
        }
        await update.message.reply_text(
            messages.get(language, messages["en"]),
            reply_markup=get_main_menu_keyboard(language)
        )
        return
    
    # Format patient list with language-specific headers
    headers = {
        "en": "*📋 Your Recent Patients:*\n\n",
        "hi": "*📋 आपके हाल के मरीज़:*\n\n",
        "ta": "*📋 உங்கள் சமீபத்திய நோயாளிகள்:*\n\n",
        "ml": "*📋 നിങ്ങളുടെ സമീപകാല രോഗികൾ:*\n\n"
    }
    patient_list = headers.get(language, headers["en"])
    
    unknown_name = {"en": "Unknown Patient", "hi": "अज्ञात मरीज़", "ta": "அறியப்படாத நோயாளி", "ml": "അജ്ഞാത രോഗി"}.get(language, "Unknown Patient")
    age_label = {"en": "Age", "hi": "उम्र", "ta": "வயது", "ml": "വയസ്സ്"}.get(language, "Age")
    gender_label = {"en": "Gender", "hi": "लिंग", "ta": "பாலினம்", "ml": "ലിംഗം"}.get(language, "Gender")
    village_label = {"en": "Village", "hi": "गाँव", "ta": "கிராமம்", "ml": "ഗ്രാമം"}.get(language, "Village")
    conditions_label = {"en": "Conditions", "hi": "बीमारियाँ", "ta": "நோய்கள்", "ml": "രോഗങ്ങൾ"}.get(language, "Conditions")
    visit_label = {"en": "Last visit", "hi": "अंतिम विज़िट", "ta": "கடைசி வருகை", "ml": "അവസാന സന്ദർശനം"}.get(language, "Last visit")

    for i, patient in enumerate(patients, 1):
        # Gracefully handle null/empty names
        display_name = patient.name if patient.name and patient.name.strip().lower() not in ("null", "none", "") else unknown_name
        patient_list += f"{i}. *{display_name}*\n"
        if patient.age:
            patient_list += f"   {age_label}: {patient.age}\n"
        if patient.gender:
            patient_list += f"   {gender_label}: {patient.gender}\n"
        if patient.village:
            patient_list += f"   {village_label}: {patient.village}\n"
        if patient.known_conditions:
            conditions_str = ", ".join(patient.known_conditions)
            patient_list += f"   {conditions_label}: {conditions_str}\n"
        # Show last visit info
        if patient.visits:
            last_visit = patient.visits[0]
            patient_list += f"   {visit_label}: {last_visit.visit_date.strftime('%d/%m/%Y')}\n"
        patient_list += "\n"
    
    logger.info(f"Displaying {len(patients)} patients for worker {worker.id}")
    
    await update.message.reply_text(
        patient_list,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_menu_keyboard(language)
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command."""
    user = update.effective_user
    await send_typing_action(update)
    
    with get_db_context() as db:
        worker = crud.get_or_create_asha_worker(db, str(user.id))
        stats = crud.get_worker_stats(db, worker.id)
        language = worker.preferred_language
    
    if language == "hi":
        message = f"""
📊 *आपके आँकड़े*

👥 कुल मरीज़: {stats['total_patients']}
📝 कुल विज़िट: {stats['total_visits']}
📅 इस हफ्ते की विज़िट: {stats['visits_this_week']}
⏰ पेंडिंग फॉलो-अप: {stats['pending_follow_ups']}
"""
    elif language == "ta":
        message = f"""
📊 *உங்கள் புள்ளிவிவரங்கள்*

👥 மொத்த நோயாளிகள்: {stats['total_patients']}
📝 மொத்த வருகைகள்: {stats['total_visits']}
📅 இந்த வாரம் வருகைகள்: {stats['visits_this_week']}
⏰ நிலுவையில் உள்ள பின்தொடர்தல்கள்: {stats['pending_follow_ups']}
"""
    elif language == "ml":
        message = f"""
📊 *നിങ്ങളുടെ സ്ഥിതിവിവരക്കണക്കുകൾ*

👥 മൊത്തം രോഗികൾ: {stats['total_patients']}
📝 മൊത്തം സന്ദർശനങ്ങൾ: {stats['total_visits']}
📅 ഈ ആഴ്ചത്തെ സന്ദർശനങ്ങൾ: {stats['visits_this_week']}
⏰ ബാക്കിയുള്ള ഫോളോ-അപ്പുകൾ: {stats['pending_follow_ups']}
"""
    else:
        message = f"""
📊 *Your Statistics*

👥 Total Patients: {stats['total_patients']}
📝 Total Visits: {stats['total_visits']}
📅 Visits This Week: {stats['visits_this_week']}
⏰ Pending Follow-ups: {stats['pending_follow_ups']}
"""
    
    await update.message.reply_text(
        message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_menu_keyboard(language)
    )


async def followups_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /followups command."""
    user = update.effective_user
    await send_typing_action(update)
    
    with get_db_context() as db:
        worker = crud.get_or_create_asha_worker(db, str(user.id))
        follow_ups = crud.get_follow_ups_due(db, worker.id, days_ahead=7)
        language = worker.preferred_language
        
        # Build follow-up data while session is active to avoid DetachedInstanceError
        followup_data = []
        for fu in follow_ups:
            followup_data.append({
                "patient_name": fu.patient.name if fu.patient else "Unknown",
                "date": fu.follow_up_date.strftime("%d %b") if fu.follow_up_date else "",
                "symptoms": list(fu.symptoms[:3]) if fu.symptoms else []
            })
    
    if not followup_data:
        messages = {
            "en": "✅ No pending follow-ups in the next 7 days!",
            "hi": "✅ अगले 7 दिनों में कोई पेंडिंग फॉलो-अप नहीं!",
            "ta": "✅ அடுத்த 7 நாட்களில் நிலுவையில் உள்ள பின்தொடர்தல்கள் இல்லை!",
            "ml": "✅ അടുത്ത 7 ദിവസത്തിനുള്ളിൽ ഫോളോ-അപ്പുകൾ ഇല്ല!"
        }
        await update.message.reply_text(
            messages.get(language, messages["en"]),
            reply_markup=get_main_menu_keyboard(language)
        )
        return
    
    headers = {
        "en": "*📅 Pending Follow-ups:*\n\n",
        "hi": "*📅 पेंडिंग फॉलो-अप:*\n\n",
        "ta": "*📅 நிலுவையிலுள்ள பின்தொடர்தல்கள்:*\n\n",
        "ml": "*📅 ബാക്കിയുള்ള ഫോളോ-അപ്പുകൾ:*\n\n"
    }
    symptoms_label = {
        "en": "Symptoms",
        "hi": "लक्षण",
        "ta": "அறிகுறிகள்",
        "ml": "ലക്ഷണങ്ങൾ"
    }
    
    message = headers.get(language, headers["en"])
    
    for fu in followup_data:
        message += f"📅 *{fu['patient_name']}* - {fu['date']}\n"
        if fu['symptoms']:
            label = symptoms_label.get(language, symptoms_label["en"])
            message += f"   {label}: {', '.join(fu['symptoms'])}\n"
        message += "\n"
    
    await update.message.reply_text(
        message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_menu_keyboard(language)
    )


# ============== Message Handlers ==============

async def handle_language_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle language selection from keyboard."""
    text = update.message.text
    user = update.effective_user
    
    # Map selection to language code
    language_map = {
        "English": "en",
        "हिन्दी (Hindi)": "hi",
        "தமிழ் (Tamil)": "ta",
        "മലയാളം (Malayalam)": "ml",
        "🌐 Change Language": "change",
        "🌐 भाषा बदलें": "change",
        "🌐 மொழி மாற்று": "change",
        "🌐 ഭാഷ മാറ്റുക": "change"
    }
    
    selected = language_map.get(text)
    
    if selected == "change":
        await language_command(update, context)
        return
    
    if selected:
        with get_db_context() as db:
            worker = crud.get_or_create_asha_worker(db, str(user.id))
            crud.update_worker_language(db, worker.id, selected)
        
        confirmations = {
            "en": "✅ Language set to English",
            "hi": "✅ भाषा हिंदी में सेट की गई",
            "ta": "✅ மொழி தமிழில் அமைக்கப்பட்டது",
            "ml": "✅ ഭാഷ മലയാളത്തിൽ സജ്ജമാക്കി"
        }
        
        await update.message.reply_text(
            confirmations.get(selected, confirmations["en"]),
            reply_markup=get_main_menu_keyboard(selected)
        )


async def handle_menu_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle main menu button selections."""
    text = update.message.text
    
    # Map menu items to commands
    menu_map = {
        "📋 My Patients": patients_command,
        "📋 मेरे मरीज़": patients_command,
        "📋 என் நோயாளிகள்": patients_command,
        "📋 എന്റെ രോഗികൾ": patients_command,
        "📊 Stats": stats_command,
        "📊 आँकड़े": stats_command,
        "📊 புள்ளிவிவரங்கள்": stats_command,
        "📊 സ്ഥിതിവിവരക്കണക്കുകൾ": stats_command,
        "📅 Follow-ups": followups_command,
        "📅 फॉलो-अप": followups_command,
        "📅 பின்தொடர்தல்": followups_command,
        "📅 ഫോളോ-അപ്പ്": followups_command,
        "❓ Help": help_command,
        "❓ मदद": help_command,
        "❓ உதவி": help_command,
        "❓ സഹായം": help_command
    }
    
    handler = menu_map.get(text)
    if handler:
        await handler(update, context)
        return True
    return False


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages from users."""
    user = update.effective_user
    voice = update.message.voice
    
    logger.info(f"Received voice message from {user.id}, duration: {voice.duration}s")
    
    # Get worker info - extract scalar values while in session to avoid DetachedInstanceError
    with get_db_context() as db:
        worker = crud.get_or_create_asha_worker(db, str(user.id))
        worker_id = worker.id
        language = worker.preferred_language
    
    # Send processing indicator
    processing_messages = {
        "en": "🎤 Processing your voice message...",
        "hi": "🎤 आपका वॉइस संदेश प्रोसेस हो रहा है...",
        "ta": "🎤 உங்கள் குரல் செய்தியை செயலாக்குகிறது...",
        "ml": "🎤 നിങ്ങളുടെ വോയ്സ് സന്ദേശം പ്രോസസ്സ് ചെയ്യുന്നു..."
    }
    
    status_message = await update.message.reply_text(
        processing_messages.get(language, processing_messages["en"])
    )
    
    try:
        start_time = time.time()
        
        # Download voice file
        voice_file = await context.bot.get_file(voice.file_id)
        voice_bytes = await voice_file.download_as_bytearray()
        
        # Transcribe using Deepgram
        deepgram_service = get_deepgram_service()
        transcript, confidence, detected_lang = await deepgram_service.transcribe_telegram_voice(
            voice_file_bytes=bytes(voice_bytes),
            language=language
        )
        
        if not transcript or len(transcript.strip()) < 3:
            error_messages = {
                "en": "❌ Sorry, I couldn't understand the audio clearly. Please try again or send a text message.",
                "hi": "❌ क्षमा करें, मैं ऑडियो स्पष्ट रूप से नहीं समझ पाया/पाई। कृपया पुनः प्रयास करें या टेक्स्ट संदेश भेजें।"
            }
            await status_message.edit_text(error_messages.get(language, error_messages["en"]))
            return
        
        # Fix for Malayalam voice being transcribed as Tamil by Deepgram Whisper:
        # (whisper-large confuses Malayalam/Tamil phonetics even with language=ml)
        transcript_has_tamil = any('\u0B80' <= c <= '\u0BFF' for c in transcript)
        if language == "ml" and transcript_has_tamil:
            display_transcript = f"📝 Transcribed (voice in Malayalam):\n_{transcript[:300]}_"
            logger.info("Malayalam voice detected as Tamil script by Deepgram - AI will respond in Malayalam")
        elif len(transcript) > 300:
            display_transcript = f"📝 Transcribed: _{transcript[:300]}..._"
        else:
            display_transcript = f"📝 Transcribed: _{transcript}_"

        # Update status
        await status_message.edit_text(display_transcript, parse_mode=ParseMode.MARKDOWN)
        
        # Process the transcribed text using worker's preferred language for response
        # (do NOT use detected_lang - Deepgram may misidentify ml as ta)
        await process_health_update(
            update=update,
            context=context,
            text=transcript,
            worker_id=worker_id,
            language=language,
            is_voice=True,
            voice_file_id=voice.file_id,
            confidence=confidence,
            status_message=status_message
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Voice message processed in {processing_time}ms")
        
    except Exception as e:
        import traceback
        logger.error(f"Error processing voice message: {e}")
        logger.error(traceback.format_exc())
        error_messages = {
            "en": "❌ Error processing voice message. Please try again.",
            "hi": "❌ वॉइस संदेश प्रोसेस करने में त्रुटि। कृपया पुनः प्रयास करें।",
            "ta": "❌ குரல் செய்தியை செயலாக்குவதில் பிழை. மீண்டும் முயற்சிக்கவும்.",
            "ml": "❌ വോയ്സ് സന്ദേശം പ്രോസസ്സ് ചെയ്യുന്നതിൽ പിശക്. വീണ്ടും ശ്രമിക്കുക."
        }
        await status_message.edit_text(error_messages.get(language, error_messages["en"]))


async def check_and_handle_emergency(update: Update, text: str, language: str) -> bool:
    """
    Check if the message describes an emergency and send emergency alert if so.
    Returns True if emergency was detected.
    """
    # Keywords that indicate emergency situations (case-insensitive check)
    emergency_keywords = [
        # English
        'heart attack', 'cardiac arrest', 'stroke', 'unconscious', 'not breathing',
        'severe bleeding', 'heavy bleeding', 'choking', 'seizure', 'convulsion',
        'poisoning', 'snake bite', 'drowning', 'accident', 'dying', 'critical',
        'very serious', 'life threatening', 'cannot breathe', 'chest pain',
        'difficulty breathing', 'severe pain', 'high fever', 'no pulse',
        # Hindi
        'दिल का दौरा', 'हार्ट अटैक', 'बेहोश', 'सांस नहीं', 'खून बह रहा',
        'गंभीर', 'मर रहा', 'दौरा पड़ा', 'जहर', 'सांप ने काटा', 'सीने में दर्द',
        # Tamil
        'மாரடைப்பு', 'சுயநினைவில்லை', 'மூச்சு விடவில்லை', 'அவசரம்', 'இதய தாக்குதல்',
        # Malayalam
        'ഹൃദയാഘാതം', 'ബോധമില്ല', 'ശ്വാസം ഇല്ല', 'അടിയന്തര'
    ]
    
    text_lower = text.lower()
    
    # Check for emergency keywords
    is_emergency = any(keyword in text_lower for keyword in emergency_keywords)
    
    if is_emergency:
        emergency_messages = {
            "en": """🚨 *EMERGENCY DETECTED!*

This appears to be an emergency situation.

*Immediate Actions:*
1. 📞 *Call 108 (Emergency)* immediately
2. 🧘 Keep the patient calm and still
3. 🚫 Do not move the patient if there's suspected injury
4. 💓 Monitor breathing and pulse
5. ⏰ Note the time when symptoms started

⚠️ *Do not delay - seek professional medical help NOW!*

---
_Detailed guidance below:_""",
            "hi": """🚨 *आपातकाल का पता चला!*

यह एक आपातकालीन स्थिति है।

*तुरंत कार्रवाई:*
1. 📞 *108 (आपातकालीन) पर तुरंत कॉल करें*
2. 🧘 मरीज़ को शांत रखें
3. 🚫 चोट की आशंका हो तो मरीज़ को हिलाएं नहीं
4. 💓 सांस और नब्ज़ की निगरानी करें
5. ⏰ लक्षण शुरू होने का समय नोट करें

⚠️ *देरी न करें - अभी चिकित्सा सहायता लें!*""",
            "ta": """🚨 *அவசரநிலை கண்டறியப்பட்டது!*

இது அவசரநிலை.

*உடனடி நடவடிக்கைகள்:*
1. 📞 *108 உடனடியாக அழைக்கவும்*
2. 🧘 நோயாளியை அமைதியாக வைக்கவும்
3. 🚫 காயம் இருந்தால் நகர்த்த வேண்டாம்
4. 💓 சுவாசம் கண்காணிக்கவும்
5. ⏰ நேரத்தை குறிக்கவும்

⚠️ *தாமதிக்காதீர்கள்!*""",
            "ml": """🚨 *അടിയന്തരാവസ്ഥ!*

ഇത് അടിയന്തര സാഹചര്യം.

*ഉടനടി നടപടികൾ:*
1. 📞 *108 ഉടൻ വിളിക്കുക*
2. 🧘 രോഗിയെ ശാന്തമാക്കുക
3. 🚫 പരിക്കുണ്ടെങ്കിൽ നീക്കരുത്
4. 💓 ശ്വാസം നിരീക്ഷിക്കുക
5. ⏰ സമയം രേഖപ്പെടുത്തുക

⚠️ *വൈകരുത്!*"""
        }
        
        try:
            await update.message.reply_text(
                emergency_messages.get(language, emergency_messages["en"]),
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception:
            plain_msg = emergency_messages.get(language, emergency_messages["en"]).replace('*', '').replace('_', '')
            await update.message.reply_text(plain_msg)
        
        logger.warning(f"EMERGENCY detected in message: {text[:100]}")
        return True
    
    return False


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages from users."""
    user = update.effective_user
    text = update.message.text
    
    # Check if it's a menu selection
    if await handle_menu_selection(update, context):
        return
    
    # Check if it's a language selection
    if text in ["English", "हिन्दी (Hindi)", "தமிழ் (Tamil)", "മലയാളം (Malayalam)"] or text.startswith("🌐"):
        await handle_language_selection(update, context)
        return
    
    logger.info(f"Received text message from user_id={user.id}, length={len(text)} chars")
    
    # Get worker info - extract scalar values while in session to avoid DetachedInstanceError
    with get_db_context() as db:
        worker = crud.get_or_create_asha_worker(db, str(user.id))
        worker_id = worker.id
        language = worker.preferred_language
    
    await send_typing_action(update)
    
    # CRITICAL: Check for emergency FIRST before any classification
    is_emergency = await check_and_handle_emergency(update, text, language)
    
    # Classify the message
    openai_service = get_openai_service()
    classification = await openai_service.classify_message(text)
    message_type = classification.get("message_type", "question")
    has_patient_data = classification.get("has_patient_data", False)
    
    logger.info(f"Message classified as: {message_type}, has_patient_data: {has_patient_data}, is_emergency: {is_emergency}")
    
    # Handle based on message type (skip greeting if emergency was detected)
    if message_type == "greeting" and not is_emergency:
        await handle_greeting(update, context, language)
        return
    
    if message_type == "question" and not has_patient_data:
        await handle_healthcare_question(update, context, text, worker_id, language, is_emergency)
        return
    
    # For patient_update or messages with patient data, process as health update
    await process_health_update(
        update=update,
        context=context,
        text=text,
        worker_id=worker_id,
        language=language,
        is_voice=False
    )


async def handle_greeting(update: Update, context: ContextTypes.DEFAULT_TYPE, language: str) -> None:
    """Handle simple greeting messages."""
    greetings = {
        "en": "Hello! कैसे मदद कर सकता हूँ? How can I assist you today? 😊\n\nYou can:\n• Send patient visit details (voice or text)\n• Ask healthcare questions\n• Use the menu buttons below",
        "hi": "नमस्ते! 🙏 मैं आपकी कैसे मदद कर सकता/सकती हूँ?\n\nआप:\n• मरीज़ विज़िट की जानकारी भेजें (वॉइस या टेक्स्ट)\n• स्वास्थ्य संबंधी सवाल पूछें\n• नीचे दिए मेन्यू बटन का उपयोग करें",
        "ta": "வணக்கம்! 🙏 நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?\n\nநீங்கள்:\n• நோயாளி வருகை விவரங்களை அனுப்பவும்\n• சுகாதார கேள்விகள் கேளுங்கள்\n• மெனு பொத்தான்களைப் பயன்படுத்தவும்",
        "ml": "നമസ്കാരം! 🙏 എനിക്ക് എങ്ങനെ സഹായിക്കാനാകും?\n\nനിങ്ങൾക്ക്:\n• രോഗി സന്ദർശന വിശദാംശങ്ങൾ അയയ്ക്കാം\n• ആരോഗ്യ ചോദ്യങ്ങൾ ചോദിക്കാം\n• മെനു ബട്ടണുകൾ ഉപയോഗിക്കാം"
    }
    
    await update.message.reply_text(
        greetings.get(language, greetings["en"]),
        reply_markup=get_main_menu_keyboard(language)
    )


async def handle_healthcare_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question: str,
    worker_id: int,
    language: str,
    is_emergency: bool = False
) -> None:
    """Handle general healthcare questions without patient data extraction."""
    cache_service = get_cache_service()
    ai_response = None
    from_cache = False
    conversation_log_id = None
    
    try:
        openai_service = get_openai_service()
        rag_service = get_rag_service()
        
        # Get recent negative feedback for improving responses
        negative_feedback = []
        try:
            with get_db_context() as db:
                recent_neg = crud.get_recent_negative_feedback(db, worker_id, limit=3)
                negative_feedback = [
                    {"question": fb.original_question, "response": fb.original_response}
                    for fb in recent_neg if fb.original_question
                ]
        except Exception as fb_err:
            logger.warning(f"Could not fetch negative feedback: {fb_err}")
        
        # Track RAG sources for logging
        rag_sources_used = []
        
        # Check cache first for common queries (offline mode support)
        cached = cache_service.get_cached_response(question, language)
        if cached and cached[1] >= 0.7:  # Use cache if confidence >= 70%
            ai_response = cached[0]
            from_cache = True
            logger.info(f"Using cached response (confidence: {cached[1]:.2f})")
        else:
            # Get RAG context for the question (with sources for logging)
            rag_context, rag_sources_used = await rag_service.get_context_with_sources(question)
            
            # Get AI response using RAG context (with negative feedback for improvement)
            try:
                ai_response = await openai_service.get_chat_response(
                    user_message=question,
                    conversation_history=None,
                    rag_context=rag_context,
                    language=language,
                    negative_feedback_context=negative_feedback if negative_feedback else None
                )
            except Exception as api_error:
                logger.error(f"API error, trying cache fallback: {api_error}")
                # Fall back to cache on API failure (even lower confidence)
                if cached:
                    ai_response = cached[0]
                    from_cache = True
                    logger.info("Using cached response as fallback")
                else:
                    # Try fuzzy cache match
                    keywords = cache_service.extract_keywords(question, language)
                    if keywords:
                        with get_db_context() as db:
                            similar = crud.search_similar_cached_responses(db, keywords, language, limit=1)
                            if similar:
                                ai_response = similar[0].response_text
                                from_cache = True
                                logger.info("Using fuzzy cache match as fallback")
                
                # If still no response, show offline message
                if not ai_response:
                    offline_msg = cache_service.get_offline_fallback(language)
                    await update.message.reply_text(
                        offline_msg,
                        reply_markup=get_main_menu_keyboard(language)
                    )
                    return
        
        # Add disclaimer
        disclaimers = {
            "en": "\n\n⚠️ _DISCLAIMER: This information is for guidance only. Always consult a qualified healthcare professional for medical decisions._",
            "hi": "\n\n⚠️ _अस्वीकरण: यह जानकारी केवल मार्गदर्शन के लिए है। चिकित्सा निर्णयों के लिए हमेशा डॉक्टर से परामर्श करें।_",
            "ta": "\n\n⚠️ _மறுப்பு: இந்த தகவல் வழிகாட்டலுக்கு மட்டுமே. மருத்துவ முடிவுகளுக்கு எப்போதும் மருத்துவரை அணுகவும்._",
            "ml": "\n\n⚠️ _നിരാകരണം: ഈ വിവരങ്ങൾ മാർഗ്ഗനിർദ്ദേശത്തിനു മാത്രമാണ്. വൈദ്യ തീരുമാനങ്ങൾക്ക് എല്ലായ്‌പ്പോഴും ഡോക്ടറെ സമീപിക്കുക._"
        }
        
        # Add cache indicator if response came from cache
        cache_note = ""
        if from_cache:
            cache_notes = {
                "en": "\n📦 _Response from saved answers (offline mode)_",
                "hi": "\n📦 _सहेजे गए उत्तरों से (ऑफ़लाइन मोड)_",
                "ta": "\n📦 _சேமிக்கப்பட்ட பதில்களிலிருந்து (ஆஃப்லைன் பயன்முறை)_",
                "ml": "\n📦 _സേവ് ചെയ്ത ഉത്തരങ്ങളിൽ നിന്ന് (ഓഫ്‌ലൈൻ മോഡ്)_"
            }
            cache_note = cache_notes.get(language, cache_notes["en"])
        
        response = clean_ai_response(ai_response) + disclaimers.get(language, disclaimers["en"]) + cache_note
        
        # Send response with markdown fallback
        try:
            await update.message.reply_text(
                response,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=get_main_menu_keyboard(language)
            )
        except Exception:
            plain_text = re.sub(r'[*_`]', '', response)
            await update.message.reply_text(
                plain_text,
                reply_markup=get_main_menu_keyboard(language)
            )
        
        # Log conversation and get ID for feedback
        try:
            with get_db_context() as db:
                conv_log = crud.log_conversation(
                    db=db,
                    asha_worker_id=worker_id,
                    message_type="text",
                    input_text=question,
                    input_language=language,
                    ai_response=ai_response,
                    response_language=language,
                    rag_chunks_used=rag_sources_used if rag_sources_used else []
                )
                conversation_log_id = conv_log.id
        except Exception as log_error:
            logger.warning(f"Failed to log conversation: {log_error}")
        
        # Cache successful fresh responses for common queries
        # Cache ALL successful fresh responses (aggressive caching for offline support)
        # IMPORTANT: Clean the response BEFORE caching to avoid storing duplicate headers
        if not from_cache and ai_response:
            try:
                cleaned_for_cache = clean_ai_response(ai_response)
                cache_service.cache_response(question, cleaned_for_cache, language, conversation_log_id)
            except Exception as cache_err:
                logger.warning(f"Failed to cache response: {cache_err}")
        
        # Add feedback + TTS buttons
        if ai_response:
            feedback_key = f"{worker_id}_{update.message.message_id}"
            tts_key = f"tts_{update.effective_user.id}_{update.message.message_id}" if settings.tts_enabled else None
            
            # Store context for feedback and TTS
            context.bot_data[f"fb_{feedback_key}"] = {
                "conv_id": conversation_log_id,
                "worker_id": worker_id,
                "question": question,
                "response": ai_response[:1000],  # Truncate for storage
                "language": language,
                "tts_key": tts_key  # Keep speak button visible after feedback
            }
            if tts_key:
                context.bot_data[tts_key] = {"text": strip_for_tts(ai_response)[:600], "language": language}
            
            # Send feedback buttons (with TTS if enabled)
            feedback_markup = create_feedback_buttons(
                feedback_key=feedback_key,
                language=language,
                include_tts=settings.tts_enabled,
                tts_key=tts_key
            )
            
            feedback_prompts = {
                "en": "Was this helpful?",
                "hi": "क्या यह सहायक था?",
                "ta": "இது உதவியாக இருந்ததா?",
                "ml": "ഇത് സഹായകരമായിരുന്നോ?"
            }
            await update.message.reply_text(
                feedback_prompts.get(language, feedback_prompts["en"]),
                reply_markup=feedback_markup
            )
            
    except Exception as e:
        logger.error(f"Error handling healthcare question: {e}")
        error_messages = {
            "en": "❌ Sorry, I couldn't process your question. Please try again.",
            "hi": "❌ क्षमा करें, आपका सवाल प्रोसेस नहीं हो सका। कृपया पुनः प्रयास करें।",
            "ta": "❌ மன்னிக்கவும், உங்கள் கேள்வியை செயலாக்க முடியவில்லை. மீண்டும் முயற்சிக்கவும்.",
            "ml": "❌ ക്ഷമിക്കണം, നിങ്ങളുടെ ചോദ്യം പ്രോസസ്സ് ചെയ്യാനായില്ല. വീണ്ടും ശ്രമിക്കുക."
        }
        await update.message.reply_text(
            error_messages.get(language, error_messages["en"]),
            reply_markup=get_main_menu_keyboard(language)
        )


async def process_health_update(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    worker_id: int,
    language: str,
    is_voice: bool = False,
    voice_file_id: Optional[str] = None,
    confidence: float = 1.0,
    status_message=None
) -> None:
    """
    Process a health update (from voice or text).
    Extracts data, stores it, and provides response.
    """
    try:
        openai_service = get_openai_service()
        rag_service = get_rag_service()
        extraction_service = get_extraction_service()
        
        # Detect language of input text for logging purposes only
        # DO NOT override worker's preferred language - Deepgram may output Tamil
        # script for Malayalam speech, but the response must be in worker's language
        detected_language = await openai_service.detect_language(text)
        # Use detected_language for logging, but keep `language` as worker's preference
        input_language = detected_language  # For conversation log
        
        # Extract structured data
        extracted_data = await extraction_service.extract_patient_data(
            text=text,
            patient_context=None  # Could add context from previous visits
        )
        
        # Check for emergency
        if extracted_data.is_emergency:
            emergency_message = """
🚨 *EMERGENCY DETECTED!*

This appears to be an emergency situation.

*Immediate Actions:*
1. Call 108 (Emergency) immediately
2. Keep the patient calm
3. Do not move the patient if there's suspected injury
4. Monitor breathing and pulse
5. Note the time when symptoms started

⚠️ Do not delay - seek professional medical help NOW!
"""
            await update.message.reply_text(emergency_message, parse_mode=ParseMode.MARKDOWN)
        
        # Get RAG context for medical guidance (may fail offline - that's OK)
        rag_context = ""
        if extracted_data.symptoms or "?" in text:
            try:
                # Get relevant medical information
                search_query = " ".join(extracted_data.symptoms) if extracted_data.symptoms else text
                rag_context = await rag_service.get_context_for_query(search_query)
            except Exception as rag_err:
                logger.warning(f"RAG context unavailable (offline?): {rag_err}")
                rag_context = ""
        
        # Store patient and visit data if patient name was extracted
        # Use plain scalar variables (not ORM objects) outside the DB session
        patient_saved = False
        saved_patient_name = None
        saved_patient_id = None
        saved_visit_id = None
        patient_known_conditions = None
        conversation_log_id = None
        
        if extracted_data.patient_name and extracted_data.patient_name.strip() and worker_id:
            logger.info(f"Attempting to save patient for worker_id={worker_id}")
            try:
                with get_db_context() as db:
                    # Find or create patient
                    patient = crud.find_patient_by_name(
                        db=db,
                        asha_worker_id=worker_id,
                        name=extracted_data.patient_name.strip()
                    )
                    
                    if patient:
                        logger.info(f"Found existing patient ID={patient.id} for worker_id={worker_id}")
                        # Update patient info with new data if available
                        if extracted_data.age and not patient.age:
                            patient.age = extracted_data.age
                        if extracted_data.gender and not patient.gender:
                            patient.gender = extracted_data.gender
                        if extracted_data.village and not patient.village:
                            patient.village = extracted_data.village
                        if extracted_data.phone and not patient.phone:
                            patient.phone = extracted_data.phone
                        if extracted_data.address and not patient.address:
                            patient.address = extracted_data.address
                        if extracted_data.is_pregnant:
                            patient.is_pregnant = True
                        if extracted_data.is_child:
                            patient.is_child = True
                        if extracted_data.known_conditions:
                            existing = patient.known_conditions or []
                            patient.known_conditions = list(set(existing + extracted_data.known_conditions))
                        if extracted_data.allergies:
                            existing = patient.allergies or []
                            patient.allergies = list(set(existing + extracted_data.allergies))
                        if extracted_data.notes and not patient.notes:
                            patient.notes = extracted_data.notes
                        db.commit()
                    else:
                        patient = crud.create_patient(
                            db=db,
                            asha_worker_id=worker_id,
                            name=extracted_data.patient_name.strip(),
                            age=extracted_data.age,
                            gender=extracted_data.gender,
                            village=extracted_data.village,
                            phone=extracted_data.phone,
                            address=extracted_data.address,
                            is_pregnant=extracted_data.is_pregnant,
                            is_child=extracted_data.is_child,
                            known_conditions=extracted_data.known_conditions or [],
                            allergies=extracted_data.allergies or [],
                            notes=extracted_data.notes
                        )
                        logger.info(f"Created new patient ID={patient.id} for worker_id={worker_id}")
                    
                    # Create visit record
                    visit_data = extraction_service.create_visit_data(extracted_data)
                    visit_data["raw_transcript"] = text
                    
                    visit = crud.create_visit(
                        db=db,
                        patient_id=patient.id,
                        asha_worker_id=worker_id,
                        **visit_data
                    )
                    logger.info(f"Created visit record ID={visit.id} for patient_id={patient.id}")
                    
                    # Log conversation (will be updated with AI response later)
                    conv_log = crud.log_conversation(
                        db=db,
                        asha_worker_id=worker_id,
                        message_type="voice" if is_voice else "text",
                        input_text=text,
                        input_language=input_language,  # Use detected language for logging
                        transcription=text if is_voice else None,
                        transcription_confidence=confidence if is_voice else None,
                        voice_file_id=voice_file_id,
                        patient_id=patient.id,
                        visit_id=visit.id
                    )
                    conversation_log_id = int(conv_log.id)
                    
                    # Extract scalar values BEFORE session closes to avoid DetachedInstanceError
                    saved_patient_name = str(patient.name)
                    saved_patient_id = int(patient.id)
                    saved_visit_id = int(visit.id)
                    patient_known_conditions = list(patient.known_conditions) if patient.known_conditions else []
                    
                    patient_saved = True
                    logger.info(f"Successfully saved patient_id={saved_patient_id}, visit_id={saved_visit_id} to database")
                    
            except Exception as db_error:
                import traceback
                logger.error(f"Database error saving patient: {db_error}")
                logger.error(traceback.format_exc())
        else:
            logger.info(f"No patient name extracted from text, skipping DB save. Confidence: {extracted_data.confidence_score}")
        
        # Build conversation history for context
        conversation_history = []
        if patient_saved and saved_patient_name:
            conversation_history.append({
                "role": "system",
                "content": f"Patient context: {saved_patient_name}, known conditions: {patient_known_conditions}"
            })
        
        # Get AI response (with offline fallback)
        ai_response = None
        is_offline = False
        try:
            ai_response = await openai_service.get_chat_response(
                user_message=text,
                conversation_history=conversation_history,
                rag_context=rag_context,
                language=language
            )
        except Exception as api_error:
            logger.warning(f"Azure OpenAI unavailable for response, using offline mode: {api_error}")
            is_offline = True
            # Provide offline fallback messages
            offline_responses = {
                "en": "📴 *Offline Mode* - AI guidance is currently unavailable. Your patient data has been recorded. Please consult a medical professional for health advice.",
                "hi": "📴 *ऑफ़लाइन मोड* - AI मार्गदर्शन अभी उपलब्ध नहीं है। आपका मरीज़ डेटा रिकॉर्ड हो गया है। स्वास्थ्य सलाह के लिए कृपया चिकित्सक से संपर्क करें।",
                "ta": "📴 *ஆஃப்லைன் பயன்முறை* - AI வழிகாட்டுதல் தற்போது கிடைக்கவில்லை. உங்கள் நோயாளி தரவு பதிவு செய்யப்பட்டது. சுகாதார ஆலோசனைக்கு மருத்துவரை அணுகவும்.",
                "ml": "📴 *ഓഫ്‌ലൈൻ മോഡ്* - AI മാർഗ്ഗനിർദ്ദേശം നിലവിൽ ലഭ്യമല്ല. നിങ്ങളുടെ രോഗി ഡാറ്റ രേഖപ്പെടുത്തി. ആരോഗ്യ ഉപദേശത്തിന് ഡോക്ടറെ സമീപിക്കുക."
            }
            ai_response = offline_responses.get(language, offline_responses["en"])
        
        # Update conversation log with AI response
        if conversation_log_id:
            try:
                with get_db_context() as db:
                    crud.update_conversation_log(
                        db=db,
                        conversation_id=conversation_log_id,
                        ai_response=ai_response[:2000] if ai_response else None,  # Truncate if too long
                        response_language=language
                    )
            except Exception as log_err:
                logger.warning(f"Failed to update conversation log with AI response: {log_err}")
        
        # Build final response
        response_parts = []
        
        # Add extracted data summary if patient data was found
        if extracted_data.patient_name and extracted_data.confidence_score > 0.3:
            extraction_summary = extraction_service.format_for_display(extracted_data, language)
            response_parts.append(extraction_summary)
            
            # Confirm patient/visit save with language-specific message
            if patient_saved and saved_visit_id:
                save_messages = {
                    "en": f"\n✅ *Patient saved:* {saved_patient_name} (ID: {saved_patient_id})\n✅ *Visit recorded* (ID: {saved_visit_id})",
                    "hi": f"\n✅ *मरीज़ सेव:* {saved_patient_name} (ID: {saved_patient_id})\n✅ *विज़िट रिकॉर्ड* (ID: {saved_visit_id})",
                    "ta": f"\n✅ *நோயாளி சேமிக்கப்பட்டது:* {saved_patient_name} (ID: {saved_patient_id})\n✅ *வருகை பதிவு செய்யப்பட்டது* (ID: {saved_visit_id})",
                    "ml": f"\n✅ *രോഗി സേവ് ചെയ്തു:* {saved_patient_name} (ID: {saved_patient_id})\n✅ *സന്ദർശനം രേഖപ്പെടുത്തി* (ID: {saved_visit_id})"
                }
                response_parts.append(save_messages.get(language, save_messages["en"]))
        
        # Add AI response (cleaned of markdown symbols)
        response_parts.append(f"\n{clean_ai_response(ai_response)}")
        
        # Add disclaimer in the user's language
        disclaimers = {
            "en": "⚠️ DISCLAIMER: This assistant supports ASHA workers and should not replace licensed medical professionals. Always consult a doctor for medical decisions.",
            "hi": "⚠️ अस्वीकरण: यह सहायक ASHA कार्यकर्ताओं की मदद के लिए है और यह लाइसेंस प्राप्त चिकित्सा पेशेवरों का विकल्प नहीं है। चिकित्सा निर्णयों के लिए हमेशा डॉक्टर से परामर्श करें।",
            "ta": "⚠️ மறுப்பு: இந்த உதவியாளர் ASHA பணியாளர்களுக்கு உதவுவதற்காகவே உள்ளது; இது உரிமம் பெற்ற மருத்துவ நிபுணர்களுக்கு மாற்றாக அல்ல. மருத்துவ முடிவுகளுக்கு எப்போதும் மருத்துவரை அணுகவும்.",
            "ml": "⚠️ നിരാകരണം: ഈ അസിസ്റ്റന്റ് ASHA പ്രവർത്തകരെ സഹായിക്കാൻ മാത്രമുള്ളതാണ്; ഇത് ലൈസൻസുള്ള വൈദ്യ വിദഗ്ധരുടെ പകരക്കാരനല്ല. എല്ലായ്‌പ്പോഴും ഒരു ഡോക്ടറെ കാണുക.",
        }
        disclaimer = disclaimers.get(language, disclaimers["en"])
        response_parts.append(f"\n\n_{disclaimer}_")
        
        # Send response
        final_response = "\n".join(response_parts)
        
        # Helper to send with markdown fallback to plain text
        async def send_with_fallback(text, reply_markup=None):
            try:
                await update.message.reply_text(
                    text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=reply_markup
                )
            except Exception as md_error:
                # Markdown parsing failed, send as plain text
                logger.warning(f"Markdown parsing failed, sending plain text: {md_error}")
                # Remove markdown formatting characters for cleaner display
                plain_text = text.replace('*', '').replace('_', '').replace('`', '')
                await update.message.reply_text(
                    plain_text,
                    reply_markup=reply_markup
                )
        
        # Split if too long
        if len(final_response) > 4000:
            # Send in parts
            await send_with_fallback(final_response[:4000], get_main_menu_keyboard(language))
            await send_with_fallback(final_response[4000:])
        else:
            await send_with_fallback(final_response, get_main_menu_keyboard(language))
        
        # Add feedback + TTS buttons for visit responses
        if ai_response:
            feedback_key = f"{worker_id}_{update.message.message_id}"
            tts_key = f"tts_{update.effective_user.id}_{update.message.message_id}" if settings.tts_enabled else None
            
            # Store context for feedback and TTS
            context.bot_data[f"fb_{feedback_key}"] = {
                "conv_id": None,  # Visit responses don't have separate conv log
                "worker_id": worker_id,
                "question": text[:500],  # Original input
                "response": ai_response[:1000],
                "language": language,
                "tts_key": tts_key  # Keep speak button visible after feedback
            }
            if tts_key:
                context.bot_data[tts_key] = {"text": strip_for_tts(ai_response)[:600], "language": language}
            
            # Send feedback buttons (with TTS if enabled)
            feedback_markup = create_feedback_buttons(
                feedback_key=feedback_key,
                language=language,
                include_tts=settings.tts_enabled,
                tts_key=tts_key
            )
            
            feedback_prompts = {
                "en": "Was this helpful?",
                "hi": "क्या यह सहायक था?",
                "ta": "இது உதவியாக இருந்ததா?",
                "ml": "ഇത് സഹായകരമായിരുന്നോ?"
            }
            await update.message.reply_text(
                feedback_prompts.get(language, feedback_prompts["en"]),
                reply_markup=feedback_markup
            )
        
    except Exception as e:
        logger.error(f"Error processing health update: {e}")
        error_messages = {
            "en": "❌ Sorry, there was an error processing your message. Please try again.",
            "hi": "❌ क्षमा करें, आपके संदेश को प्रोसेस करने में त्रुटि हुई। कृपया पुनः प्रयास करें।",
            "ta": "❌ உங்கள் செய்தியை செயலாக்குவதில் பிழை. மீண்டும் முயற்சிக்கவும்.",
            "ml": "❌ നിങ്ങളുടെ സന്ദേശം പ്രോസസ്സ് ചെയ്യുന്നതിൽ പിശക്. വീണ്ടും ശ്രമിക്കുക."
        }
        await update.message.reply_text(
            error_messages.get(language, error_messages["en"]),
            reply_markup=get_main_menu_keyboard(language)
        )


# ============== TTS Callback Handler ==============

async def tts_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the 🔊 Speak inline button - generate and send voice message."""
    query = update.callback_query
    
    # Answer the callback query (may fail if button click is too old)
    try:
        await query.answer("Generating audio...")
    except Exception:
        pass  # Query expired but we can still try to process the request

    tts_key = query.data
    data = context.bot_data.get(tts_key)
    if not data:
        try:
            await query.answer("⏳ Text expired, please ask again.", show_alert=True)
        except Exception:
            pass
        return

    tts_service = get_tts_service()
    if not tts_service.is_enabled():
        try:
            await query.answer("❌ TTS is disabled.", show_alert=True)
        except Exception:
            pass
        return

    # Show typing indicator while generating
    await update.effective_chat.send_action(ChatAction.RECORD_VOICE)

    audio_bytes = await tts_service.text_to_speech(data["text"], data["language"])
    if audio_bytes:
        import io
        caption_map = {"en": "🔊 Voice", "hi": "🔊 ध्वनि", "ta": "🔊 குரல்", "ml": "🔊 ശബ്ദം"}
        await update.effective_chat.send_voice(
            voice=io.BytesIO(audio_bytes),
            caption=caption_map.get(data["language"], "🔊 Voice")
        )
        # Don't edit the message - keep the buttons intact for re-listening or feedback
        try:
            await query.answer("✅ Voice sent!")
        except Exception:
            pass
        logger.info(f"TTS button: sent {len(audio_bytes)} bytes voice for key={tts_key}")
    else:
        try:
            await query.answer("❌ Could not generate audio. Try again.", show_alert=True)
        except Exception:
            pass
        logger.error(f"TTS button: failed to generate audio for key={tts_key}")


# ============== Feedback Callback Handler ==============

async def feedback_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle feedback button clicks (👍/👎)."""
    query = update.callback_query
    
    # Answer the callback query (may fail if button click is too old)
    try:
        await query.answer()
    except Exception:
        pass  # Query expired but we can still try to process the request
    
    callback_data = query.data  # e.g., "fb_yes_123_456" or "fb_no_123_456"
    parts = callback_data.split("_", 2)  # ["fb", "yes/no", "feedback_key"]
    
    if len(parts) < 3:
        logger.warning(f"Invalid feedback callback data: {callback_data}")
        return
    
    is_helpful = parts[1] == "yes"
    feedback_key = parts[2]
    
    # Get stored context
    fb_data = context.bot_data.get(f"fb_{feedback_key}")
    if not fb_data:
        try:
            await query.edit_message_text("⏳ Feedback session expired.")
        except Exception:
            pass  # Message too old to edit
        return
    
    cache_service = get_cache_service()
    
    try:
        # Store feedback in database
        with get_db_context() as db:
            crud.create_feedback(
                db=db,
                conversation_log_id=fb_data.get("conv_id"),  # None is valid — FK is nullable
                asha_worker_id=fb_data["worker_id"],
                is_helpful=is_helpful,
                original_question=fb_data.get("question"),
                original_response=fb_data.get("response"),
                language=fb_data.get("language")
            )
        logger.info(f"Feedback saved: helpful={is_helpful}, worker={fb_data['worker_id']}")
    except Exception as e:
        logger.error(f"Error saving feedback to DB: {e}")
        try:
            await query.edit_message_text("❌ Could not save feedback. Please try again.")
        except Exception:
            pass
        return

    # Update cache quality metrics (separate try — don't let cache errors block the thank-you)
    if fb_data.get("question"):
        try:
            cache_service.update_feedback(
                fb_data["question"],
                fb_data.get("language", "en"),
                is_helpful
            )
        except Exception as ce:
            logger.warning(f"Cache feedback update failed (non-critical): {ce}")
    
    # Thank user for feedback with a toast notification
    language = fb_data.get("language", "en")

    if is_helpful:
        thanks = {
            "en": "✅ Thank you! Glad it helped.",
            "hi": "✅ धन्यवाद! खुशी हुई कि मददगार था।",
            "ta": "✅ நன்றி! உதவியாக இருந்ததில் மகிழ்ச்சி.",
            "ml": "✅ നന്ദി! സഹായകരമായതിൽ സന്തോഷം."
        }
    else:
        thanks = {
            "en": "📝 Thanks for the feedback!",
            "hi": "📝 प्रतिक्रिया के लिए धन्यवाद!",
            "ta": "📝 கருத்துக்கு நன்றி!",
            "ml": "📝 ഫീഡ്‌ബാക്കിന് നന്ദി!"
        }

    thank_text = thanks.get(language, thanks["en"])

    # Show toast notification ONLY - do NOT edit message (keeps all buttons visible)
    try:
        await query.answer(thank_text, show_alert=True)
    except Exception:
        pass
    
    # Do NOT edit or remove buttons - user can still use TTS button if they want

    # Clean up feedback data (keep TTS data so the speak button still works)
    context.bot_data.pop(f"fb_{feedback_key}", None)


# ============== Error Handler ==============

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
    
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "❌ An error occurred. Please try again or use /help for assistance."
        )


# ============== Daily Follow-up Reminder Job ==============

async def send_daily_followup_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Scheduled job: runs every morning and sends each ASHA worker
    a personalised message listing their follow-ups due today.
    """
    logger.info("Running daily follow-up reminder job...")

    try:
        with get_db_context() as db:
            workers = crud.get_all_active_workers(db)
            # Extract all needed data while session is open
            worker_data = []
            for worker in workers:
                follow_ups = crud.get_follow_ups_due(db, worker.id, days_ahead=1)
                if not follow_ups:
                    continue
                entries = []
                for fu in follow_ups:
                    patient_name = fu.patient.name if fu.patient else "Unknown"
                    symptoms = list(fu.symptoms[:2]) if fu.symptoms else []
                    entries.append({"name": patient_name, "symptoms": symptoms})
                worker_data.append({
                    "telegram_user_id": worker.telegram_user_id,
                    "language": worker.preferred_language or "en",
                    "followups": entries,
                })
    except Exception as e:
        logger.error(f"Daily reminder job: DB error fetching data: {e}")
        return

    sent = 0
    for wd in worker_data:
        try:
            lang = wd["language"]
            followups = wd["followups"]
            count = len(followups)

            # Build greeting line
            greetings = {
                "en": f"🌅 Good morning! You have *{count} follow-up{'s' if count > 1 else ''}* due today:",
                "hi": f"🌅 सुप्रभात! आज आपके *{count} फॉलो-अप* हैं:",
                "ta": f"🌅 காலை வணக்கம்! இன்று உங்களுக்கு *{count} பின்தொடர்தல்கள்* உள்ளன:",
                "ml": f"🌅 സുപ്രഭാതം! ഇന്ന് നിങ്ങൾക്ക് *{count} ഫോളോ-അപ്പ്* ഉണ്ട്:",
            }
            lines = [greetings.get(lang, greetings["en"])]

            # Patient list
            for fu in followups:
                symptom_str = f" ({', '.join(fu['symptoms'])})" if fu["symptoms"] else ""
                lines.append(f"• *{fu['name']}*{symptom_str}")

            # Footer
            footers = {
                "en": "\nTap /followups for full details. Stay safe! 💪",
                "hi": "\nपूरी जानकारी के लिए /followups टैप करें। सुरक्षित रहें! 💪",
                "ta": "\nமுழு விவரங்களுக்கு /followups தட்டவும். பாதுகாப்பாக இருங்கள்! 💪",
                "ml": "\nകൂടുതൽ വിവരങ്ങൾക്ക് /followups ടാപ്പ് ചെയ്യൂ. സുരക്ഷിതരായിരിക്കൂ! 💪",
            }
            lines.append(footers.get(lang, footers["en"]))

            message = "\n".join(lines)

            await context.bot.send_message(
                chat_id=wd["telegram_user_id"],
                text=message,
                parse_mode="Markdown",
            )
            sent += 1
            logger.info(f"Sent follow-up reminder to worker {wd['telegram_user_id']} ({count} items)")
        except Exception as e:
            logger.warning(f"Could not send reminder to {wd['telegram_user_id']}: {e}")

    logger.info(f"Daily reminder job done — sent to {sent}/{len(worker_data)} workers")


# ============== Setup Function ==============

def setup_handlers(application: Application) -> None:
    """Setup all handlers for the Telegram bot."""
    
    # Command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("language", language_command))
    application.add_handler(CommandHandler("patients", patients_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("followups", followups_command))
    
    # Voice message handler
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    
    # Text message handler (must be last to catch all text)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    # TTS speak button callback
    application.add_handler(CallbackQueryHandler(tts_button_callback, pattern=r"^tts_"))
    
    # Feedback button callback
    application.add_handler(CallbackQueryHandler(feedback_button_callback, pattern=r"^fb_"))

    # Error handler
    application.add_error_handler(error_handler)
    
    logger.info("All handlers registered successfully")
