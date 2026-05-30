"""
Text-to-Speech Service using Google TTS (gTTS) - Free.
Converts AI responses to voice messages in multiple Indian languages.
"""

import io
import logging
import asyncio
import tempfile
import os
from typing import Optional
from functools import partial

from gtts import gTTS

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Language code mapping for gTTS
GTTS_LANGUAGE_MAP = {
    "en": "en",   # English
    "hi": "hi",   # Hindi
    "ta": "ta",   # Tamil
    "ml": "ml",   # Malayalam
}

# TLD for accent
GTTS_TLD_MAP = {
    "en": "co.in",  # Indian English accent
    "hi": "com",
    "ta": "com",
    "ml": "com",
}


class TTSService:
    """
    Text-to-Speech service using Google TTS (gTTS) - 100% free.
    Supports English, Hindi, Tamil, and Malayalam.
    """

    def __init__(self):
        """Initialize gTTS service."""
        self.enabled = settings.tts_enabled
        if self.enabled:
            try:
                from gtts import gTTS as _check
                logger.info("Google TTS (gTTS) service initialized successfully")
            except ImportError:
                logger.error("gTTS not installed. Run: pip install gTTS")
                self.enabled = False
        else:
            logger.info("TTS disabled in settings (set TTS_ENABLED=true to enable)")

    def is_enabled(self) -> bool:
        """Check if TTS service is available."""
        return self.enabled

    def _generate_sync(self, text: str, lang: str, tld: str) -> Optional[bytes]:
        """Synchronous gTTS call - runs in thread executor."""
        try:
            tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            logger.error(f"gTTS generation failed: {e}")
            return None

    async def text_to_speech(
        self,
        text: str,
        language: str = "en",
    ) -> Optional[bytes]:
        """
        Convert text to speech audio using Google TTS.

        Args:
            text: Text to convert to speech
            language: Target language (en, hi, ta, ml)

        Returns:
            Audio bytes in MP3 format, or None if failed
        """
        if not self.enabled:
            return None
        if not text or len(text.strip()) < 5:
            return None

        lang = GTTS_LANGUAGE_MAP.get(language, "en")
        tld = GTTS_TLD_MAP.get(language, "com")

        # Limit text length for faster response (~600 chars ≈ 30 seconds)
        text = text.strip()[:600]

        logger.info(f"Generating TTS via gTTS: lang={lang}, chars={len(text)}")

        try:
            loop = asyncio.get_event_loop()
            func = partial(self._generate_sync, text, lang, tld)
            audio_bytes = await loop.run_in_executor(None, func)

            if audio_bytes:
                logger.info(f"gTTS generated {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    async def text_to_speech_file(
        self,
        text: str,
        language: str = "en",
        file_path: Optional[str] = None
    ) -> Optional[str]:
        """Convert text to speech and save to file."""
        audio_bytes = await self.text_to_speech(text, language)
        if not audio_bytes:
            return None
        try:
            if file_path is None:
                fd, file_path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
            return file_path
        except Exception as e:
            logger.error(f"Failed to save TTS file: {e}")
            return None


# Singleton instance
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get singleton TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
