"""
Deepgram Speech-to-Text Service for ASHA AI Assistant.
Handles voice note transcription using Deepgram's nova-2 model.
Uses direct HTTP API calls for reliability.
"""

import io
import logging
import wave
from typing import Optional, Tuple
import httpx

# PyAV for OGG to WAV conversion (bundles its own FFmpeg - no system FFmpeg needed)
try:
    import av as pyav
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

# pydub fallback — set FFmpeg path explicitly (winget installs here)
try:
    from pydub import AudioSegment
    import os as _os
    _ffmpeg_bin = r"C:\Users\ASUS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
    if _os.path.isdir(_ffmpeg_bin):
        AudioSegment.converter = _os.path.join(_ffmpeg_bin, "ffmpeg.exe")
        AudioSegment.ffprobe   = _os.path.join(_ffmpeg_bin, "ffprobe.exe")
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DeepgramService:
    """Service for speech-to-text using Deepgram nova-2 model via HTTP API."""
    
    def __init__(self):
        """Initialize Deepgram service."""
        self.api_key = settings.deepgram_api_key
        self.base_url = "https://api.deepgram.com/v1/listen"
        
        # Language mappings for Deepgram
        self.language_mapping = {
            "en": "en",
            "hi": "hi",
            "ta": "ta",
            "ml": "ml",
            "english": "en",
            "hindi": "hi", 
            "tamil": "ta",
            "malayalam": "ml"
        }
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        mime_type: str = "audio/ogg"
    ) -> Tuple[str, float, str]:
        """
        Transcribe audio data to text using Deepgram REST API directly.
        
        Args:
            audio_data: Raw audio bytes (from Telegram voice note)
            language: Language code for transcription (en, hi, ta, ml)
            mime_type: MIME type of the audio file
        
        Returns:
            Tuple of (transcript, confidence, detected_language)
        """
        try:
            # Determine model based on language
            # nova-2 has limited support for Indian languages
            # Use whisper-large for Hindi, Tamil, Malayalam (better multilingual support)
            dg_language = self.language_mapping.get(language, None) if language else None
            
            if dg_language in ["hi", "ta", "ml"]:
                # Hindi, Tamil and Malayalam use whisper model
                # (nova-2 often returns empty transcriptions for these languages)
                model = "whisper-large"
                params = {
                    "model": model,
                    "language": dg_language,
                    "smart_format": "true",
                    "punctuate": "true",
                }
            elif dg_language:
                # English and Hindi work with nova-2
                model = "nova-2"
                params = {
                    "model": model,
                    "language": dg_language,
                    "smart_format": "true",
                    "punctuate": "true",
                }
            else:
                # Auto-detect language
                model = "nova-2"
                params = {
                    "model": model,
                    "detect_language": "true",
                    "smart_format": "true",
                    "punctuate": "true",
                }
            
            logger.info(f"Sending audio to Deepgram (size: {len(audio_data)} bytes, mime: {mime_type}, model: {model}, lang: {dg_language or 'auto'})")
            
            # Make HTTP request to Deepgram API
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.base_url,
                    params=params,
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": mime_type,
                    },
                    content=audio_data,
                )
                
                if response.status_code != 200:
                    logger.error(f"Deepgram API error: {response.status_code} - {response.text}")
                    raise Exception(f"Deepgram API error: {response.status_code}")
                
                result = response.json()
            
            # Extract results
            if result.get("results") and result["results"].get("channels"):
                channel = result["results"]["channels"][0]
                if channel.get("alternatives"):
                    alternative = channel["alternatives"][0]
                    transcript = alternative.get("transcript", "")
                    confidence = alternative.get("confidence", 0.0)
                    
                    # Get detected language
                    detected_language = channel.get("detected_language", language or "en")
                    
                    logger.info(f"Transcription successful: {len(transcript)} chars, confidence: {confidence:.2f}")
                    
                    return transcript, confidence, detected_language
            
            logger.warning("No transcription results returned")
            return "", 0.0, "en"
            
        except Exception as e:
            logger.error(f"Deepgram transcription error: {e}")
            raise

    async def transcribe_telegram_voice(
        self,
        voice_file_bytes: bytes,
        language: Optional[str] = None
    ) -> Tuple[str, float, str]:
        """
        Transcribe a Telegram voice note.
        Telegram voice notes are OGG/Opus format (.oga files).
        We convert OGG → WAV using pydub before sending to Deepgram.
        
        Args:
            voice_file_bytes: Voice file bytes downloaded from Telegram
            language: Expected language code
        
        Returns:
            Tuple of (transcript, confidence, detected_language)
        """
        import traceback
        
        logger.info(f"Transcribing Telegram voice: {len(voice_file_bytes)} bytes, language: {language}")
        
        # PRIMARY METHOD: Convert OGG to WAV using PyAV (bundles FFmpeg - no system install needed)
        if PYAV_AVAILABLE:
            try:
                logger.info("Converting OGG to WAV using PyAV...")
                ogg_buffer = io.BytesIO(voice_file_bytes)

                # Decode OGG/Opus → raw PCM samples
                input_container = pyav.open(ogg_buffer, format="ogg")
                resampler = pyav.AudioResampler(format="s16", layout="mono", rate=16000)

                pcm_frames = []
                for frame in input_container.decode(audio=0):
                    for resampled in resampler.resample(frame):
                        pcm_frames.append(bytes(resampled.planes[0]))
                # flush resampler
                for resampled in resampler.resample(None):
                    pcm_frames.append(bytes(resampled.planes[0]))
                input_container.close()

                pcm_data = b"".join(pcm_frames)

                # Wrap PCM in WAV header
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)   # s16 = 2 bytes
                    wf.setframerate(16000)
                    wf.writeframes(pcm_data)
                wav_bytes = wav_buffer.getvalue()

                logger.info(f"PyAV converted OGG ({len(voice_file_bytes)}B) → WAV ({len(wav_bytes)}B)")
                return await self.transcribe_audio(
                    audio_data=wav_bytes,
                    language=language,
                    mime_type="audio/wav"
                )
            except Exception as e:
                import traceback as _tb
                logger.warning(f"PyAV OGG→WAV failed: {e}\n{_tb.format_exc()}")

        # SECONDARY METHOD: pydub (requires system FFmpeg)
        if PYDUB_AVAILABLE:
            try:
                logger.info("Converting OGG to WAV using pydub...")
                
                # Load OGG from bytes
                ogg_buffer = io.BytesIO(voice_file_bytes)
                audio = AudioSegment.from_ogg(ogg_buffer)
                
                # Convert to WAV (mono, 16kHz for better STT)
                audio = audio.set_channels(1).set_frame_rate(16000)
                
                # Export to WAV bytes
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                wav_bytes = wav_buffer.read()
                
                logger.info(f"Converted OGG ({len(voice_file_bytes)} bytes) to WAV ({len(wav_bytes)} bytes)")
                
                # Send WAV to Deepgram
                return await self.transcribe_audio(
                    audio_data=wav_bytes,
                    language=language,
                    mime_type="audio/wav"
                )
            except Exception as e:
                logger.warning(f"OGG→WAV conversion failed: {e}. Falling back to direct OGG...")
                logger.warning(traceback.format_exc())
        else:
            logger.warning("pydub not available. Install pydub and FFmpeg for best results.")
        
        # FALLBACK: Try different MIME types for OGG Opus directly
        mime_types_to_try = [
            "audio/ogg",
            "audio/ogg; codecs=opus",
            "audio/opus",
            "audio/webm",
        ]
        
        last_error = None
        for mime_type in mime_types_to_try:
            try:
                logger.info(f"Trying MIME type: {mime_type}")
                result = await self.transcribe_audio(
                    audio_data=voice_file_bytes,
                    language=language,
                    mime_type=mime_type
                )
                # If we get here, transcription succeeded
                if result[0]:  # If transcript is not empty
                    return result
                logger.warning(f"Empty transcript with MIME type: {mime_type}")
            except Exception as e:
                logger.warning(f"Failed with MIME type {mime_type}: {e}")
                last_error = e
                continue
        
        # All methods failed
        logger.error(f"All transcription methods failed. Last error: {last_error}")
        logger.error(traceback.format_exc())
        raise last_error or Exception("Failed to transcribe audio with all methods")

    def convert_to_wav(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2
    ) -> bytes:
        """
        Convert raw PCM audio data to WAV format.
        
        Args:
            pcm_data: Raw PCM audio bytes
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            sample_width: Sample width in bytes (2 for 16-bit)
        
        Returns:
            WAV formatted audio bytes
        """
        try:
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            
            wav_buffer.seek(0)
            return wav_buffer.read()
            
        except Exception as e:
            logger.error(f"Error converting to WAV: {e}")
            raise

    async def transcribe_wav(
        self,
        wav_data: bytes,
        language: Optional[str] = None
    ) -> Tuple[str, float, str]:
        """
        Transcribe WAV audio data.
        
        Args:
            wav_data: WAV formatted audio bytes
            language: Expected language code
        
        Returns:
            Tuple of (transcript, confidence, detected_language)
        """
        return await self.transcribe_audio(
            audio_data=wav_data,
            language=language,
            mime_type="audio/wav"
        )

    def get_supported_languages(self) -> dict:
        """Get dictionary of supported languages."""
        return {
            "en": "English",
            "hi": "Hindi (हिन्दी)",
            "ta": "Tamil (தமிழ்)",
            "ml": "Malayalam (മലയാളം)"
        }


# Singleton instance
_deepgram_service: Optional[DeepgramService] = None


def get_deepgram_service() -> DeepgramService:
    """Get singleton Deepgram service instance."""
    global _deepgram_service
    if _deepgram_service is None:
        _deepgram_service = DeepgramService()
    return _deepgram_service
