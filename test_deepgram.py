"""
Test Deepgram STT - Terminal Test
Records from microphone and transcribes using Deepgram API.
"""

import os
import io
import wave
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("  Deepgram STT Test for ASHA Bot")
print("=" * 60)

# Check for API key
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    print("❌ ERROR: DEEPGRAM_API_KEY not found in environment!")
    exit(1)
else:
    print(f"✅ Deepgram API Key found: {DEEPGRAM_API_KEY[:10]}...")

print("\n[1/3] Loading audio libraries...")
try:
    import pyaudio
    import numpy as np
    print("  ✅ PyAudio and NumPy loaded")
except ImportError as e:
    print(f"  ⚠️ PyAudio not available: {e}")
    print("  Will test with a sample audio file instead")
    pyaudio = None

print("[2/3] Loading HTTP client...")
import httpx
print("  ✅ httpx loaded")

print("[3/3] Ready!\n")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5
MAX_RECORD_SECONDS = 10


def record_audio() -> bytes:
    """Record audio from microphone until silence."""
    if pyaudio is None:
        print("❌ PyAudio not available - cannot record")
        return b""
    
    pa = pyaudio.PyAudio()
    
    # Find input device
    input_device = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] >= 1:
            try:
                stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                               input=True, input_device_index=i, frames_per_buffer=CHUNK)
                stream.stop_stream()
                stream.close()
                input_device = i
                print(f"  Using microphone: {info['name']}")
                break
            except:
                continue
    
    if input_device is None:
        print("❌ No working microphone found!")
        pa.terminate()
        return b""
    
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CHUNK,
    )
    
    frames = []
    silent_chunks = 0
    speaking_started = False
    max_chunks = int(MAX_RECORD_SECONDS * SAMPLE_RATE / CHUNK)
    silence_chunks_threshold = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)
    
    print("\n🎤 Listening... speak now (max 10 seconds)")
    print("   (Stop speaking for 1.5s to finish recording)\n")
    
    for _ in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Calculate RMS
        samples = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(float)**2))
        
        # Visual indicator
        level = int(rms / 100)
        bar = "█" * min(level, 30)
        print(f"\r  Level: {bar:<30} RMS={rms:.0f}", end="", flush=True)
        
        if rms > SILENCE_THRESHOLD:
            speaking_started = True
            silent_chunks = 0
        elif speaking_started:
            silent_chunks += 1
            if silent_chunks >= silence_chunks_threshold:
                print("\n  ✅ Silence detected - stopping recording")
                break
    
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    if not speaking_started:
        print("\n  ⚠️ No speech detected!")
        return b""
    
    audio_data = b"".join(frames)
    print(f"  📊 Recorded {len(audio_data)} bytes ({len(frames)} frames)")
    return audio_data


def audio_to_wav(audio_data: bytes) -> bytes:
    """Convert raw PCM to WAV format."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)
    return wav_buffer.getvalue()


async def transcribe_with_deepgram(audio_data: bytes, language: str = None) -> str:
    """Transcribe audio using Deepgram REST API."""
    
    # Convert to WAV
    wav_data = audio_to_wav(audio_data)
    print(f"\n📤 Sending to Deepgram ({len(wav_data)} bytes WAV)")
    
    # For Tamil/Malayalam, use whisper model (better multilingual support)
    # For English/Hindi, use nova-2
    if language in ["ta", "ml"]:
        # Use whisper for better Tamil/Malayalam support
        params = {
            "model": "whisper-large",
            "language": language,
            "smart_format": "true",
            "punctuate": "true",
        }
        print(f"   Language: {language}, Model: whisper-large")
    elif language:
        params = {
            "model": "nova-2",
            "language": language,
            "smart_format": "true",
            "punctuate": "true",
        }
        print(f"   Language: {language}, Model: nova-2")
    else:
        # Auto-detect with nova-2 or fallback
        params = {
            "model": "nova-2",
            "detect_language": "true",
            "smart_format": "true",
            "punctuate": "true",
        }
        print(f"   Language: auto-detect, Model: nova-2")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "audio/wav",
                },
                content=wav_data,
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                return ""
            
            result = response.json()
            
            # Debug: print full response structure
            print("\n📥 Response received:")
            
            if result.get("results") and result["results"].get("channels"):
                channel = result["results"]["channels"][0]
                if channel.get("alternatives"):
                    alt = channel["alternatives"][0]
                    transcript = alt.get("transcript", "")
                    confidence = alt.get("confidence", 0)
                    detected_lang = channel.get("detected_language", "unknown")
                    
                    print(f"   Detected language: {detected_lang}")
                    print(f"   Confidence: {confidence:.2%}")
                    print(f"   Transcript: '{transcript}'")
                    
                    return transcript
            
            print("   ⚠️ No transcript in response")
            print(f"   Raw response: {result}")
            return ""
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return ""


async def main():
    print("\n" + "=" * 60)
    print("  TEST 1: Record and transcribe (auto language detection)")
    print("=" * 60)
    
    audio = record_audio()
    if audio:
        transcript = await transcribe_with_deepgram(audio)
        if transcript:
            print(f"\n✅ SUCCESS! Transcript: '{transcript}'")
        else:
            print("\n❌ No transcript returned")
    
    print("\n" + "=" * 60)
    print("  TEST 2: Record and transcribe (Tamil - ta)")
    print("=" * 60)
    
    input("\nPress Enter to record Tamil speech...")
    audio = record_audio()
    if audio:
        transcript = await transcribe_with_deepgram(audio, language="ta")
        if transcript:
            print(f"\n✅ SUCCESS! Transcript: '{transcript}'")
        else:
            print("\n❌ No transcript returned")
    
    print("\n" + "=" * 60)
    print("  TEST 3: Record and transcribe (Hindi - hi)")
    print("=" * 60)
    
    input("\nPress Enter to record Hindi speech...")
    audio = record_audio()
    if audio:
        transcript = await transcribe_with_deepgram(audio, language="hi")
        if transcript:
            print(f"\n✅ SUCCESS! Transcript: '{transcript}'")
        else:
            print("\n❌ No transcript returned")
    
    print("\n" + "=" * 60)
    print("  All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
