"""
Quick terminal test for Google TTS (gTTS).
Run: python test_tts.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


async def test():
    from app.services.tts_service import get_tts_service

    tts = get_tts_service()
    print(f"TTS enabled: {tts.is_enabled()}")
    if not tts.is_enabled():
        print("ERROR: TTS is disabled. Set TTS_ENABLED=true in .env")
        return

    tests = [
        ("en", "Anemia is a condition where you don't have enough healthy red blood cells to carry oxygen. Symptoms include fatigue, pale skin, and shortness of breath."),
        ("ml", "അനീമിയ ഒരു അവസ്ഥയാണ്, അതിൽ ശരീരത്തിൽ ആവശ്യത്തിന് ചുവന്ന രക്തകോശങ്ങൾ ഇല്ല."),
        ("hi", "एनीमिया एक ऐसी स्थिति है जिसमें शरीर में पर्याप्त लाल रक्त कोशिकाएं नहीं होती हैं।"),
    ]

    for lang, text in tests:
        print(f"\nTesting {lang}: '{text[:50]}...'")
        audio = await tts.text_to_speech(text, lang)
        if audio:
            fname = f"test_tts_{lang}.mp3"
            with open(fname, "wb") as f:
                f.write(audio)
            print(f"  ✅ Generated {len(audio)} bytes → {fname}")
            os.startfile(os.path.abspath(fname))
            await asyncio.sleep(1)
        else:
            print(f"  ❌ FAILED for {lang}")

    print("\nDone! Audio files should be playing.")


asyncio.run(test())
