#!/usr/bin/env python3
"""
Seed the response cache with common healthcare questions and answers.
This enables offline mode for ASHA workers in areas with poor connectivity.

Run this once after deployment: python seed_cache.py
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database.connection import get_db_context, init_db
from app.database import crud
from app.services.cache_service import get_cache_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-defined healthcare Q&A for offline mode
# These are evidence-based responses from NHM guidelines
SEED_RESPONSES = {
    "en": [
        {
            "question": "What are the danger signs during pregnancy?",
            "response": """**Danger Signs During Pregnancy - Seek Immediate Medical Help:**

1. **Severe headache** with blurred vision (possible preeclampsia)
2. **Heavy vaginal bleeding** (threatened abortion or placenta previa)
3. **Severe abdominal pain** (ectopic pregnancy or abruption)
4. **High fever** (>100.4°F/38°C) with chills
5. **Swelling** of face, hands, or feet (preeclampsia sign)
6. **Reduced or no fetal movement** after 20 weeks
7. **Water breaking** before 37 weeks (preterm labor)
8. **Convulsions/fits** (eclampsia - EMERGENCY)

**Action:** If any of these signs appear, take the woman to the nearest health facility IMMEDIATELY. Call 108 for ambulance.

Source: NHM Safe Motherhood Guidelines"""
        },
        {
            "question": "What is the immunization schedule for infants?",
            "response": """**Immunization Schedule for Infants (National Immunization Schedule):**

**At Birth:**
- BCG (Tuberculosis)
- OPV-0 (Oral Polio Vaccine)
- Hepatitis B - Birth dose

**At 6 Weeks:**
- OPV-1, Pentavalent-1, Rotavirus-1, fIPV-1, PCV-1

**At 10 Weeks:**
- OPV-2, Pentavalent-2, Rotavirus-2

**At 14 Weeks:**
- OPV-3, Pentavalent-3, Rotavirus-3, fIPV-2, PCV-2

**At 9-12 Months:**
- Measles-Rubella (MR-1), Vitamin A (1st dose), PCV Booster

**At 16-24 Months:**
- DPT Booster-1, OPV Booster, MR-2, Vitamin A (2nd dose)

**Important:** Keep immunization card safe. Never miss a dose!

Source: NHM Universal Immunization Programme"""
        },
        {
            "question": "How to manage fever in children?",
            "response": """**Managing Fever in Children:**

**Step 1: Check Temperature**
- Normal: 97-99°F (36-37°C)
- Fever: Above 100.4°F (38°C)
- High Fever: Above 103°F (39.4°C) - SEEK CARE

**Home Care for Mild Fever:**
1. Give plenty of fluids (water, ORS, breast milk)
2. Dress child in light clothing
3. Sponge with lukewarm water (NOT cold)
4. Give Paracetamol syrup as per weight (10-15mg/kg)

**Danger Signs - Go to Hospital:**
- Fever lasting more than 3 days
- Child is very drowsy or difficult to wake
- Convulsions/fits
- Fast breathing
- Unable to drink/breastfeed
- Severe vomiting

**DO NOT:** Give Aspirin to children. Do not over-bundle the child.

Source: IMNCI Guidelines"""
        },
        {
            "question": "What are the symptoms of hypertension?",
            "response": """**Hypertension (High Blood Pressure) - Silent Killer:**

**Often No Symptoms!** That's why regular BP checks are important.

**When Symptoms Occur:**
- Severe headache
- Dizziness or lightheadedness
- Blurred vision
- Chest pain
- Difficulty breathing
- Nosebleeds
- Blood in urine

**BP Categories:**
- Normal: <120/80 mmHg
- Elevated: 120-129/<80 mmHg
- High (Stage 1): 130-139/80-89 mmHg
- High (Stage 2): ≥140/90 mmHg
- Crisis: >180/120 mmHg (EMERGENCY)

**For Pregnant Women:**
BP ≥140/90 with headache, swelling, or vision changes = Possible Preeclampsia. URGENT REFERRAL.

**Management:**
- Reduce salt intake
- Regular exercise
- Maintain healthy weight
- Take prescribed medicines regularly

Source: NHM Hypertension Guidelines"""
        },
        {
            "question": "How to prepare and give ORS?",
            "response": """**ORS (Oral Rehydration Solution) Preparation:**

**Using ORS Packet:**
1. Wash hands with soap
2. Take 1 litre of clean drinking water
3. Pour entire ORS packet into water
4. Stir well until dissolved
5. Use within 24 hours, discard after

**If No ORS Packet (Home-made Solution):**
- 1 litre clean water
- 6 level teaspoons sugar
- 1/2 level teaspoon salt
- Mix well

**How to Give:**
- Give small sips frequently
- Use spoon or cup (not bottle)
- For children: Give 50-100ml after each loose stool
- For adults: Give as much as wanted

**Continue:**
- Breastfeeding (for infants)
- Normal food (don't stop feeding)

**Danger Signs - Go to Hospital:**
- Unable to drink
- Sunken eyes
- Very drowsy
- Blood in stool
- No urine for 6+ hours

Source: IMNCI Diarrhoea Management"""
        },
        {
            "question": "What are anemia symptoms and treatment?",
            "response": """**Anemia - Low Hemoglobin/Blood:**

**Symptoms:**
- Pale skin, nails, and inner eyelids
- Extreme tiredness/weakness
- Dizziness or lightheadedness
- Shortness of breath
- Fast heartbeat
- Cold hands and feet
- Brittle nails
- Headache

**Common Causes:**
- Iron deficiency (most common)
- Worm infestation
- Heavy menstrual bleeding
- Poor diet
- Pregnancy (increased need)

**Treatment:**
1. **Iron tablets:** IFA (Iron + Folic Acid)
   - Pregnant women: 1 tablet daily
   - Adolescent girls: 1 tablet weekly
   - Take with Vitamin C (lemon water) for better absorption
   - Don't take with tea/milk (reduces absorption)

2. **Diet:** Eat iron-rich foods
   - Green leafy vegetables (spinach, amaranth)
   - Jaggery, dates, raisins
   - Eggs, meat, fish
   - Pulses and beans

3. **Deworming:** Every 6 months

**Severe Anemia (Hb <7):** Needs hospital care, may need blood transfusion.

Source: NHM Anemia Mukt Bharat Guidelines"""
        },
        {
            "question": "How to care for newborn baby?",
            "response": """**Essential Newborn Care:**

**Immediately After Birth:**
1. **Warmth:** Dry baby, skin-to-skin with mother
2. **Breathing:** Clear airway if needed
3. **Breastfeeding:** Within 1 hour of birth
4. **Cord care:** Keep clean and dry, no application needed

**Daily Care:**
- **Feeding:** Only breastmilk for 6 months
  - Feed 8-12 times/day
  - No water, honey, or other feeds
- **Keep Warm:** Kangaroo care, warm clothing
- **Hygiene:** Keep cord stump dry and clean
- **Sleep:** On back, on firm surface

**Danger Signs - Seek Care Immediately:**
- Not breastfeeding well
- Fever (>37.5°C) or too cold (<36°C)
- Fast breathing (>60/min)
- Chest in-drawing
- Yellow skin in first 24 hours
- Convulsions
- Bleeding from cord
- Pus or redness around cord

**Immunization:** BCG, OPV-0, Hep-B at birth

Source: Home Based Newborn Care Guidelines"""
        },
        {
            "question": "What is diabetes and its symptoms?",
            "response": """**Diabetes - High Blood Sugar:**

**Types:**
- Type 1: Body doesn't make insulin (usually starts young)
- Type 2: Body doesn't use insulin well (more common, adults)
- Gestational: During pregnancy

**Symptoms:**
- Frequent urination (especially at night)
- Excessive thirst
- Unexplained weight loss
- Extreme hunger
- Blurred vision
- Slow-healing wounds
- Tingling in hands/feet
- Fatigue

**Testing:**
- Fasting Blood Sugar: Normal <100 mg/dL
- Random Blood Sugar: Normal <140 mg/dL
- Diabetes: Fasting ≥126 or Random ≥200 mg/dL

**Management:**
1. **Diet:** Avoid sweets, white rice, maida
   - Eat vegetables, whole grains, protein
2. **Exercise:** 30 min daily walking
3. **Medicines:** Take as prescribed, don't skip
4. **Monitoring:** Regular blood sugar checks
5. **Foot care:** Check feet daily, wear proper footwear

**Complications if Uncontrolled:**
Heart disease, kidney damage, eye problems, nerve damage, foot ulcers

Source: NHM NCD Guidelines"""
        },
    ],
    "hi": [
        {
            "question": "गर्भावस्था में खतरे के लक्षण क्या हैं?",
            "response": """**गर्भावस्था में खतरे के लक्षण - तुरंत अस्पताल जाएं:**

1. **तेज सिरदर्द** धुंधली नज़र के साथ (प्रीक्लेम्पसिया)
2. **योनि से अधिक रक्तस्राव** (खून बहना)
3. **तेज पेट दर्द**
4. **तेज बुखार** (100.4°F/38°C से ऊपर) ठंड के साथ
5. **चेहरे, हाथ या पैर में सूजन**
6. **बच्चे की हलचल कम या बंद** (20 सप्ताह के बाद)
7. **पानी की थैली फटना** 37 सप्ताह से पहले
8. **दौरे/ऐंठन** - आपातकाल

**क्या करें:** इनमें से कोई भी लक्षण दिखे तो महिला को तुरंत नजदीकी अस्पताल ले जाएं। 108 पर कॉल करें।

स्रोत: NHM सुरक्षित मातृत्व दिशानिर्देश"""
        },
        {
            "question": "बच्चों में बुखार का इलाज कैसे करें?",
            "response": """**बच्चों में बुखार का प्रबंधन:**

**तापमान जांचें:**
- सामान्य: 97-99°F (36-37°C)
- बुखार: 100.4°F (38°C) से ऊपर
- तेज बुखार: 103°F (39.4°C) से ऊपर - डॉक्टर के पास जाएं

**घरेलू देखभाल:**
1. खूब पानी, ORS, मां का दूध दें
2. हल्के कपड़े पहनाएं
3. गुनगुने पानी से पोंछें (ठंडा पानी नहीं)
4. वज़न के अनुसार पैरासिटामोल सिरप दें

**खतरे के लक्षण - अस्पताल जाएं:**
- 3 दिन से ज्यादा बुखार
- बच्चा बहुत सुस्त या जगता नहीं
- दौरे/ऐंठन
- तेज सांस
- कुछ पी या खा नहीं पाता
- ज्यादा उल्टी

**न करें:** बच्चों को एस्पिरिन न दें।

स्रोत: IMNCI दिशानिर्देश"""
        },
        {
            "question": "ORS कैसे बनाएं और दें?",
            "response": """**ORS (ओआरएस) बनाने की विधि:**

**ORS पैकेट से:**
1. हाथ साबुन से धोएं
2. 1 लीटर साफ पीने का पानी लें
3. पूरा ORS पैकेट पानी में डालें
4. अच्छी तरह मिलाएं
5. 24 घंटे में इस्तेमाल करें, बाद में फेंक दें

**घर पर बनाएं (ORS न हो तो):**
- 1 लीटर साफ पानी
- 6 सपाट चम्मच चीनी
- आधा सपाट चम्मच नमक
- अच्छे से मिलाएं

**कैसे दें:**
- थोड़ा-थोड़ा बार-बार पिलाएं
- चम्मच या कप से दें (बोतल नहीं)
- बच्चों को: हर दस्त के बाद 50-100ml
- बड़ों को: जितना चाहें

**जारी रखें:**
- स्तनपान (शिशुओं के लिए)
- सामान्य भोजन (खाना बंद न करें)

**खतरे के लक्षण:**
- कुछ पी नहीं पाता
- आंखें धंसी हुई
- बहुत सुस्त
- पेशाब 6+ घंटे से नहीं

स्रोत: IMNCI दस्त प्रबंधन"""
        },
    ],
    "ta": [
        {
            "question": "கர்ப்ப காலத்தில் ஆபத்தான அறிகுறிகள் என்ன?",
            "response": """**கர்ப்ப காலத்தில் ஆபத்தான அறிகுறிகள் - உடனடியாக மருத்துவமனை செல்லுங்கள்:**

1. **கடுமையான தலைவலி** மங்கலான பார்வையுடன்
2. **அதிக இரத்தப்போக்கு**
3. **கடுமையான வயிற்று வலி**
4. **அதிக காய்ச்சல்** (100.4°F/38°C மேல்) குளிருடன்
5. **முகம், கை அல்லது கால்களில் வீக்கம்**
6. **குழந்தையின் அசைவு குறைவு அல்லது இல்லை** (20 வாரங்களுக்குப் பிறகு)
7. **பனிக்குடம் உடைதல்** 37 வாரங்களுக்கு முன்
8. **வலிப்பு** - அவசரநிலை

**என்ன செய்ய வேண்டும்:** இவற்றில் ஏதேனும் அறிகுறி தென்பட்டால், உடனடியாக அருகிலுள்ள மருத்துவமனைக்கு அழைத்துச் செல்லுங்கள். 108 அழைக்கவும்.

ஆதாரம்: NHM பாதுகாப்பான தாய்மை வழிகாட்டுதல்கள்"""
        },
    ],
    "ml": [
        {
            "question": "ഗർഭകാലത്തെ അപകട ലക്ഷണങ്ങൾ എന്തൊക്കെ?",
            "response": """**ഗർഭകാലത്തെ അപകട ലക്ഷണങ്ങൾ - ഉടൻ ആശുപത്രിയിൽ പോകുക:**

1. **കഠിനമായ തലവേദന** കാഴ്ച മങ്ങലോടെ
2. **അധിക രക്തസ്രാവം**
3. **കഠിനമായ വയറുവേദന**
4. **കടുത്ത പനി** (100.4°F/38°C-ൽ കൂടുതൽ) തണുപ്പോടെ
5. **മുഖം, കൈ അല്ലെങ്കിൽ കാലിൽ നീർക്കെട്ട്**
6. **കുഞ്ഞിന്റെ ചലനം കുറവ് അല്ലെങ്കിൽ ഇല്ല** (20 ആഴ്ചയ്ക്ക് ശേഷം)
7. **വെള്ളം പൊട്ടുക** 37 ആഴ്ചയ്ക്ക് മുമ്പ്
8. **അപസ്മാരം** - അടിയന്തിരം

**എന്ത് ചെയ്യണം:** ഇവയിൽ ഏതെങ്കിലും ലക്ഷണം കണ്ടാൽ, ഉടൻ അടുത്തുള്ള ആശുപത്രിയിലേക്ക് കൊണ്ടുപോകുക. 108 വിളിക്കുക.

ഉറവിടം: NHM സുരക്ഷിത മാതൃത്വ മാർഗ്ഗനിർദ്ദേശങ്ങൾ"""
        },
    ],
}


def seed_cache():
    """Seed the response cache with common healthcare Q&A."""
    logger.info("Starting cache seeding...")
    
    # Initialize database
    init_db()
    
    cache_service = get_cache_service()
    
    total_seeded = 0
    for language, qa_pairs in SEED_RESPONSES.items():
        logger.info(f"Seeding {len(qa_pairs)} responses for language: {language}")
        
        for qa in qa_pairs:
            question = qa["question"]
            response = qa["response"]
            
            try:
                # Cache the response
                success = cache_service.cache_response(
                    query=question,
                    response=response,
                    language=language,
                    conversation_id=None  # No associated conversation
                )
                
                if success:
                    total_seeded += 1
                    logger.info(f"  ✓ Cached: {question[:50]}...")
                else:
                    logger.warning(f"  ✗ Failed to cache: {question[:50]}...")
                    
            except Exception as e:
                logger.error(f"  ✗ Error caching '{question[:50]}...': {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Cache seeding complete! Total responses cached: {total_seeded}")
    logger.info(f"{'='*50}")
    
    # Verify by checking database
    try:
        with get_db_context() as db:
            from app.database.models import ResponseCache
            count = db.query(ResponseCache).count()
            logger.info(f"Total entries in response_cache table: {count}")
    except Exception as e:
        logger.error(f"Could not verify cache count: {e}")
    
    return total_seeded


if __name__ == "__main__":
    seeded = seed_cache()
    print(f"\nSeeded {seeded} responses into the cache for offline mode.")
    print("ASHA workers can now get answers even without internet!")
