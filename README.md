# ASHA AI Assistant 🏥

A resilient, voice-enabled, and privacy-conscious Telegram bot designed to support **ASHA (Accredited Social Health Activist)** workers in rural and semi-urban environments.

[![Telegram](https://img.shields.io/badge/Telegram-@asha__ai__assistant__bot-blue?style=flat&logo=telegram)](https://t.me/asha_ai_assistant_bot)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=flat&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-teal?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Demo Video](https://img.shields.io/badge/Demo-Watch%20Video-red?style=flat&logo=google-drive)](https://drive.google.com/file/d/19k36W5oIJnWvEemtT5gqIgBNY9GFn-Tn/view?usp=sharing)

> 🎬 **[Watch Demo Video](https://drive.google.com/file/d/19k36W5oIJnWvEemtT5gqIgBNY9GFn-Tn/view?usp=sharing)** — 2-minute walkthrough showing voice input, multilingual support, patient logging, RAG, and emergency detection.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Multilingual Support](#multilingual-support)
- [RAG System](#rag-system)
- [Safety & Ethics](#safety--ethics)
- [Deployment](#deployment)
- [Screenshots](#screenshots)

---

## 🎯 Overview

ASHA AI Assistant is a production-grade Telegram bot that helps ASHA workers:
- Record patient visits using voice notes or text messages
- Extract structured health data automatically
- Get evidence-based medical guidance from official NHM guidelines
- Track patient histories and follow-ups
- Communicate in multiple Indian languages

**Bot URL:** https://t.me/asha_ai_assistant_bot

---

## ✨ Features

### 🎤 Voice Note Processing
- Accept voice messages in Hindi, Tamil, Malayalam, and English
- Transcribe using Deepgram's nova-2 model
- Automatic language detection

### 📊 Structured Data Extraction
- Parse conversational updates into structured JSON
- Extract patient names, vital signs, symptoms, and more
- Function calling for consistent extraction

**Example:**
```
Input: "Visited Radha today, her blood pressure was 130/80 and she complained about dizziness."

Extracted JSON:
{
  "patient_name": "Radha",
  "blood_pressure": "130/80",
  "symptoms": ["dizziness"],
  "visit_type": "home_visit"
}
```

### 🔍 RAG (Retrieval-Augmented Generation)
- Grounded responses using official healthcare documents
- NHM guidelines and ASHA handbooks
- ChromaDB vector storage with Azure OpenAI embeddings

### 👥 Patient Management
- Automatic patient profile creation
- Visit history tracking
- Follow-up reminders
- Recurring patient recognition

### 🌐 Multilingual Support
- **English** - Full support
- **Hindi (हिन्दी)** - Full support
- **Tamil (தமிழ்)** - Full support
- **Malayalam (മലയാളം)** - Full support

### 🚨 Safety Features
- Emergency symptom detection
- Medical disclaimers on all responses
- Referral recommendations when needed
- PII protection with secure storage

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ASHA AI Assistant                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Telegram   │────▶│   FastAPI    │────▶│   Azure      │    │
│  │     Bot      │     │   Backend    │     │   OpenAI     │    │
│  └──────────────┘     └──────────────┘     │   GPT-4o     │    │
│         │                    │             └──────────────┘    │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Deepgram   │     │    Neon      │     │   ChromaDB   │    │
│  │   STT API    │     │  PostgreSQL  │     │  Vector DB   │    │
│  │   (nova-2)   │     │              │     │              │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Voice/Text Message
         │
         ▼
    Telegram Bot
         │
         ├──[Voice]──▶ Deepgram STT ──▶ Transcription
         │
         ▼
    FastAPI Backend
         │
         ├──▶ Azure OpenAI (Language Detection)
         │
         ├──▶ Azure OpenAI (Structured Extraction)
         │
         ├──▶ Neon PostgreSQL (Store Patient/Visit Data)
         │
         ├──▶ ChromaDB RAG (Retrieve Medical Context)
         │
         ▼
    Azure OpenAI GPT-4o (Generate Response)
         │
         ▼
    Telegram Response to User
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **Bot Framework** | python-telegram-bot 21.0 |
| **Backend** | FastAPI + Uvicorn |
| **AI Engine** | Azure OpenAI GPT-4o |
| **Embeddings** | Azure OpenAI text-embedding-3-large |
| **Speech-to-Text** | Deepgram nova-2 |
| **Database** | Neon PostgreSQL (Serverless) |
| **Vector Store** | ChromaDB |
| **RAG Framework** | LangChain |
| **ORM** | SQLAlchemy |

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Telegram Bot Token
- Azure OpenAI API access
- Deepgram API key
- Neon PostgreSQL database

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/asha-ai-assistant.git
cd asha-ai-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. **Initialize database**
```bash
python init_db.py
```

6. **Index healthcare documents**
```bash
python index_documents.py
```

7. **Run the bot**
```bash
python run_bot.py
```

---

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o

# Azure OpenAI Embeddings
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large-2
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Deepgram Speech-to-Text
DEEPGRAM_API_KEY=your_deepgram_api_key

# Database (Neon PostgreSQL)
DATABASE_URL=postgresql://user:password@host/database?sslmode=require

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./docs
```

---

## 📱 Usage

### Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Start the bot and select language |
| `/help` | View help guide |
| `/language` | Change language preference |
| `/patients` | View your patient list |
| `/stats` | View your statistics |
| `/followups` | View pending follow-ups |

### Recording Patient Visits

**Voice:** Send a voice note describing the visit in any supported language.

**Text:** Type the update directly:
```
Visited Radha today, her BP was 130/80, temperature 99°F. 
She complained of headache and dizziness. Gave paracetamol.
Need follow-up in 3 days.
```

### Asking Medical Questions

Simply type your question:
```
What are the danger signs during pregnancy?
```

The bot will respond with information from official NHM guidelines.

---

## 📡 API Documentation

The FastAPI backend provides REST endpoints:

### Health Check
```http
GET /api/v1/health
```

### Get Patient List
```http
GET /api/v1/patients/{worker_telegram_id}
```

### Get Worker Statistics
```http
GET /api/v1/stats/{worker_telegram_id}
```

### Query RAG
```http
POST /api/v1/rag/query
Content-Type: application/json

{
  "query": "symptoms of hypertension",
  "k": 4
}
```

### Interactive Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🌍 Multilingual Support

### Language Detection
The system automatically detects the input language and responds accordingly.

### Translation Logic
1. Voice notes are transcribed in the detected language
2. Structured data is extracted and normalized to English for storage
3. Responses are generated in the user's preferred language
4. Medical terms are preserved with local translations

### Supported Languages
- **English (en)** - Primary development language
- **Hindi (hi)** - हिन्दी with Devanagari script support
- **Tamil (ta)** - தமிழ் with Tamil script support  
- **Malayalam (ml)** - മലയാളം with Malayalam script support

---

## 📚 RAG System

### Document Sources

The RAG system indexes these official healthcare documents:

| Document | Content |
|----------|---------|
| `guidelines-on-asha.pdf` | ASHA program guidelines |
| `Handbook for ASHA on Home Based Care for Young Child-English.pdf` | Home-based care protocols |
| `Hypertension_full.pdf` | Hypertension management |
| `imnci_chart_booklet.pdf` | Child health (IMNCI) |
| `sba_guidelines_for_skilled_attendance_at_birth2.pdf` | Maternal health / skilled birth attendance |
| `NTEP.pdf` | National Tuberculosis Elimination Programme |
| `Operational_Guidelines_for_Introduction_of_Rotavac_in_UIP.pdf` | Rotavirus vaccine guidelines |
| `Revised_Home_Based_New_Born_Care_Operational_Guidelines_2014.pdf` | Newborn home-based care |

### How RAG Works

1. **Document Indexing**
   - PDFs are loaded and parsed
   - Text is split into chunks (1000 chars, 200 overlap)
   - Chunks are embedded using Azure OpenAI
   - Embeddings stored in ChromaDB

2. **Query Processing**
   - User query is embedded
   - Similar chunks retrieved from ChromaDB
   - Retrieved context sent to GPT-4o
   - Response grounded in official guidelines

### Indexing Documents

```bash
python index_documents.py
```

---

## �️ Ethical AI & Data Privacy Statement

This section addresses the three pillars required for UNESCO-affiliated AI projects.

---

### I. Medical Safety — Handling AI Hallucinations

AI language models can occasionally generate confident but incorrect information ("hallucinations"). This system uses a **multi-layer defence strategy** to protect ASHA workers and their patients:

#### Layer 1: RAG Grounding (Retrieval-Augmented Generation)
Every AI response is grounded in **official NHM/WHO documents** retrieved from a verified vector database. The model is instructed to answer only from retrieved context, not from general knowledge.

> *"Only answer based on the provided NHM guidelines context. If the context does not contain enough information to answer, say so clearly."*

**Sources used:**
- NHM ASHA Programme Guidelines
- NHM Hypertension Management Guidelines
- IMNCI (Integrated Management of Newborn and Childhood Illness) Chart Booklet
- SBA Guidelines for Skilled Birth Attendance
- National Tuberculosis Elimination Programme (NTEP) Guidelines
- Revised Home-Based Newborn Care Guidelines

#### Layer 2: Mandatory Disclaimer on Every Response
Every single AI response — without exception — appends:
> ⚠️ *DISCLAIMER: This assistant supports ASHA workers and should not replace licensed medical professionals. Always consult a doctor for medical decisions.*

This is enforced in code (`clean_ai_response()` removes any AI-generated disclaimer before our standardised one is appended, ensuring exactly one disclaimer always appears).

#### Layer 3: Assistant Identity Framing
The system prompt explicitly frames the bot as a **support tool, not a doctor**:
> *"You are ASHA Sahayi, an AI assistant that helps ASHA (Accredited Social Health Activist) workers. You are NOT a doctor. Always recommend consulting a healthcare professional for diagnosis and treatment."*

#### Layer 4: Emergency Override
When any critical symptom is detected (chest pain, convulsions, heavy bleeding, unconsciousness, etc.), the bot **bypasses the AI entirely** and immediately displays a hard-coded emergency message directing the user to call **108** and go to the nearest health facility.

#### Layer 5: No Diagnosis Policy
The bot is instructed to never provide:
- Definitive diagnoses
- Prescription advice
- Dosage instructions beyond basic ORS/Paracetamol first aid

---

### II. PII Protection — Personally Identifiable Information Strategy

#### What PII Is Collected
The bot collects the minimum data necessary for ASHA workers to track patient care:

| Data Type | Storage | Purpose |
|-----------|---------|---------|
| Patient name | PostgreSQL (encrypted at rest via Neon) | Patient identification |
| Age, gender | PostgreSQL | Clinical context |
| Village/address | PostgreSQL | Geographic tracking |
| Phone number | PostgreSQL | Emergency contact |
| Health vitals (BP, temp) | PostgreSQL | Medical record |
| Pregnancy status | PostgreSQL | Antenatal care tracking |
| Voice transcriptions | Not stored permanently | Used only for extraction, then discarded |

#### What PII Is NOT Stored
- **Application logs** contain only anonymous worker IDs and event types — no patient names, health data, or personal details
- **Credentials and API keys** are stored exclusively in environment variables (`.env`), never in source code
- **Conversation AI responses** are stored only for quality improvement and are not shared with third parties

#### Access Control
- Patient data is **scoped per ASHA worker** — a worker can only access patients they registered
- No cross-worker data access is possible through the bot interface
- Database access requires authenticated PostgreSQL credentials

#### Data Storage Security
- Hosted on **Neon PostgreSQL** (serverless, SOC 2 compliant)
- All connections use `sslmode=require`
- No raw SQL string construction — all queries use **SQLAlchemy ORM** with parameterised queries (prevents SQL injection)

#### Data Minimisation
- Names in non-Latin scripts (Tamil, Hindi, Malayalam) are stored as-is without transliteration
- The system validates that extracted patient names are real names (rejects values like "null", "hi", single characters) — *this is actively being improved*

---

### III. Consent — Ensuring Workers Have the Right to Log Data

#### Worker Registration Consent
When an ASHA worker sends `/start`, they are shown an onboarding message that explicitly states:
> *"This bot helps you record patient visits and health information. By using this bot, you confirm you are an authorised ASHA worker and have informed your patients that their health data will be recorded digitally for care management."*

Workers must actively choose a language and proceed — this constitutes **informed engagement consent**.

#### Patient Data Consent Responsibility
Under India's **Digital Personal Data Protection Act 2023** and NHM guidelines:
- ASHA workers, as registered health workers, are authorised to maintain patient health records
- The ASHA programme itself is a government-mandated health tracking programme
- Workers are trained to inform patients about health record keeping as part of their programme obligations

#### Data Purpose Limitation
Data collected is used **only** for:
1. Tracking patient health progress by the assigned ASHA worker
2. Generating follow-up reminders for the same worker
3. Improving response quality via anonymised feedback signals

Data is **not** used for:
- Training AI models
- Sharing with third parties
- Advertising or commercial purposes
- Research without explicit additional consent

#### Right to Deletion
Workers can contact the system administrator to request deletion of their data and all associated patient records. (A self-service `/delete_my_data` command is planned for a future release.)

---

### Emergency Handling

The bot monitors for these emergency symptoms across all languages:
- Severe chest pain / difficulty breathing
- High fever (>103°F / >39.4°C)
- Heavy bleeding (obstetric or otherwise)
- Loss of consciousness / convulsions / seizures
- Stroke symptoms (facial drooping, arm weakness, speech difficulty)
- Severe preeclampsia signs (BP ≥140/90 + headache + swelling in pregnancy)

When detected, the bot immediately:
1. Displays a prominent emergency alert
2. Instructs the worker to call **108** (National Ambulance Service)
3. Provides basic first-aid steps while help is on the way
4. Recommends the nearest PHC/CHC/District Hospital

---

## 🚀 Deployment

### Local Development
```bash
# Run bot in polling mode
python run_bot.py

# Run FastAPI server separately
python run_server.py
```

### Production (Railway/Render)

1. **Set environment variables** in the platform dashboard

2. **Start command:**
```bash
python run_bot.py
```

3. **Configure webhook** (optional):
```python
webhook_url = "https://your-app.railway.app/webhook/telegram"
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "run_bot.py"]
```

---

## 📸 Screenshots

### Bot Start Screen
```
🙏 Namaste, [User]!

Welcome to ASHA AI Assistant - your healthcare support companion.

I can help you with:
• 📝 Recording patient visits (voice or text)
• 📋 Tracking patient health records
• 🏥 Providing medical guidance from official NHM guidelines
• 🗣️ Multilingual support (Hindi, Tamil, Malayalam, English)
```

### Data Extraction
```
📋 Extracted Data:
👤 Patient: Radha
   Age: 45
🩺 BP: 130/80 mmHg
🌡️ Temp: 37.2°C
🤒 Symptoms: headache, dizziness
📍 Visit: Home Visit

✅ Visit recorded (ID: 42)

Based on the symptoms and vital signs, here are some recommendations...
```

### Statistics Dashboard
```
📊 Your Statistics

👥 Total Patients: 24
📝 Total Visits: 156
📅 Visits This Week: 12
⏰ Pending Follow-ups: 3
```

---

## 📄 License

This project is developed for educational and humanitarian purposes to support ASHA workers in India.

---

## 🙏 Acknowledgments

- **National Health Mission (NHM)** for healthcare guidelines
- **ASHA Program** for the inspiration
- **UNESCO** for promoting ethical AI in healthcare

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

*Built with ❤️ for ASHA workers serving communities across India*
