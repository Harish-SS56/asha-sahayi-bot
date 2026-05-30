"""
SQLAlchemy models for ASHA AI Assistant.
Defines database tables for patients, visits, conversations, and workers.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, 
    Boolean, JSON, Float, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()


class VisitType(enum.Enum):
    """Types of patient visits."""
    HOME_VISIT = "home_visit"
    CLINIC_VISIT = "clinic_visit"
    EMERGENCY = "emergency"
    FOLLOW_UP = "follow_up"
    ROUTINE_CHECKUP = "routine_checkup"
    IMMUNIZATION = "immunization"
    ANTENATAL = "antenatal"
    POSTNATAL = "postnatal"


class ASHAWorker(Base):
    """ASHA Worker profile linked to Telegram user."""
    __tablename__ = "asha_workers"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_user_id = Column(String(50), unique=True, index=True, nullable=False)
    telegram_username = Column(String(100), nullable=True)
    name = Column(String(200), nullable=True)
    phone = Column(String(20), nullable=True)
    village = Column(String(200), nullable=True)
    district = Column(String(200), nullable=True)
    state = Column(String(100), nullable=True)
    preferred_language = Column(String(10), default="en")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patients = relationship("Patient", back_populates="asha_worker")
    conversations = relationship("ConversationLog", back_populates="asha_worker")


class Patient(Base):
    """Patient profile with health information."""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    asha_worker_id = Column(Integer, ForeignKey("asha_workers.id"), nullable=False)
    
    # Basic Information
    name = Column(String(200), nullable=False, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    phone = Column(String(20), nullable=True)
    address = Column(Text, nullable=True)
    village = Column(String(200), nullable=True)
    
    # Health Information
    blood_group = Column(String(10), nullable=True)
    known_conditions = Column(JSON, default=list)  # List of known medical conditions
    allergies = Column(JSON, default=list)  # List of allergies
    medications = Column(JSON, default=list)  # Current medications
    
    # Pregnancy tracking (if applicable)
    is_pregnant = Column(Boolean, default=False)
    expected_delivery_date = Column(DateTime, nullable=True)
    last_menstrual_period = Column(DateTime, nullable=True)
    
    # Child tracking (if applicable)
    is_child = Column(Boolean, default=False)
    date_of_birth = Column(DateTime, nullable=True)
    immunization_status = Column(JSON, default=dict)
    
    # Metadata
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    asha_worker = relationship("ASHAWorker", back_populates="patients")
    visits = relationship("Visit", back_populates="patient", order_by="desc(Visit.visit_date)")


class Visit(Base):
    """Record of patient visits and health measurements."""
    __tablename__ = "visits"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    asha_worker_id = Column(Integer, ForeignKey("asha_workers.id"), nullable=False)
    
    # Visit Details
    visit_type = Column(SQLEnum(VisitType), default=VisitType.HOME_VISIT)
    visit_date = Column(DateTime, default=datetime.utcnow)
    location = Column(String(200), nullable=True)
    
    # Vital Signs
    blood_pressure_systolic = Column(Integer, nullable=True)
    blood_pressure_diastolic = Column(Integer, nullable=True)
    pulse_rate = Column(Integer, nullable=True)
    temperature = Column(Float, nullable=True)  # In Celsius
    weight = Column(Float, nullable=True)  # In kg
    height = Column(Float, nullable=True)  # In cm
    blood_sugar = Column(Float, nullable=True)
    oxygen_saturation = Column(Integer, nullable=True)
    
    # Symptoms and Observations
    symptoms = Column(JSON, default=list)  # List of reported symptoms
    observations = Column(Text, nullable=True)
    
    # Actions Taken
    advice_given = Column(Text, nullable=True)
    medications_given = Column(JSON, default=list)
    referral_needed = Column(Boolean, default=False)
    referral_facility = Column(String(200), nullable=True)
    
    # Follow-up
    follow_up_required = Column(Boolean, default=False)
    follow_up_date = Column(DateTime, nullable=True)
    
    # Raw data from conversation
    raw_transcript = Column(Text, nullable=True)
    extracted_data = Column(JSON, default=dict)  # Full structured extraction
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="visits")


class ConversationLog(Base):
    """Log of all conversations with the bot."""
    __tablename__ = "conversation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    asha_worker_id = Column(Integer, ForeignKey("asha_workers.id"), nullable=False)
    
    # Message Details
    telegram_message_id = Column(String(50), nullable=True)
    message_type = Column(String(20), nullable=False)  # text, voice, photo
    input_text = Column(Text, nullable=True)  # Original or transcribed text
    input_language = Column(String(10), nullable=True)
    
    # Voice Processing
    voice_file_id = Column(String(200), nullable=True)
    transcription = Column(Text, nullable=True)
    transcription_confidence = Column(Float, nullable=True)
    
    # AI Processing
    ai_response = Column(Text, nullable=True)
    response_language = Column(String(10), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    
    # Context
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    visit_id = Column(Integer, ForeignKey("visits.id"), nullable=True)
    rag_chunks_used = Column(JSON, default=list)  # References to RAG sources
    
    # Error Handling
    had_error = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Relationships
    asha_worker = relationship("ASHAWorker", back_populates="conversations")


class Feedback(Base):
    """User feedback on bot responses for quality improvement."""
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_log_id = Column(Integer, nullable=True)  # plain ref, no FK constraint
    asha_worker_id = Column(Integer, ForeignKey("asha_workers.id"), nullable=False)
    
    # Feedback data
    is_helpful = Column(Boolean, nullable=False)  # True = 👍, False = 👎
    feedback_text = Column(Text, nullable=True)  # Optional detailed feedback
    
    # Context for improvement
    original_question = Column(Text, nullable=True)
    original_response = Column(Text, nullable=True)
    language = Column(String(10), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    asha_worker = relationship("ASHAWorker")


class ResponseCache(Base):
    """Cache of common responses for offline/low-connectivity mode."""
    __tablename__ = "response_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Query matching
    query_hash = Column(String(64), unique=True, index=True, nullable=False)  # SHA256 hash of normalized query
    query_normalized = Column(Text, nullable=False)  # Normalized query text for fuzzy matching
    query_keywords = Column(JSON, default=list)  # Extracted keywords for matching
    language = Column(String(10), default="en")
    
    # Cached response
    response_text = Column(Text, nullable=False)
    
    # Quality metrics
    hit_count = Column(Integer, default=0)  # How many times this cache was used
    positive_feedback_count = Column(Integer, default=0)
    negative_feedback_count = Column(Integer, default=0)
    confidence_score = Column(Float, default=1.0)  # Quality score based on feedback
    
    # Source tracking
    is_curated = Column(Boolean, default=False)  # Manually curated vs auto-cached
    source_conversation_id = Column(Integer, ForeignKey("conversation_logs.id"), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # Optional expiry for time-sensitive info
    is_active = Column(Boolean, default=True)
