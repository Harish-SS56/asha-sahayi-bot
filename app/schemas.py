"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field


# ============== Patient Schemas ==============

class PatientBase(BaseModel):
    """Base patient schema."""
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    village: Optional[str] = None


class PatientCreate(PatientBase):
    """Schema for creating a patient."""
    blood_group: Optional[str] = None
    known_conditions: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)


class PatientResponse(PatientBase):
    """Schema for patient response."""
    id: int
    known_conditions: List[str] = []
    is_pregnant: bool = False
    is_child: bool = False
    created_at: datetime
    
    class Config:
        from_attributes = True


class PatientDetailResponse(PatientResponse):
    """Schema for detailed patient response with visits."""
    recent_visits: List[dict] = []
    total_visits: int = 0


# ============== Visit Schemas ==============

class VitalsSchema(BaseModel):
    """Schema for vital signs."""
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    pulse_rate: Optional[int] = None
    temperature: Optional[float] = None
    weight: Optional[float] = None
    blood_sugar: Optional[float] = None
    oxygen_saturation: Optional[int] = None


class VisitCreate(BaseModel):
    """Schema for creating a visit."""
    patient_id: int
    visit_type: str = "home_visit"
    symptoms: List[str] = Field(default_factory=list)
    observations: Optional[str] = None
    vitals: Optional[VitalsSchema] = None
    medications_given: List[str] = Field(default_factory=list)
    referral_needed: bool = False
    follow_up_required: bool = False
    follow_up_days: Optional[int] = None


class VisitResponse(BaseModel):
    """Schema for visit response."""
    id: int
    patient_id: int
    visit_type: str
    visit_date: datetime
    symptoms: List[str]
    blood_pressure_systolic: Optional[int]
    blood_pressure_diastolic: Optional[int]
    temperature: Optional[float]
    referral_needed: bool
    follow_up_required: bool
    
    class Config:
        from_attributes = True


# ============== Worker Schemas ==============

class WorkerStatsResponse(BaseModel):
    """Schema for worker statistics."""
    total_patients: int
    total_visits: int
    visits_this_week: int
    pending_follow_ups: int


class WorkerProfileResponse(BaseModel):
    """Schema for worker profile."""
    id: int
    telegram_user_id: str
    name: Optional[str]
    preferred_language: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============== RAG Schemas ==============

class RAGQueryRequest(BaseModel):
    """Schema for RAG query request."""
    query: str = Field(..., min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)
    filter_type: Optional[str] = None


class RAGChunkResponse(BaseModel):
    """Schema for a single RAG chunk."""
    content: str
    source: str
    page: Any
    document_type: str
    relevance_score: float


class RAGQueryResponse(BaseModel):
    """Schema for RAG query response."""
    query: str
    results: List[RAGChunkResponse]


class RAGStatusResponse(BaseModel):
    """Schema for RAG index status."""
    indexed: bool
    document_count: int
    persist_directory: str
    docs_directory: str


# ============== Message Schemas ==============

class ChatMessageRequest(BaseModel):
    """Schema for chat message request."""
    message: str = Field(..., min_length=1, max_length=2000)
    telegram_user_id: str
    language: Optional[str] = "en"


class ChatMessageResponse(BaseModel):
    """Schema for chat message response."""
    response: str
    extracted_data: Optional[dict] = None
    patient_id: Optional[int] = None
    visit_id: Optional[int] = None
    rag_sources: List[str] = []
    language: str


# ============== Health Check Schemas ==============

class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    status: str
    database: str
    rag_index: str
    version: str


# ============== Error Schemas ==============

class ErrorResponse(BaseModel):
    """Schema for error response."""
    detail: str
    type: Optional[str] = None
    code: Optional[str] = None
