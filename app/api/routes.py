"""
FastAPI route definitions for ASHA AI Assistant.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List
from pydantic import BaseModel

from app.database.connection import get_db
from app.database import crud
from app.services.rag_service import get_rag_service
from app.config import get_settings

router = APIRouter()
settings = get_settings()


# ============== Request/Response Models ==============

class HealthCheckResponse(BaseModel):
    status: str
    database: str
    rag_index: str
    version: str


class PatientResponse(BaseModel):
    id: int
    name: str
    age: Optional[int]
    gender: Optional[str]
    village: Optional[str]
    known_conditions: List[str]
    is_pregnant: bool
    is_child: bool
    
    class Config:
        from_attributes = True


class WorkerStatsResponse(BaseModel):
    total_patients: int
    total_visits: int
    visits_this_week: int
    pending_follow_ups: int


class RAGQueryRequest(BaseModel):
    query: str
    k: int = 4


class RAGQueryResponse(BaseModel):
    query: str
    results: List[dict]


# ============== Health & Status Endpoints ==============

@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    """Check health status of all services."""
    try:
        # Check database
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check RAG index
    try:
        rag_service = get_rag_service()
        rag_status = rag_service.get_index_status()
        rag_status_str = f"indexed ({rag_status['document_count']} docs)" if rag_status['indexed'] else "not indexed"
    except Exception as e:
        rag_status_str = f"error: {str(e)}"
    
    return HealthCheckResponse(
        status="ok" if db_status == "healthy" else "degraded",
        database=db_status,
        rag_index=rag_status_str,
        version="1.0.0"
    )


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ASHA AI Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }


# ============== Patient Endpoints ==============

@router.get("/patients/{worker_telegram_id}", response_model=List[PatientResponse])
async def get_worker_patients(
    worker_telegram_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get all patients for an ASHA worker."""
    worker = crud.get_or_create_asha_worker(db, worker_telegram_id)
    patients = crud.get_patients_for_worker(db, worker.id, limit)
    return patients


@router.get("/patients/{worker_telegram_id}/{patient_id}")
async def get_patient_details(
    worker_telegram_id: str,
    patient_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed patient information including history."""
    worker = crud.get_or_create_asha_worker(db, worker_telegram_id)
    patient = crud.get_patient_by_id(db, patient_id)
    
    if not patient or patient.asha_worker_id != worker.id:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get visit history
    visits = crud.get_patient_history(db, patient_id, limit=10)
    
    return {
        "patient": patient,
        "recent_visits": visits,
        "total_visits": len(patient.visits)
    }


@router.get("/stats/{worker_telegram_id}", response_model=WorkerStatsResponse)
async def get_worker_stats(
    worker_telegram_id: str,
    db: Session = Depends(get_db)
):
    """Get statistics for an ASHA worker."""
    worker = crud.get_or_create_asha_worker(db, worker_telegram_id)
    stats = crud.get_worker_stats(db, worker.id)
    return WorkerStatsResponse(**stats)


# ============== RAG Endpoints ==============

@router.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """Query the RAG system for relevant medical information."""
    try:
        rag_service = get_rag_service()
        results = await rag_service.retrieve_relevant_context(
            query=request.query,
            k=request.k
        )
        return RAGQueryResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/index")
async def index_documents(force: bool = False):
    """Index or reindex healthcare documents."""
    try:
        rag_service = get_rag_service()
        count = rag_service.index_documents(force_reindex=force)
        return {"message": f"Indexed {count} document chunks", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/status")
async def get_rag_status():
    """Get RAG index status."""
    try:
        rag_service = get_rag_service()
        return rag_service.get_index_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Follow-up Endpoints ==============

@router.get("/follow-ups/{worker_telegram_id}")
async def get_pending_follow_ups(
    worker_telegram_id: str,
    days_ahead: int = 3,
    db: Session = Depends(get_db)
):
    """Get pending follow-ups for an ASHA worker."""
    worker = crud.get_or_create_asha_worker(db, worker_telegram_id)
    follow_ups = crud.get_follow_ups_due(db, worker.id, days_ahead)
    
    return {
        "worker_id": worker.id,
        "days_ahead": days_ahead,
        "pending_follow_ups": [
            {
                "visit_id": v.id,
                "patient_name": v.patient.name,
                "follow_up_date": v.follow_up_date,
                "visit_type": v.visit_type.value,
                "symptoms": v.symptoms
            }
            for v in follow_ups
        ]
    }
