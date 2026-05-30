"""
Database CRUD operations for ASHA AI Assistant.
"""

from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_

from app.database.models import Patient, Visit, ConversationLog, ASHAWorker, VisitType
import logging

logger = logging.getLogger(__name__)


# ============== ASHA Worker Operations ==============

def get_or_create_asha_worker(
    db: Session,
    telegram_user_id: str,
    telegram_username: Optional[str] = None,
    name: Optional[str] = None,
    preferred_language: str = "en"
) -> ASHAWorker:
    """Get existing ASHA worker or create new one."""
    worker = db.query(ASHAWorker).filter(
        ASHAWorker.telegram_user_id == telegram_user_id
    ).first()
    
    if not worker:
        worker = ASHAWorker(
            telegram_user_id=telegram_user_id,
            telegram_username=telegram_username,
            name=name,
            preferred_language=preferred_language
        )
        db.add(worker)
        db.flush()  # Flush to get ID, context manager handles commit
        logger.info(f"Created new ASHA worker: {telegram_user_id}")
    else:
        # Update username if changed
        if telegram_username and worker.telegram_username != telegram_username:
            worker.telegram_username = telegram_username
            db.flush()
    
    return worker


def update_worker_language(db: Session, worker_id: int, language: str) -> ASHAWorker:
    """Update ASHA worker's preferred language."""
    worker = db.query(ASHAWorker).filter(ASHAWorker.id == worker_id).first()
    if worker:
        worker.preferred_language = language
        db.flush()  # Flush changes, context manager handles commit
    return worker


# ============== Patient Operations ==============

def create_patient(
    db: Session,
    asha_worker_id: int,
    name: str,
    **kwargs
) -> Patient:
    """Create a new patient record."""
    logger.info(f"Creating patient: name='{name}', worker_id={asha_worker_id}, kwargs={kwargs}")
    patient = Patient(
        asha_worker_id=asha_worker_id,
        name=name,
        **kwargs
    )
    db.add(patient)
    db.flush()  # Flush to get ID without committing (context manager handles commit)
    logger.info(f"Created patient ID={patient.id}: {name} for worker {asha_worker_id}")
    return patient


def find_patient_by_name(
    db: Session,
    asha_worker_id: int,
    name: str,
    fuzzy_match: bool = True
) -> Optional[Patient]:
    """
    Find patient by name for a specific ASHA worker.
    Uses fuzzy matching to handle spelling variations.
    """
    logger.info(f"Finding patient: name='{name}', worker_id={asha_worker_id}, fuzzy={fuzzy_match}")
    # Exact match first
    patient = db.query(Patient).filter(
        Patient.asha_worker_id == asha_worker_id,
        func.lower(Patient.name) == func.lower(name),
        Patient.is_active == True
    ).first()
    
    if patient:
        logger.info(f"Found patient by exact match: ID={patient.id}, name='{patient.name}'")
        return patient
    
    if fuzzy_match:
        # Try partial match
        patient = db.query(Patient).filter(
            Patient.asha_worker_id == asha_worker_id,
            or_(
                func.lower(Patient.name).contains(name.lower()),
                func.lower(name).contains(func.lower(Patient.name))
            ),
            Patient.is_active == True
        ).first()
        
        if patient:
            logger.info(f"Found patient by fuzzy match: ID={patient.id}, name='{patient.name}'")
    
    if not patient:
        logger.info(f"No patient found for name='{name}'")
    
    return patient


def get_patient_by_id(db: Session, patient_id: int) -> Optional[Patient]:
    """Get patient by ID."""
    return db.query(Patient).filter(Patient.id == patient_id).first()


def get_patients_for_worker(
    db: Session,
    asha_worker_id: int,
    limit: int = 50
) -> List[Patient]:
    """Get all patients for an ASHA worker."""
    patients = db.query(Patient).options(
        joinedload(Patient.visits)  # Eagerly load visits to avoid DetachedInstanceError
    ).filter(
        Patient.asha_worker_id == asha_worker_id,
        Patient.is_active == True
    ).order_by(Patient.updated_at.desc()).limit(limit).all()
    
    logger.info(f"Retrieved {len(patients)} patients for worker {asha_worker_id}")
    return patients


def update_patient(
    db: Session,
    patient_id: int,
    **kwargs
) -> Optional[Patient]:
    """Update patient information."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient:
        for key, value in kwargs.items():
            if hasattr(patient, key) and value is not None:
                setattr(patient, key, value)
        patient.updated_at = datetime.utcnow()
        db.flush()  # Flush changes, context manager handles commit
    return patient


def get_patient_history(
    db: Session,
    patient_id: int,
    limit: int = 10
) -> List[Visit]:
    """Get recent visit history for a patient."""
    return db.query(Visit).filter(
        Visit.patient_id == patient_id
    ).order_by(Visit.visit_date.desc()).limit(limit).all()


# ============== Visit Operations ==============

def create_visit(
    db: Session,
    patient_id: int,
    asha_worker_id: int,
    visit_type: VisitType = VisitType.HOME_VISIT,
    **kwargs
) -> Visit:
    """Create a new visit record."""
    logger.info(f"Creating visit: patient_id={patient_id}, worker_id={asha_worker_id}, type={visit_type}")
    visit = Visit(
        patient_id=patient_id,
        asha_worker_id=asha_worker_id,
        visit_type=visit_type,
        **kwargs
    )
    db.add(visit)
    db.flush()  # Flush to get ID without committing
    logger.info(f"Created visit ID={visit.id} for patient {patient_id}, type={visit_type.value}")
    return visit


def get_recent_visits(
    db: Session,
    asha_worker_id: int,
    days: int = 7,
    limit: int = 20
) -> List[Visit]:
    """Get recent visits for an ASHA worker."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    return db.query(Visit).filter(
        Visit.asha_worker_id == asha_worker_id,
        Visit.visit_date >= cutoff
    ).order_by(Visit.visit_date.desc()).limit(limit).all()


def get_follow_ups_due(
    db: Session,
    asha_worker_id: int,
    days_ahead: int = 3
) -> List[Visit]:
    """Get visits that need follow-up in the next few days."""
    now = datetime.utcnow()
    future = now + timedelta(days=days_ahead)
    
    return db.query(Visit).options(
        joinedload(Visit.patient)  # Eagerly load patient to avoid DetachedInstanceError
    ).filter(
        Visit.asha_worker_id == asha_worker_id,
        Visit.follow_up_required == True,
        Visit.follow_up_date >= now,
        Visit.follow_up_date <= future
    ).order_by(Visit.follow_up_date).all()


# ============== Conversation Log Operations ==============

def log_conversation(
    db: Session,
    asha_worker_id: int,
    message_type: str,
    input_text: Optional[str] = None,
    input_language: Optional[str] = None,
    transcription: Optional[str] = None,
    ai_response: Optional[str] = None,
    **kwargs
) -> ConversationLog:
    """Log a conversation interaction."""
    log = ConversationLog(
        asha_worker_id=asha_worker_id,
        message_type=message_type,
        input_text=input_text,
        input_language=input_language,
        transcription=transcription,
        ai_response=ai_response,
        **kwargs
    )
    db.add(log)
    db.flush()  # Flush to get ID, context manager handles commit
    return log


def update_conversation_log(
    db: Session,
    conversation_id: int,
    ai_response: Optional[str] = None,
    response_language: Optional[str] = None,
    rag_chunks_used: Optional[List[str]] = None,
    tokens_used: Optional[int] = None
) -> Optional[ConversationLog]:
    """Update a conversation log with AI response data."""
    log = db.query(ConversationLog).filter(ConversationLog.id == conversation_id).first()
    if log:
        if ai_response is not None:
            log.ai_response = ai_response
        if response_language is not None:
            log.response_language = response_language
        if rag_chunks_used is not None:
            log.rag_chunks_used = rag_chunks_used
        if tokens_used is not None:
            log.tokens_used = tokens_used
        db.flush()
        logger.info(f"Updated conversation log ID={conversation_id}")
    return log


def get_conversation_history(
    db: Session,
    asha_worker_id: int,
    limit: int = 10
) -> List[ConversationLog]:
    """Get recent conversation history for context."""
    return db.query(ConversationLog).filter(
        ConversationLog.asha_worker_id == asha_worker_id,
        ConversationLog.had_error == False
    ).order_by(ConversationLog.created_at.desc()).limit(limit).all()


# ============== Analytics ==============

def get_all_active_workers(db: Session) -> List[ASHAWorker]:
    """Get all ASHA workers who have used the bot (for scheduled jobs)."""
    return db.query(ASHAWorker).filter(ASHAWorker.is_active == True).all()


def get_worker_stats(db: Session, asha_worker_id: int) -> dict:
    """Get statistics for an ASHA worker."""
    total_patients = db.query(func.count(Patient.id)).filter(
        Patient.asha_worker_id == asha_worker_id,
        Patient.is_active == True
    ).scalar()
    
    total_visits = db.query(func.count(Visit.id)).filter(
        Visit.asha_worker_id == asha_worker_id
    ).scalar()
    
    visits_this_week = db.query(func.count(Visit.id)).filter(
        Visit.asha_worker_id == asha_worker_id,
        Visit.visit_date >= datetime.utcnow() - timedelta(days=7)
    ).scalar()
    
    pending_follow_ups = db.query(func.count(Visit.id)).filter(
        Visit.asha_worker_id == asha_worker_id,
        Visit.follow_up_required == True,
        Visit.follow_up_date >= datetime.utcnow()
    ).scalar()
    
    return {
        "total_patients": total_patients or 0,
        "total_visits": total_visits or 0,
        "visits_this_week": visits_this_week or 0,
        "pending_follow_ups": pending_follow_ups or 0
    }


# ============== Feedback Operations ==============

def create_feedback(
    db: Session,
    conversation_log_id: int,
    asha_worker_id: int,
    is_helpful: bool,
    original_question: Optional[str] = None,
    original_response: Optional[str] = None,
    language: Optional[str] = None,
    feedback_text: Optional[str] = None
) -> "Feedback":
    """Create a new feedback record."""
    from app.database.models import Feedback
    
    feedback = Feedback(
        conversation_log_id=conversation_log_id,
        asha_worker_id=asha_worker_id,
        is_helpful=is_helpful,
        original_question=original_question,
        original_response=original_response,
        language=language,
        feedback_text=feedback_text
    )
    db.add(feedback)
    db.flush()
    logger.info(f"Created feedback: conv_id={conversation_log_id}, helpful={is_helpful}")
    return feedback


def get_recent_negative_feedback(
    db: Session,
    asha_worker_id: int,
    limit: int = 5
) -> List["Feedback"]:
    """Get recent negative feedback for improving responses."""
    from app.database.models import Feedback
    
    return db.query(Feedback).filter(
        Feedback.asha_worker_id == asha_worker_id,
        Feedback.is_helpful == False
    ).order_by(Feedback.created_at.desc()).limit(limit).all()


def get_feedback_stats(db: Session, asha_worker_id: Optional[int] = None) -> dict:
    """Get feedback statistics."""
    from app.database.models import Feedback
    
    query = db.query(Feedback)
    if asha_worker_id:
        query = query.filter(Feedback.asha_worker_id == asha_worker_id)
    
    total = query.count()
    positive = query.filter(Feedback.is_helpful == True).count()
    negative = query.filter(Feedback.is_helpful == False).count()
    
    return {
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "satisfaction_rate": (positive / total * 100) if total > 0 else 0
    }


# ============== Response Cache Operations ==============

def get_cached_response(
    db: Session,
    query_hash: str
) -> Optional["ResponseCache"]:
    """Get cached response by query hash."""
    from app.database.models import ResponseCache
    
    cache = db.query(ResponseCache).filter(
        ResponseCache.query_hash == query_hash,
        ResponseCache.is_active == True,
        or_(
            ResponseCache.expires_at == None,
            ResponseCache.expires_at > datetime.utcnow()
        )
    ).first()
    
    if cache:
        # Increment hit count
        cache.hit_count += 1
        db.flush()
        logger.info(f"Cache hit for hash={query_hash[:16]}..., hits={cache.hit_count}")
    
    return cache


def create_or_update_cache(
    db: Session,
    query_hash: str,
    query_normalized: str,
    response_text: str,
    language: str = "en",
    query_keywords: Optional[List[str]] = None,
    source_conversation_id: Optional[int] = None,
    is_curated: bool = False
) -> "ResponseCache":
    """Create or update a cached response."""
    from app.database.models import ResponseCache
    
    existing = db.query(ResponseCache).filter(
        ResponseCache.query_hash == query_hash
    ).first()
    
    if existing:
        # Update existing cache
        existing.response_text = response_text
        existing.query_keywords = query_keywords or existing.query_keywords
        existing.updated_at = datetime.utcnow()
        db.flush()
        logger.info(f"Updated cache for hash={query_hash[:16]}...")
        return existing
    
    # Create new cache entry
    cache = ResponseCache(
        query_hash=query_hash,
        query_normalized=query_normalized,
        response_text=response_text,
        language=language,
        query_keywords=query_keywords or [],
        source_conversation_id=source_conversation_id,
        is_curated=is_curated
    )
    db.add(cache)
    db.flush()
    logger.info(f"Created cache for hash={query_hash[:16]}...")
    return cache


def update_cache_feedback(
    db: Session,
    query_hash: str,
    is_positive: bool
) -> Optional["ResponseCache"]:
    """Update cache quality metrics based on feedback."""
    from app.database.models import ResponseCache
    
    cache = db.query(ResponseCache).filter(
        ResponseCache.query_hash == query_hash
    ).first()
    
    if cache:
        if is_positive:
            cache.positive_feedback_count += 1
        else:
            cache.negative_feedback_count += 1
        
        # Recalculate confidence score
        total = cache.positive_feedback_count + cache.negative_feedback_count
        if total > 0:
            cache.confidence_score = cache.positive_feedback_count / total
        
        # Deactivate low-quality cached responses
        if cache.confidence_score < 0.3 and total >= 5:
            cache.is_active = False
            logger.info(f"Deactivated low-quality cache: hash={query_hash[:16]}...")
        
        db.flush()
    
    return cache


def search_similar_cached_responses(
    db: Session,
    keywords: List[str],
    language: str = "en",
    limit: int = 3
) -> List["ResponseCache"]:
    """Search for similar cached responses by keywords.
    
    IMPROVED: Requires at least 2 keyword matches or 40% of keywords to match,
    not just any single keyword. This prevents returning unrelated responses.
    """
    from app.database.models import ResponseCache
    
    if not keywords or len(keywords) < 2:
        return []  # Need at least 2 keywords for reliable matching
    
    # Build OR conditions for keyword matching in JSON array
    results = db.query(ResponseCache).filter(
        ResponseCache.is_active == True,
        ResponseCache.language == language,
        ResponseCache.confidence_score >= 0.7,  # Increased from 0.5 to 0.7 for stricter matching
        or_(
            ResponseCache.expires_at == None,
            ResponseCache.expires_at > datetime.utcnow()
        )
    ).order_by(
        ResponseCache.confidence_score.desc(),
        ResponseCache.hit_count.desc()
    ).limit(limit * 5).all()  # Get more to filter by keywords
    
    # Filter by keywords - require multiple keyword matches
    min_matches_required = max(2, int(len(keywords) * 0.4))  # At least 2 or 40% of keywords
    
    matched = []
    for r in results:
        query_lower = (r.query_normalized or "").lower()
        # Count how many keywords match
        match_count = sum(1 for kw in keywords if kw.lower() in query_lower)
        if match_count >= min_matches_required:
            matched.append((r, match_count))
    
    # Sort by match count (most matches first) and return
    matched.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in matched[:limit]]
