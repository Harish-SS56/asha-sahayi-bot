"""
Database connection management for Neon PostgreSQL.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create SQLAlchemy engine with connection pooling disabled for serverless
engine = create_engine(
    settings.database_url,
    poolclass=NullPool,  # Recommended for serverless/Neon
    echo=settings.app_env == "development",
    connect_args={
        "sslmode": "require"
    } if "neon.tech" in settings.database_url else {}
)

# Create session factory
# expire_on_commit=False keeps objects usable after session closes (avoids DetachedInstanceError)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency for FastAPI to get database session.
    Yields a session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions outside of FastAPI.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    Creates all tables defined in models.
    """
    from app.database.models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully")

    # Migrations: fix feedback table schema created by older code versions
    try:
        with engine.begin() as conn:
            # Drop FK constraint on conversation_log_id (it's a plain ref, not enforced)
            conn.execute(text(
                "ALTER TABLE feedback DROP CONSTRAINT IF EXISTS feedback_conversation_log_id_fkey"
            ))
            # Make conversation_log_id nullable
            conn.execute(text(
                "ALTER TABLE feedback ALTER COLUMN conversation_log_id DROP NOT NULL"
            ))
        logger.info("Feedback table migrations applied successfully")
    except Exception as e:
        logger.debug(f"Feedback migrations skipped (already applied): {e}")


async def check_db_connection() -> bool:
    """
    Check if database connection is healthy.
    """
    try:
        with get_db_context() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
