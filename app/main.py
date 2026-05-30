"""
FastAPI Application Entry Point for ASHA AI Assistant.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.routes import router as api_router
from app.database.connection import init_db
from app.services.rag_service import initialize_rag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting ASHA AI Assistant...")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Initialize RAG service
    try:
        await initialize_rag()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.warning(f"RAG initialization warning: {e}")
    
    logger.info("ASHA AI Assistant started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ASHA AI Assistant...")


# Create FastAPI application
app = FastAPI(
    title="ASHA AI Assistant API",
    description="Voice-enabled, privacy-conscious Telegram bot API for ASHA workers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again later.",
            "type": type(exc).__name__
        }
    )


# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["API"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ASHA AI Assistant API",
        "version": "1.0.0",
        "description": "Voice-enabled healthcare support for ASHA workers",
        "endpoints": {
            "api": "/api/v1",
            "docs": "/docs",
            "health": "/api/v1/health"
        },
        "disclaimer": settings.medical_disclaimer
    }


# Telegram webhook endpoint (will be configured by bot)
@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    """
    Telegram webhook endpoint.
    This is a placeholder - the actual handler is set up by the Telegram bot.
    """
    return {"status": "webhook endpoint ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development"
    )
