"""
Entry point to run the ASHA AI Assistant Telegram Bot.
Supports both polling (development) and webhook (production) modes.
"""

import logging
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.bot.bot import ASHABot
from app.database.connection import init_db
from app.services.rag_service import get_rag_service
from app.services.tts_service import get_tts_service
from app.config import get_settings

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
settings = get_settings()


async def run_webhook_mode():
    """Run bot in webhook mode for production."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    from telegram import Update
    
    bot = ASHABot()
    application = bot.build()
    
    # Create FastAPI app for webhook
    app = FastAPI(title="ASHA AI Assistant")
    
    @app.post("/webhook")
    async def telegram_webhook(request: Request):
        """Handle incoming Telegram updates via webhook."""
        try:
            data = await request.json()
            update = Update.de_json(data, application.bot)
            await application.process_update(update)
            return JSONResponse({"status": "ok"})
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "mode": "webhook"}
    
    @app.on_event("startup")
    async def on_startup():
        """Set webhook on startup."""
        await application.initialize()
        webhook_url = f"{settings.webhook_url}/webhook"
        await application.bot.set_webhook(
            url=webhook_url,
            secret_token=settings.webhook_secret if settings.webhook_secret else None,
            allowed_updates=["message", "callback_query", "edited_message"]
        )
        logger.info(f"Webhook set to: {webhook_url}")
    
    @app.on_event("shutdown")
    async def on_shutdown():
        """Cleanup on shutdown."""
        await application.shutdown()
    
    # Run the server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.webhook_port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """Main entry point for running the bot."""
    logger.info("Starting ASHA AI Assistant Bot...")
    
    # Initialize database
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Initialize RAG service (index documents if needed)
    try:
        logger.info("Initializing RAG service...")
        rag_service = get_rag_service()
        status = rag_service.get_index_status()
        
        if not status.get("indexed"):
            logger.info("Indexing healthcare documents...")
            count = rag_service.index_documents()
            logger.info(f"Indexed {count} document chunks")
        else:
            logger.info(f"RAG index ready with {status['document_count']} documents")
    except Exception as e:
        logger.warning(f"RAG initialization warning: {e}")
    
    # Initialize TTS service
    try:
        logger.info("Initializing TTS service...")
        tts_service = get_tts_service()
        if tts_service.is_enabled():
            logger.info("Google TTS (gTTS) service ready - voice messages enabled")
        else:
            logger.info("TTS service disabled")
    except Exception as e:
        logger.warning(f"TTS initialization warning: {e}")
    
    # Create and run bot
    try:
        if settings.use_webhook and settings.webhook_url:
            logger.info(f"Starting bot in WEBHOOK mode at {settings.webhook_url}")
            asyncio.run(run_webhook_mode())
        else:
            logger.info("Starting bot in POLLING mode (development)...")
            bot = ASHABot()
            bot.build()
            logger.info("Bot built successfully. Starting polling...")
            bot.run_polling()
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
