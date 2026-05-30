"""
ASHA AI Assistant Telegram Bot.
Main bot class and application setup.
"""

import logging
import datetime
from typing import Optional
from telegram import Update, Bot
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler
)

from app.config import get_settings
from app.bot.handlers import setup_handlers, send_daily_followup_reminders

logger = logging.getLogger(__name__)
settings = get_settings()


class ASHABot:
    """Main ASHA AI Assistant Telegram Bot class."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the ASHA bot.
        
        Args:
            token: Telegram bot token. If not provided, uses settings.
        """
        self.token = token or settings.telegram_bot_token
        self.application: Optional[Application] = None
        
    def build(self) -> Application:
        """Build and configure the Telegram application."""
        # Create application with builder
        self.application = (
            ApplicationBuilder()
            .token(self.token)
            .build()
        )
        
        # Setup all handlers
        setup_handlers(self.application)

        # Schedule daily follow-up reminder at 08:00 IST (02:30 UTC)
        job_queue = self.application.job_queue
        if job_queue:
            reminder_time = datetime.time(hour=2, minute=30, tzinfo=datetime.timezone.utc)
            job_queue.run_daily(
                send_daily_followup_reminders,
                time=reminder_time,
                name="daily_followup_reminder",
            )
            logger.info("Daily follow-up reminder scheduled at 08:00 IST (02:30 UTC)")
        
        logger.info("ASHA Bot built successfully")
        return self.application
    
    def run_polling(self):
        """Run the bot using polling (for development)."""
        if self.application is None:
            self.build()
        
        logger.info("Starting ASHA Bot in polling mode...")
        self.application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
    
    async def run_webhook(self, webhook_url: str, port: int = 8443):
        """
        Run the bot using webhooks (for production).
        
        Args:
            webhook_url: The webhook URL to set
            port: Port to listen on
        """
        if self.application is None:
            self.build()
        
        # Set webhook
        await self.application.bot.set_webhook(
            url=webhook_url,
            allowed_updates=Update.ALL_TYPES
        )
        
        logger.info(f"Starting ASHA Bot with webhook at {webhook_url}")
        
        # Start webhook server
        await self.application.run_webhook(
            listen="0.0.0.0",
            port=port,
            webhook_url=webhook_url
        )
    
    def get_application(self) -> Application:
        """Get the Telegram application instance."""
        if self.application is None:
            self.build()
        return self.application


def create_bot() -> ASHABot:
    """Factory function to create and configure the bot."""
    bot = ASHABot()
    bot.build()
    return bot


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler for the bot."""
    logger.error(f"Exception while handling an update: {context.error}")
    
    if isinstance(update, Update) and update.effective_message:
        error_message = (
            "Sorry, something went wrong. Please try again.\n"
            "यदि समस्या बनी रहती है, तो /help आज़माएं।"
        )
        try:
            await update.effective_message.reply_text(error_message)
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
