"""
Configuration management for ASHA AI Assistant.
Loads environment variables and provides application settings.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Telegram Bot Configuration
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(default="2024-12-01-preview", env="AZURE_OPENAI_API_VERSION")
    azure_openai_chat_deployment: str = Field(default="gpt-4o", env="AZURE_OPENAI_CHAT_DEPLOYMENT")
    
    # Azure OpenAI Embeddings Configuration
    azure_openai_embedding_api_version: str = Field(default="2024-02-01", env="AZURE_OPENAI_EMBEDDING_API_VERSION")
    azure_openai_embedding_deployment: str = Field(default="text-embedding-3-large-2", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    azure_openai_embedding_model: str = Field(default="text-embedding-3-large", env="AZURE_OPENAI_EMBEDDING_MODEL")
    
    # Deepgram Speech-to-Text Configuration
    deepgram_api_key: str = Field(..., env="DEEPGRAM_API_KEY")
    
    # ElevenLabs Text-to-Speech Configuration
    elevenlabs_api_key: str = Field(default="", env="ELEVENLABS_API_KEY")
    tts_enabled: bool = Field(default=True, env="TTS_ENABLED")
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Webhook Configuration (for production)
    webhook_url: str = Field(default="", env="WEBHOOK_URL")  # e.g., https://yourdomain.com/webhook
    webhook_secret: str = Field(default="", env="WEBHOOK_SECRET")
    webhook_port: int = Field(default=8443, env="WEBHOOK_PORT")
    use_webhook: bool = Field(default=False, env="USE_WEBHOOK")
    
    # Application Settings
    app_env: str = Field(default="development", env="APP_ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    chroma_persist_dir: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    docs_dir: str = Field(default="./docs", env="DOCS_DIR")
    
    # Supported Languages
    supported_languages: list[str] = ["en", "hi", "ta", "ml"]
    default_language: str = "en"
    
    # Medical Disclaimer
    medical_disclaimer: str = (
        "⚠️ DISCLAIMER: This assistant supports ASHA workers and should not replace "
        "licensed medical professionals. Always consult a doctor for medical decisions."
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Language mappings for Deepgram
LANGUAGE_CODES = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "ml": "ml",
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "malayalam": "ml"
}

# Language display names
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "ta": "தமிழ் (Tamil)",
    "ml": "മലയാളം (Malayalam)"
}
