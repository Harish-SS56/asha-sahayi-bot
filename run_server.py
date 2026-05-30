"""
Entry point to run the FastAPI backend server.
"""

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.config import get_settings

settings = get_settings()


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        log_level="info"
    )


if __name__ == "__main__":
    main()
