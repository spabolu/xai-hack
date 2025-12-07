"""Configuration management for the NBA commentary agent."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    # xAI Grok API Configuration
    XAI_API_KEY: Optional[str] = os.getenv("XAI_API_KEY")
    GROK_MODEL: str = os.getenv("GROK_MODEL")

    # Agent Configuration
    TEMPERATURE: float = float(os.getenv("TEMPERATURE"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS"))

    # Thread/Context Management
    THREAD_ID: str = os.getenv("THREAD_ID")

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        if not cls.XAI_API_KEY:
            raise ValueError(
                "XAI_API_KEY environment variable is required. "
                "Set it in your .env file or environment."
            )
