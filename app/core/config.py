
import os
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings managed through Pydantic-Settings.

    Loads values from environment variables and falls back to a .env file.
    """

    # Application Metadata
    APP_NAME: str = Field(
        default="Fluid AI Autonomous Agent",
        description="The name of the FastAPI application.",
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="The version of the FastAPI application.",
    )

    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Global log filter level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # LLM Configuration
    # We set default to None but validate that it is not empty in runtime or startup
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Gemini API key required for LLM integrations.",
    )
    DEFAULT_MODEL: str = Field(
        default="gemini-1.5-flash",
        description="The default Gemini LLM model identifier to use.",
    )

    # Output Storage
    OUTPUT_DIR: str = Field(
        default="outputs",
        description="Relative or absolute directory path where generated Word (.docx) files are saved.",
    )

    # Use SettingsConfigDict for Pydantic v2 configurations
    model_config = SettingsConfigDict(
        # Seek the .env file in the workspace root directory (two directories up from app/core)
        env_file=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            ".env",
        ),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validates that the log level is a standard level."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in allowed_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: '{v}'. Must be one of {allowed_levels}"
            )
        return upper_v

    @property
    def output_path(self) -> Path:
        """Helper to return Path object for the output directory and ensure it exists."""
        path = Path(self.OUTPUT_DIR)
        # Create output directory if it does not exist
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_gemini_configured(self) -> bool:
        """Returns True if the Gemini API key is configured with a non-empty string."""
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip())


# Create singleton instance of Settings to be imported throughout the project
settings = Settings()
