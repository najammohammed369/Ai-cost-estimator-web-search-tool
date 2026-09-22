"""
Vessel Cost Estimator — Application Configuration

Loads all settings from environment variables / .env file.
Uses pydantic-settings for validation and type coercion.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # LLM Configuration
    # -------------------------------------------------------------------------
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model_name: str = "databricks-meta-llama-3-3-70b-instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # -------------------------------------------------------------------------
    # Web Search
    # -------------------------------------------------------------------------
    search_provider: str = "serpapi"
    serpapi_key: str = ""

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'cache' / 'vessel_cache.db'}"

    # -------------------------------------------------------------------------
    # Application Directories
    # -------------------------------------------------------------------------
    upload_dir: str = str(PROJECT_ROOT / "data" / "extracted")
    report_dir: str = str(PROJECT_ROOT / "data" / "reports")
    log_level: str = "INFO"

    # -------------------------------------------------------------------------
    # Cache TTL (seconds)
    # -------------------------------------------------------------------------
    search_cache_ttl: int = 604800       # 7 days
    webpage_cache_ttl: int = 2592000     # 30 days

    # -------------------------------------------------------------------------
    # Search Configuration
    # -------------------------------------------------------------------------
    max_search_queries: int = 15
    max_results_per_query: int = 10
    max_concurrent_requests: int = 5

    def ensure_directories(self) -> None:
        """Create required data directories if they don't exist."""
        for dir_path in [
            self.upload_dir,
            self.report_dir,
            str(PROJECT_ROOT / "data" / "cache"),
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
