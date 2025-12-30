"""
Configuration management using Pydantic Settings.
Supports environment variables and .env files.
"""

from functools import lru_cache
from typing import List, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PreprocessingSettings(BaseSettings):
    """Image preprocessing configuration."""
    
    model_config = SettingsConfigDict(env_prefix="PREPROCESSING_")
    
    enabled: bool = True
    deskew: bool = True
    denoise: bool = True
    binarize: bool = False
    enhance_contrast: bool = True
    auto_crop: bool = True
    target_dpi: int = 300
    max_dimension: int = 4096


class CacheSettings(BaseSettings):
    """Caching configuration."""
    
    model_config = SettingsConfigDict(env_prefix="CACHE_")
    
    enabled: bool = True
    ttl_seconds: int = 86400  # 24 hours
    max_size: int = 1000  # Max items in memory cache
    redis_url: str = "redis://localhost:6379/0"


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""
    
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")
    
    enabled: bool = True
    requests: int = 100
    window_seconds: int = 60


class RetrySettings(BaseSettings):
    """Retry policy configuration."""
    
    model_config = SettingsConfigDict(env_prefix="RETRY_")
    
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Google Cloud
    google_application_credentials: str = Field(
        default="",
        description="Path to Google Cloud service account JSON"
    )
    google_credentials_json: str = Field(
        default="",
        description="Raw Google Cloud service account JSON content"
    )
    google_cloud_project: str = Field(
        default="",
        description="Google Cloud project ID"
    )
    
    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_cors_origins: List[str] = ["http://localhost:8501", "http://localhost:3000"]
    
    # Processing
    max_image_size_mb: int = 20
    max_batch_size: int = 100
    default_detection_type: Literal["TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION"] = "DOCUMENT_TEXT_DETECTION"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    
    # Nested settings
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    
    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
