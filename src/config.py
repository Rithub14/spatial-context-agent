"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings sourced from environment variables or .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/spatial_agent"

    # CLIP model
    clip_model_name: str = "ViT-B/32"
    device: str = "cpu"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Security (toggleable)
    enable_auth: bool = False
    enable_rate_limit: bool = False
    api_key: str = "dev-key-change-in-production"
    rate_limit_rpm: int = 60

    # Azure deployment
    azure_resource_group: str = "spatial-agent-rg"
    azure_location: str = "westeurope"


settings = Settings()
