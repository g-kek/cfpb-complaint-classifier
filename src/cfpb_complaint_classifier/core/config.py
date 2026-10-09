from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/cfpb"
    )
    log_level: str = "INFO"
    mlflow_tracking_uri: str = "http://127.0.0.1:5001"
    mlflow_model_name: str = "cfpb-complaint-classifier"
    mlflow_model_alias: str = "champion"


def get_settings() -> Settings:
    return Settings()
