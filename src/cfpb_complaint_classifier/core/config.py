import math

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", allow_inf_nan=False
    )

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/cfpb"
    )
    log_level: str = "INFO"
    mlflow_tracking_uri: str = "http://127.0.0.1:5001"
    mlflow_model_name: str = "cfpb-complaint-classifier"
    mlflow_model_alias: str = "champion"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    task_queue_name: str = "complaints.tasks"
    backpressure_enabled: bool = True
    queue_max_length: int = Field(default=20, ge=1)
    worker_concurrency: int = Field(default=2, ge=1)
    prefetch_count: int | None = Field(default=None, ge=1)
    inference_delay_seconds: float = Field(default=1.0, ge=0)
    publish_timeout_seconds: float = Field(default=1.0, gt=0)
    source_retry_initial_seconds: float = Field(default=0.05, gt=0)
    source_retry_max_seconds: float = Field(default=0.5, gt=0)
    shutdown_timeout_seconds: float = Field(default=30.0, gt=0)
    task_max_attempts: int = Field(default=3, ge=1)
    task_retry_delay_seconds: float = Field(default=0.5, gt=0)

    @model_validator(mode="after")
    def validate_limits(self) -> "Settings":
        if self.prefetch_count and self.prefetch_count > self.worker_concurrency:
            raise ValueError("prefetch_count must not exceed worker_concurrency")
        if self.source_retry_initial_seconds > self.source_retry_max_seconds:
            raise ValueError("Initial retry delay must not exceed maximum delay")
        return self

    @property
    def effective_prefetch_count(self) -> int:
        if not self.backpressure_enabled:
            return 0
        return self.prefetch_count or self.worker_concurrency

    @property
    def retry_after_seconds(self) -> int:

        return max(1, math.ceil(self.inference_delay_seconds / self.worker_concurrency))


def get_settings() -> Settings:
    return Settings()
