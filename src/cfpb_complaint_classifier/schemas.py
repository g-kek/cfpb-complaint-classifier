import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(strict=True, min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            raise ValueError("Text must contain non-whitespace characters")
        return normalized


class ProcessResponse(BaseModel):
    category: str
    model_name: str
    model_version: str


class HealthStatus(StrEnum):
    healthy = "healthy"
    unavailable = "unavailable"


class ReportStatus(StrEnum):
    healthy = "healthy"
    degraded = "degraded"


class LivenessResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    version: str


class ComponentHealth(BaseModel):
    status: HealthStatus
    version: str | None = None
    response_time_ms: float


class HealthReport(BaseModel):
    status: ReportStatus
    components: dict[str, ComponentHealth]
