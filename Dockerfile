FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group web --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --group web

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "cfpb_complaint_classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]
