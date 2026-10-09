FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-default-groups --group web --group serving --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-cache --no-default-groups --group web --group serving

RUN useradd --create-home appuser \
    && chown appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "cfpb_complaint_classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]
