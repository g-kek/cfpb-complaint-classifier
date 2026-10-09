FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY docker/init_bucket.py ./init_bucket.py
RUN uv sync --frozen --no-cache --no-default-groups --group tracking --no-install-project

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN useradd --create-home mlflow && chown mlflow:mlflow /app
USER mlflow

EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]
