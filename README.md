# cfpb-complaint-classifier

MLOps-система для автоматической классификации и анализа клиентских жалоб.

## Что реализовано

- Асинхронное FastAPI-приложение со структурой `src/cfpb_complaint_classifier`.
- `GET /healthz` для быстрой liveness-проверки без версии API: этот endpoint нужен инфраструктуре и балансировщикам, поэтому он намеренно короткий и стабильный.
- `GET /api/v1/version` возвращает версию установленного Python-пакета из metadata проекта.
- `GET /api/v1/health` выполняет end-to-end проверку PostgreSQL и возвращает статус, версию компонента и время ответа.
- `uv` lock-файл для воспроизводимой установки.
- Ruff lint/format, pytest coverage, pre-commit, Dockerfile, Docker Compose, CI и CD.

## Локальный запуск

```bash
python3 -m uv sync --all-groups
python3 -m uv run uvicorn cfpb_complaint_classifier.app:app --reload
```

Приложение будет доступно на `http://127.0.0.1:8000`.

## Запуск через Docker Compose

```bash
docker compose up --build
```

Compose поднимает два сервиса:

- `app` на порту `8000`;
- `postgres` на порту `5432` с volume `postgres-data`.

Оба сервиса имеют healthcheck, лимиты ресурсов и json-file logging с ротацией.

## Проверки

```bash
python3 -m uv run ruff check .
python3 -m uv run ruff format --check .
python3 -m uv run pytest
python3 -m uv run pre-commit run --all-files
```

Coverage настроен в `pyproject.toml` и падает ниже 85%.

## CI/CD

CI (`.github/workflows/ci.yml`) запускается на push в `main` и на pull request. Он устанавливает `uv`, синхронизирует зависимости, проверяет Ruff lint, Ruff format и pytest с coverage.

CD (`.github/workflows/cd.yml`) запускается на push в `main` и на semver tag вида `v*.*.*`. Registry выбран GitHub Container Registry, потому что он не требует внешних секретов для учебного проекта: публикация идёт через `GITHUB_TOKEN` в `ghcr.io/${{ github.repository }}`.

Стратегия версионирования образов:

- каждый push получает immutable tag `sha-<commit>`;
- push в `main` дополнительно обновляет `main` и `latest`;
- tag `v1.2.3` публикует `v1.2.3`, `1.2.3` и `1.2`.

Такой запуск даёт быстрый образ для проверки `main` и воспроизводимые релизные версии для защиты.
