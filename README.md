# cfpb-complaint-classifier

MLOps-система для автоматической классификации и анализа клиентских жалоб

Выполнили:
Зыков Макар 489608
Кек Герман 466149

## Что реализовано

- Асинхронное FastAPI-приложение со структурой `src/cfpb_complaint_classifier`.
- `GET /healthz` для быстрой liveness-проверки без версии API: этот endpoint нужен инфраструктуре и балансировщикам, поэтому он намеренно короткий и стабильный.
- `GET /api/v1/version` возвращает версию установленного Python-пакета из metadata проекта.
- `GET /api/v1/health` выполняет end-to-end проверку PostgreSQL и возвращает статус, версию компонента и время ответа.
- `uv` lock-файл для воспроизводимой установки.
- Ruff lint/format, pytest coverage, pre-commit, Dockerfile, Docker Compose, CI и CD.
- MLflow с PostgreSQL для метаданных и MinIO для артефактов.
- Подготовка выборки CFPB, EDA, три варианта классификатора и Model Registry.

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

Compose поднимает сервисы:

- `app` на порту `8080`;
- `postgres` на порту `5432` с volume `postgres-data`.
- `mlflow` на порту `5001`;
- `mlflow-db` — отдельный PostgreSQL без внешнего порта;
- `minio` — S3 API на порту `9000`, консоль на `9001`;
- `minio-init` — создаёт bucket `mlflow-artifacts` и завершается.

Приложение и его PostgreSQL имеют healthcheck, лимиты ресурсов и ротацию логов.
MLflow, его БД и MinIO тоже проверяются через healthcheck.

## Обучение с MLflow

Команды выполняются из корня репозитория. Нужны Docker Compose и uv.

```bash
uv sync --all-groups
docker compose up -d --build mlflow
docker compose ps -a
```

MLflow UI: http://127.0.0.1:5001. MinIO: http://127.0.0.1:9001,
логин `minio`, пароль `minio-local-password`. Значения для локального запуска
заданы в Compose; их можно заменить через `.env` по примеру `.env.example`.
Внешние порты MLflow и MinIO привязаны к localhost.
MLflow использует снаружи порт 5001, чтобы не конфликтовать с AirPlay на macOS.
При смене `MLFLOW_PORT` укажите тот же порт в `--tracking-uri` у скриптов.

MinIO собирается из исходников релиза `RELEASE.2025-10-15T17-29-55Z`:
старые официальные образы больше недоступны для анонимного скачивания.
Первая сборка занимает время: Docker скачивает зависимости Go и Python.

Backend Store — БД `mlflow` в `mlflow-db`, volume `mlflow-db-data`.
Там находятся Runs, параметры, метрики, сведения о датасетах и Registry.
Artifact Store — bucket `mlflow-artifacts` в MinIO, volume `minio-data`.
MLflow передаёт артефакты в MinIO через свой сервер, поэтому скриптам обучения
и будущему inference-приложению достаточно адреса MLflow.
Скрипты отключают прямые multipart-загрузки и скачивания: имя `minio`
доступно внутри Docker, а клиент на хосте получает файлы через MLflow.

### Данные

Задача: по `Consumer complaint narrative` предсказать `Product`.
Текущая выгрузка CFPB больше не содержит тексты жалоб:
это изменение опубликовано в [release notes](https://cfpb.github.io/api/ccdb/release-notes.html).
Используем официальный [архив CFPB](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
за ноябрь 2022 — август 2023. URL и SHA-256 архива зафиксированы в `training/prepare.py`.

```bash
uv run --group ml python -m training.prepare
```

Скрипт скачивает архив, извлекает жалобы за январь 2023 и сохраняет в `data/cfpb/`:

- `snapshot.zip` — исходный архив;
- `raw.csv` — строки выбранного периода с четырьмя используемыми колонками;
- `train.csv`, `validation.csv`, `test.csv`;
- `manifest.json` — источник, контрольные суммы, фильтры, seed и состав splits.

Оставляем тексты от 40 символов, непустые метки и ID. Удаляем повторяющиеся ID,
одинаковые тексты и все строки, где одному тексту соответствуют разные категории.
Классы с менее чем 100 оставшимися записями исключаем. Затем берём до 20 000
записей с сохранением пропорций классов и делим 70/15/15, seed 42.
В датах нижняя граница включена, верхняя исключена.

Для другого периода или размера выборки:

```bash
uv run --group ml python -m training.prepare \
  --start-date 2023-02-01 --end-date 2023-03-01 \
  --max-samples 10000 --output data/cfpb-february
```

Период должен попадать в выбранный архив. Можно передать сохранённый ZIP или CSV
с исходными названиями колонок через `--input`; для другого источника указать `--source`.
Непустая папка не перезаписывается. При повторной подготовке используйте новую папку.
Для точного повторения экспериментов используйте сохранённые splits, а не новую выгрузку.

### EDA и эксперименты

```bash
uv run --group ml python -m training.eda
uv run --group ml python -m training.train
```

Адрес сервера по умолчанию `http://127.0.0.1:5001`; его можно заменить через
`MLFLOW_TRACKING_URI` или `--tracking-uri`. Experiment: `cfpb-product-classification`.
Для своих splits у обеих команд есть `--data-dir`.

EDA Run сохраняет распределение классов, длины текстов, статистику фильтрации,
доли классов в splits и частые слова по категориям. Сводки сохраняются в CSV и JSON.
В его артефактах также лежат архив, raw.csv и готовые splits.
Локальные графики и сводки находятся в `reports/eda/`.

Обучение создаёт три Run:

| Run | Вариант |
| --- | --- |
| `baseline` | DummyClassifier, всегда самый частый класс |
| `tfidf-logreg` | TF-IDF по отдельным словам + Logistic Regression |
| `tfidf-linear-svc` | TF-IDF со словами и биграммами + LinearSVC с весами классов |

Векторизатор обучается только на train. Оба обученных классификатора сохраняются
как sklearn Pipeline вместе с TF-IDF. В последнем варианте меняются сразу
классификатор, n-граммы и веса классов; сравнение показывает качество всего варианта,
а не отдельный эффект каждого изменения.

Основная метрика — validation macro F1: каждый класс имеет одинаковый вес.
Дополнительно сохраняются accuracy, weighted F1, balanced accuracy, время обучения
и среднее время предсказания на элемент батча после прогрева.
Для каждого Run есть classification report, нормированная confusion matrix
и до 100 примеров ошибок. Скрипт также сохраняет свой код и lock-файл в артефакты.

Модель выбирается по validation macro F1. Test оценивается только для выбранной
модели и не участвует в выборе. Результаты и сравнение находятся в `reports/training/`.
В UI выделите три Run и нажмите Compare; сравнивайте `validation_macro_f1`.

### Dataset Tracking и Registry

Каждый Run связан с датасетами через `mlflow.log_input`. У обучения контексты
`training` и `validation`, у выбранной модели дополнительно `test`.

- `source` — URL исходного архивного ZIP, из которого получены записи;
- `digest` — автоматически вычисляемый MLflow отпечаток конкретного split;
- `lineage` — архив → период → фильтрация и splits → Run → версия модели.

Источник описывает исходные данные, а не готовый split. Преобразования описаны
в manifest; SHA-256 файлов хранится отдельно от digest MLflow. Перед EDA и обучением
проверяются контрольные суммы. Обучение также проверяет, что связанный EDA Run
использовал тот же manifest. Dataset Tracking хранит метаданные; сами файлы
сохраняет EDA Run. Его ID записывается в tags обучающих Runs.

Создаются две версии `cfpb-complaint-classifier`; `champion` указывает на вариант
с лучшим validation macro F1. Повторное обучение добавляет версии и обновляет alias.
Номера версий и Run ID записаны в `reports/training/registry.json`.
В UI откройте Models → cfpb-complaint-classifier: у версий видны исходные Runs,
signature, пример входа и зависимости.

Проверить загрузку модели:

```bash
MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false uv run --group ml python - <<'PY'
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5001")
client = MlflowClient()
version = client.get_model_version_by_alias("cfpb-complaint-classifier", "champion")
model = mlflow.sklearn.load_model(f"models:/cfpb-complaint-classifier/{version.version}")
print("Version:", version.version)
print(model.predict(["A debt collector keeps calling about a debt I do not owe."]))
PY
```

Для второго этапа: вход Pipeline — список строк, выход — массив строк с категориями.
Загрузка через Registry выполняется при старте приложения. Номер версии нужно
сохранить вместе с моделью, чтобы ответ `/process` показывал фактически загруженную версию.
Внутри Compose адрес MLflow — `http://mlflow:5000`.
Сам `/process` относится к следующему этапу и пока не реализован.

Остановить инфраструктуру: `docker compose down`. Без `-v` данные MLflow и MinIO
сохраняются. Папки `data/` и `reports/` не попадают в Git и Docker build context.

## Проверки

```bash
python3 -m uv run ruff check .
python3 -m uv run ruff format --check .
python3 -m uv run --group ml pytest
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
