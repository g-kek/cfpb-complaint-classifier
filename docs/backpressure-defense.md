# Защита backpressure: сценарий и ответы

Все команды выполняются из корня репозитория. Нужны Docker, Compose и модель
`cfpb-complaint-classifier@champion` в MLflow. Первый запуск с обучением описан
в README. PR и отправку ссылки преподавателю выполняет автор работы.

## 1. Автоматические доказательства

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run ruff format --check .
RUN_INTEGRATION_TESTS=1 uv run pytest
docker compose --profile training --profile batch config --quiet
```

На macOS при отсутствии `/var/run/docker.sock`:

```bash
export DOCKER_HOST="$(docker context inspect --format '{{.Endpoints.docker.Host}}')"
RUN_INTEGRATION_TESTS=1 uv run pytest tests/test_pipeline_integration.py -v --no-cov
```

Девять интеграционных сценариев используют отдельные RabbitMQ 4.1.8 и PostgreSQL
16 через Testcontainers: переполнение, возобновление источника, 503 без результата,
сохранение бюджета после перезапуска, недоступная DLQ, некорректный JSON,
конкурентные INSERT/счётчики, повторную доставку при заполненной очереди,
SIGTERM, SIGKILL до и после коммита и два воркера с одним task_id. Сценарии
сигналов запускают отдельный процесс Uvicorn;
замена модели тестовой используется только в тестовом subprocess, без флагов
подмены в production-коде. Реальная загрузка модели из MLflow проверяется отдельно.
Uvicorn может повторно поднять SIGTERM после корректного завершения lifespan;
поэтому код выхода `-15` сам по себе не означает потерю задачи: тест проверяет БД.

## 2. Измерения E1–E4

```bash
docker compose up -d postgres rabbitmq mlflow
docker compose build app
uv run python scripts/run_backpressure_experiments.py \
  --output docs/experiments/my-run --tasks 800 --seconds 20 --delay 0.2
uv run python scripts/plot_backpressure_experiments.py docs/experiments/my-run
```

Папка результата должна быть новой. Скрипт создаёт отдельную очередь каждого
режима, запускает реальную модель из MLflow по alias и сохраняет метрики,
конфигурации, события публикации, времена задач и журнал коммитов. После общего
окна наблюдения источник заканчивает тот же конечный файл, а очередь полностью
дренируется. Скрипт пересоздаёт сервисы `app`/`worker` для каждого режима и
останавливает воркеры после дренирования. Обычную конфигурацию восстановить:

```bash
docker compose up -d --no-build --scale worker=1 app worker
```

## 3. Живая демонстрация перегрузки

```bash
docker compose up -d --no-build --scale worker=1 app worker
docker compose --profile batch run --rm source \
  /app/data/cfpb/test.csv --batch-id defense-v1
```

Источник работает в отдельном терминале. Открыть http://127.0.0.1:15672
(`cfpb` / `cfpb-local-password` при стандартном `.env`). Для `complaints.tasks`
показать ready около 20, unacked не более 2 на один воркер и снижение скорости
publisher после заполнения. Ограничение `x-max-length` относится к ready,
а не к ready + unacked.

Отправить запрос под нагрузкой:

```bash
curl -i -X POST http://127.0.0.1:8080/api/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"text":"My bank did not apply my mortgage payment."}'
```

При заполненной очереди — 503 и `Retry-After: 1`. Между освобождением слота и
новым заполнением может прийти 202: это корректная гонка, очередь в тот момент
принимает сообщение. API не ждёт свободного места и не повторяет публикацию.
Источники не создают PENDING вообще.

## 4. Второй воркер и доступность event loop

```bash
docker compose up -d --no-build --scale worker=2 worker
docker compose exec worker python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8001/healthz', timeout=1).read())"
```

В UI появляются два consumer, суммарно unacked ≤ 4; скорость источника растёт
без изменения кода. Каждый процесс загружает alias однократно, имеет два потока
и два слота исполнения. `predict` выполняется в пуле; health-check отвечает
во время inference, что дополнительно проверено блокирующей тестовой моделью.

## 5. Изменение лимита/prefetch

Аргументы уже существующей очереди RabbitMQ нельзя переобъявить другими.
Для демонстрации выбрать новое имя, затем использовать одинаковые настройки
для всех процессов; удалять пользовательские очереди не требуется.

```bash
export TASK_QUEUE_NAME=defense.limit3.prefetch1
export QUEUE_MAX_LENGTH=3
export PREFETCH_COUNT=1
export WORKER_CONCURRENCY=2
docker compose up -d --no-build --force-recreate --scale worker=1 app worker
docker compose --profile batch run --rm source \
  /app/data/cfpb/test.csv --batch-id defense-limits-v1
```

Ready ≤ 3, unacked ≤ 1. Один из двух потоков намеренно простаивает: prefetch 1
ограничивает конкурентность раньше пула. Prefetch > размера пула конфигурация
отклоняет. Новая очередь не содержит сообщений из предыдущей демонстрации.
После проверки остановить источник Ctrl+C и восстановить настройки:

```bash
unset TASK_QUEUE_NAME QUEUE_MAX_LENGTH PREFETCH_COUNT WORKER_CONCURRENCY
docker compose up -d --no-build --force-recreate --scale worker=1 app worker
```

## 6. SIGTERM, SIGKILL и идемпотентность

Во время нагрузки в другом терминале:

```bash
docker compose stop -t 40 worker
docker compose up -d --no-build worker
```

SIGTERM останавливает приём и даёт FastStream завершить принятые сообщения;
ack отправляется после коммита. Ready может остаться: это ещё не принятые
воркером задачи. Пока воркер выключен, источник ждёт на заполненной очереди.
После старта обработка и публикация продолжаются. `SHUTDOWN_TIMEOUT_SECONDS=30`,
Compose ждёт 40 секунд; при увеличении длительности inference увеличить оба
значения с запасом на все попытки и запись в БД.

Аварийная остановка:

```bash
docker compose kill -s SIGKILL worker
docker compose up -d --no-build worker
```

Unacked возвращаются в очередь. Счётчик попыток уже закоммичен в PostgreSQL,
поэтому перезапуск не обнуляет бюджет. Успешно сохранённый task_id определяется
перед новым inference, повторно подтверждается и сохраняет одну строку.
Даже при конкурентной доставке PRIMARY KEY + ON CONFLICT защищают результат.
Тест с паузой после реального коммита точно воспроизводит окно «commit → crash → ack».
При трёх падениях до результата бюджет также исчерпывается и задача уйдёт в DLQ.

## 7. Некорректное сообщение

```bash
docker compose exec -T app python - <<'PY'
import asyncio
from uuid import uuid4
from cfpb_complaint_classifier.core.config import get_settings
from cfpb_complaint_classifier.messaging import create_broker, task_exchange

async def main():
    settings = get_settings()
    async with create_broker(settings) as broker:
        message_id = f"defense-poison-{uuid4()}"
        await broker.publish(
            b'{invalid json', content_type='application/json',
            message_id=message_id, persist=True, mandatory=True,
            exchange=task_exchange(settings), routing_key=settings.task_queue_name,
        )
        print(message_id)

asyncio.run(main())
PY
```

Публиковать при наличии места (при необходимости временно остановить источник).
Через три попытки в `complaints.tasks.dlq` появятся исходные байты с заголовками
`x-task-attempts: 3`, `x-task-error: ValidationError`. В логах нет содержимого
жалобы. DLQ не имеет consumer: сообщение хранится для разбора, исправления
и осознанного повторного запуска. Если DLQ недоступна, исходное сообщение
остаётся в основной очереди; исчерпанный бюджет не запускает inference снова.

## Теория: 12 ответов

1. **Команда и событие.** Команда просит выполнить действие (`ClassifyComplaint`)
   и адресована обработчику; событие фиксирует случившийся факт
   (`ComplaintClassified`) и может иметь несколько независимых подписчиков.
   Event-driven подходит для асинхронных процессов, независимых реакций и буферизации
   нагрузки; для простого запроса с немедленным ответом часто достаточно HTTP.
2. **Зачем брокер.** Он разрывает зависимость доступности producer/consumer,
   буферизует всплески, маршрутизирует и повторно доставляет сообщения. Цена:
   инфраструктура, eventual consistency, дубли, задержка и сложность диагностики.
3. **Маршрутизация и лог.** Producer публикует в exchange с routing key;
   bindings направляют сообщение в queue. Direct сопоставляет ключ, topic —
   шаблон, fanout рассылает всем связанным очередям. RabbitMQ распределяет
   сообщения очереди между consumers и удаляет после ack; Kafka хранит журнал
   разделов с offset, retention и независимыми consumer groups.
4. **Сохранность.** Durable очередь/exchange, persistent сообщения, publisher
   confirms + mandatory, ack после фиксации результата, повторы с тем же ID,
   идемпотентное хранилище. Для отказа узла нужна репликация и кластер; локальная
   classic-очередь на одном узле не переживает потерю его диска. Неопределённость
   при утрате confirm остаётся и приводит к допустимому дублю.
5. **Ack/nack/reject.** Ack подтверждает обработку и разрешает удалить сообщение;
   nack/reject отклоняют и могут вернуть его в очередь. Nack поддерживает multiple,
   reject — одно сообщение. При закрытии соединения unacked доставляются снова.
   Publisher confirm подтверждает приём брокером, а consumer ack — обработку.
6. **Гарантии доставки.** At-most-once допускает потери; at-least-once допускает
   дубли. Здесь — at-least-once с одним сохранённым результатом на task_id.
   Exactly-once для произвольных внешних эффектов без общей транзакции не обещаем;
   на практике речь о дедупликации и атомарной фиксации конкретного эффекта.
7. **Идемпотентный consumer.** Повторный task_id не меняет итог. Нужны UNIQUE/PK
   и атомарная запись, либо inbox с ID в той же транзакции, что бизнес-изменения.
   SELECT перед INSERT без ограничения допускает гонку двух воркеров.
8. **Dual write, outbox/inbox.** Запись в БД и публикация брокеру не атомарны:
   возможны запись без сообщения или сообщение без записи. Outbox сохраняет
   бизнес-изменение и событие одной транзакцией; relay публикует с повторами.
   Inbox фиксирует обработанный ID и эффект одной транзакцией. В нашем API
   нет записи PENDING, поэтому входная двойная запись устранена; БД + ack
   согласуются повторной доставкой и идемпотентностью.
9. **Polling/CDC/LISTEN.** Polling читает outbox, захватывает строки и отмечает
   публикацию после confirm; нужен контроль гонок и повторов. CDC читает WAL
   и даёт меньшую задержку при большей операционной сложности. LISTEN/NOTIFY
   не является долговечным журналом: при отключении слушателя уведомление
   теряется; его можно использовать только как сигнал проверить таблицу.
10. **Ошибки.** Задержка перед `nack(requeue=True)` исключает busy loop.
    Брокер повторно доставляет сообщение, устойчивый счётчик ограничивает число
    обработок, затем poison message отправляется в DLQ. Во время задержки
    сообщение остаётся `unacked` и занимает слот. Ошибка PostgreSQL при записи
    результата вызывает повтор этой записи без нового inference и без DLQ.
    При отказе переноса в DLQ исходное сообщение не подтверждается.
11. **Producer быстрее consumer.** Без ограничений растут ready/unacked, память
    и время ожидания. Backpressure передаёт перегрузку вверх и источник ждёт;
    load shedding отказывает внешнему клиенту (503); rate limiting ограничивает
    скорость заданным пределом, не обязательно связанным с состоянием downstream.
12. **Порядок.** FIFO одной очереди не гарантирует порядок завершения при нескольких
    consumers, потоках и повторных доставках. Для порядка по сущности используют
    один consumer на партицию/ключ и последовательную обработку либо номера
    версий в хранилище. Глобальный порядок ограничивает параллелизм и требует
    единой координации; в этой работе он не заявлен.

Документация: [RabbitMQ confirms](https://www.rabbitmq.com/docs/confirms),
[лимит очереди](https://www.rabbitmq.com/docs/maxlength),
[FastStream acknowledgements](https://faststream.ag2.ai/latest/getting-started/acknowledgement/).

Автоматический прогон живого сценария с настоящей моделью (отдельная очередь,
лимит 3, prefetch 1, 503, SIGTERM, SIGKILL и poison message):

```bash
uv run python scripts/verify_backpressure_defense.py
```

Он восстанавливает обычную конфигурацию `app`/одного `worker` в `finally`.
Измерения E1–E4 запускать отдельно, чтобы вмешательство не меняло условия опыта.
