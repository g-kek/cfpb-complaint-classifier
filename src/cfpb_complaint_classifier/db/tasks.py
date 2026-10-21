import asyncio
from uuid import UUID

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Uuid, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from cfpb_complaint_classifier.core.config import get_settings
from cfpb_complaint_classifier.db.engine import create_engine
from cfpb_complaint_classifier.schemas import TaskResult

metadata = MetaData()
task_results = Table(
    "task_results",
    metadata,
    Column("task_id", Uuid, primary_key=True),
    Column("category", String, nullable=False),
    Column("model_name", String, nullable=False),
    Column("model_version", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=False),
)
task_attempts = Table(
    "task_attempts",
    metadata,
    Column("message_key", String, primary_key=True),
    Column("attempts", Integer, nullable=False),
)


async def claim_attempt(
    engine: AsyncEngine, message_key: str, limit: int
) -> int | None:

    statement = insert(task_attempts).values(message_key=message_key, attempts=1)
    statement = statement.on_conflict_do_update(
        index_elements=[task_attempts.c.message_key],
        set_={"attempts": task_attempts.c.attempts + 1},
        where=task_attempts.c.attempts < limit,
    ).returning(task_attempts.c.attempts)
    async with engine.begin() as connection:
        attempt = (await connection.execute(statement)).scalar_one_or_none()
    return attempt


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)


async def save_result(engine: AsyncEngine, result: TaskResult) -> None:
    statement = (
        insert(task_results)
        .values(**result.model_dump())
        .on_conflict_do_nothing(index_elements=[task_results.c.task_id])
    )
    async with engine.begin() as connection:
        await connection.execute(statement)


async def get_result(engine: AsyncEngine, task_id: UUID) -> TaskResult | None:
    async with engine.connect() as connection:
        row = (
            (
                await connection.execute(
                    select(task_results).where(task_results.c.task_id == task_id)
                )
            )
            .mappings()
            .one_or_none()
        )
    return TaskResult.model_validate(dict(row)) if row else None


async def main() -> None:
    engine = create_engine(get_settings())
    try:
        await create_schema(engine)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
