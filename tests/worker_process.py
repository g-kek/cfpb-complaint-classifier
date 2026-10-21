import asyncio
import os
from pathlib import Path
from unittest.mock import Mock

import uvicorn

from cfpb_complaint_classifier import worker
from cfpb_complaint_classifier.services.inference import LoadedModel

worker.load_registered_model = lambda _: LoadedModel(
    Mock(predict=Mock(return_value=["Debt collection"])), "integration-test", "1"
)
if marker := os.getenv("TEST_PAUSE_AFTER_COMMIT"):
    original_save = worker.save_result

    async def save_then_pause(*args):
        await original_save(*args)
        await asyncio.to_thread(Path(marker).touch)
        await asyncio.Event().wait()

    worker.save_result = save_then_pause

if __name__ == "__main__":
    uvicorn.run(
        worker.create_worker_app(), host="127.0.0.1", port=int(os.environ["TEST_PORT"])
    )
