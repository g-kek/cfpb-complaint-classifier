import logging

from fastapi import APIRouter, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from cfpb_complaint_classifier.schemas import ProcessRequest, ProcessResponse
from cfpb_complaint_classifier.services.inference import LoadedModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/process", response_model=ProcessResponse)
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    model: LoadedModel = request.app.state.model
    try:
        category = await run_in_threadpool(model.predict, payload.text)
    except Exception as exc:
        logger.error("Model inference failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference is unavailable",
        ) from None
    return ProcessResponse(
        category=category, model_name=model.name, model_version=model.version
    )
