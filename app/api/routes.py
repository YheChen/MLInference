from __future__ import annotations

import time
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.config import FEATURE_DIM
from app.inference.batcher import OverloadedError
from app.metrics.prometheus import REQUEST_LATENCY_SECONDS
from app.utils.logging import generate_request_id, get_logger

logger = get_logger()

router = APIRouter()


class PredictRequest(BaseModel):
    # Exactly FEATURE_DIM values. Enforcing the width here means a wrong-sized
    # payload is a 422 for the sender alone, rather than an error raised while
    # stacking a batch that also contains other clients' valid requests.
    features: List[float] = Field(
        ...,
        min_length=FEATURE_DIM,
        max_length=FEATURE_DIM,
        description=f"Exactly {FEATURE_DIM} float features.",
    )


class PredictResponse(BaseModel):
    pred: float


@router.get("/health")
async def health(request: Request) -> dict:
    """Liveness probe.

    Reports unhealthy when the batch loop has died, which is otherwise invisible:
    the queue keeps accepting requests that will never be served.
    """
    batcher = getattr(request.app.state, "batcher", None)
    if batcher is None or not batcher.is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference pipeline unavailable",
        )
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request) -> PredictResponse:
    rid = generate_request_id()
    start = time.perf_counter()
    prob: float | None = None
    status_code = 200
    try:
        x = np.array(req.features, dtype=np.float32).reshape(1, -1)
        batcher = getattr(request.app.state, "batcher", None)
        if batcher is None:
            status_code = 503
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference pipeline unavailable",
            )
        prob = await batcher.enqueue(x)
    except OverloadedError as exc:
        status_code = 503
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server overloaded",
        ) from exc
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        REQUEST_LATENCY_SECONDS.observe(latency_ms / 1000)
        queue_depth = getattr(
            getattr(request.app.state, "batcher", None), "queue_size", None
        )
        logger.info(
            "predict",
            extra={
                "request_id": rid,
                "method": "POST",
                "path": "/predict",
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
                "queue_depth": queue_depth,
            },
        )

    return PredictResponse(pred=prob)
