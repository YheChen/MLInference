from __future__ import annotations

import time
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.inference.batcher import OverloadedError
from app.metrics.prometheus import REQUEST_LATENCY_SECONDS

router = APIRouter()


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    pred: float


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request) -> PredictResponse:
    start = time.perf_counter()
    prob: float | None = None
    try:
        x = np.array(req.features, dtype=np.float32).reshape(1, -1)
        batcher = getattr(request.app.state, "batcher", None)
        if batcher is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference pipeline unavailable",
            )
        prob = await batcher.enqueue(x)
    except OverloadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server overloaded",
        ) from exc
    finally:
        REQUEST_LATENCY_SECONDS.observe(time.perf_counter() - start)

    return PredictResponse(pred=prob)