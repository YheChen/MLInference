from typing import List
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    pred: float


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    return PredictResponse(pred=0.5)
