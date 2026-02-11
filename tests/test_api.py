"""Integration tests for the /predict and /health API endpoints."""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
import numpy as np

from app.inference.batcher import InferenceBatcher
from app.inference.predictor import Predictor
from app.main import create_app


def _make_mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.25, 0.75]])
    return model


@pytest_asyncio.fixture
async def app():
    with patch("app.model.loader.load_model", return_value=_make_mock_model()):
        application = create_app()
        # Manually run startup logic since lifespan events don't fire under httpx
        model = _make_mock_model()
        predictor = Predictor(model)
        batcher = InferenceBatcher(predictor, batch_window_ms=5, queue_max_size=100)
        batcher.start()
        application.state.model = model
        application.state.predictor = predictor
        application.state.batcher = batcher
        yield application
        await batcher.stop()


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestPredictEndpoint:
    @pytest.mark.asyncio
    async def test_predict_returns_probability(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json={"features": [0.1] * 10})
            assert resp.status_code == 200
            data = resp.json()
            assert "pred" in data
            assert isinstance(data["pred"], float)
            assert 0.0 <= data["pred"] <= 1.0

    @pytest.mark.asyncio
    async def test_predict_empty_features_rejected(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json={"features": []})
            assert resp.status_code == 422  # validation error

    @pytest.mark.asyncio
    async def test_predict_missing_body_rejected(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict")
            assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/metrics")
            assert resp.status_code == 200
            body = resp.text
            assert "inference_request_latency_seconds" in body
            assert "inference_queue_depth" in body
