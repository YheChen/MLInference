"""Tests for timeout and backpressure middleware."""

import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app.inference.batcher import InferenceBatcher
from app.inference.predictor import Predictor
from app.main import create_app


def _make_mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest_asyncio.fixture
async def app():
    with patch("app.model.loader.load_model", return_value=_make_mock_model()):
        application = create_app()
        model = _make_mock_model()
        predictor = Predictor(model)
        batcher = InferenceBatcher(predictor, batch_window_ms=5, queue_max_size=100)
        batcher.start()
        application.state.model = model
        application.state.predictor = predictor
        application.state.batcher = batcher
        yield application
        await batcher.stop()


class TestBackpressureMiddleware:
    @pytest.mark.asyncio
    async def test_rejects_when_queue_high(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            batcher = app.state.batcher
            with patch.object(type(batcher), "queue_size", new_callable=lambda: property(lambda self: 2000)):
                resp = await client.post("/predict", json={"features": [0.1] * 10})
                assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_passes_health_check(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestTimeoutMiddleware:
    @pytest.mark.asyncio
    async def test_normal_request_succeeds(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json={"features": [0.1] * 10})
            assert resp.status_code == 200
            data = resp.json()
            assert "pred" in data
            assert 0.0 <= data["pred"] <= 1.0
