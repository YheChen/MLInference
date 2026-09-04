"""Tests for request-outcome metrics and abandoned-work dropping.

Latency used to be observed inside the /predict handler, so requests shed by
backpressure (503) or killed by the timeout (504) — which never reach the
handler — were missing from `inference_request_latency_seconds` entirely. That
is exactly the traffic an overload dashboard needs. It is now observed by
MetricsMiddleware, which sits outside both.
"""

import asyncio

import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from prometheus_client import REGISTRY

from app.config import FEATURE_DIM, _env_int
from app.inference.batcher import InferenceBatcher
from app.inference.predictor import Predictor
from app.main import create_app

GOOD = np.array([[0.5] * FEATURE_DIM], dtype=np.float32)
PAYLOAD = {"features": [0.1] * FEATURE_DIM}


def _sample(name: str, labels: dict | None = None) -> float:
    """Current value of a metric sample, 0.0 if it has not been emitted yet."""
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def _working_model() -> MagicMock:
    model = MagicMock()
    model.predict_proba.side_effect = lambda x: np.column_stack([1 - x[:, 0], x[:, 0]])
    return model


@pytest_asyncio.fixture
async def app():
    with patch("app.model.loader.load_model", return_value=MagicMock()):
        application = create_app()
        batcher = InferenceBatcher(
            Predictor(_working_model()), batch_window_ms=5, queue_max_size=100
        )
        batcher.start()
        application.state.batcher = batcher
        yield application
        await batcher.stop()


class TestRequestOutcomeMetrics:
    @pytest.mark.asyncio
    async def test_successful_request_is_counted_exactly_once(self, app):
        before_lat = _sample("inference_request_latency_seconds_count")
        before_200 = _sample("inference_requests_total", {"status": "200"})

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json=PAYLOAD)
            assert resp.status_code == 200

        # Exactly one — the handler no longer observes it as well.
        assert _sample("inference_request_latency_seconds_count") == before_lat + 1
        assert _sample("inference_requests_total", {"status": "200"}) == before_200 + 1

    @pytest.mark.asyncio
    async def test_shed_request_appears_in_latency_histogram(self, app):
        """The regression: 503s from backpressure never reached the histogram."""
        before_lat = _sample("inference_request_latency_seconds_count")
        before_503 = _sample("inference_requests_total", {"status": "503"})

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            batcher = app.state.batcher
            with patch.object(
                type(batcher),
                "queue_size",
                new_callable=lambda: property(lambda self: 10_000),
            ):
                resp = await client.post("/predict", json=PAYLOAD)
                assert resp.status_code == 503

        assert _sample("inference_request_latency_seconds_count") == before_lat + 1
        assert _sample("inference_requests_total", {"status": "503"}) == before_503 + 1

    @pytest.mark.asyncio
    async def test_timed_out_request_appears_in_latency_histogram(self, app):
        before_lat = _sample("inference_request_latency_seconds_count")
        before_504 = _sample("inference_requests_total", {"status": "504"})

        # Stall the model well past the 100 ms deadline.
        app.state.batcher._predictor._model.predict_proba.side_effect = (
            lambda x: __import__("time").sleep(0.5)
            or np.column_stack([1 - x[:, 0], x[:, 0]])
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json=PAYLOAD)
            assert resp.status_code == 504

        assert _sample("inference_request_latency_seconds_count") == before_lat + 1
        assert _sample("inference_requests_total", {"status": "504"}) == before_504 + 1

    @pytest.mark.asyncio
    async def test_health_and_metrics_do_not_pollute_inference_latency(self, app):
        before = _sample("inference_request_latency_seconds_count")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get("/health")
            await client.get("/metrics")
        assert _sample("inference_request_latency_seconds_count") == before


class TestAbandonedWorkIsDropped:
    @pytest.mark.asyncio
    async def test_cancelled_request_is_not_inferred(self):
        model = _working_model()
        batcher = InferenceBatcher(
            Predictor(model), max_batch_size=8, batch_window_ms=80, queue_max_size=100
        )
        batcher.start()
        try:
            before_dropped = _sample("inference_batch_dropped_total")

            task = asyncio.create_task(batcher.enqueue(GOOD))
            await asyncio.sleep(0.01)  # let it reach the queue
            task.cancel()
            await asyncio.sleep(0.25)  # window closes, batch is executed

            # The caller is gone, so no model time was spent on it.
            assert model.predict_proba.call_count == 0
            assert _sample("inference_batch_dropped_total") == before_dropped + 1
            assert batcher.is_healthy
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_live_request_in_same_batch_is_still_served(self):
        model = _working_model()
        batcher = InferenceBatcher(
            Predictor(model), max_batch_size=8, batch_window_ms=80, queue_max_size=100
        )
        batcher.start()
        try:
            abandoned = asyncio.create_task(batcher.enqueue(GOOD))
            live = asyncio.create_task(batcher.enqueue(GOOD))
            await asyncio.sleep(0.01)
            abandoned.cancel()

            result = await asyncio.wait_for(live, timeout=2.0)
            assert isinstance(result, float)

            # Only the surviving request was stacked into the batch.
            assert model.predict_proba.call_count == 1
            assert model.predict_proba.call_args[0][0].shape[0] == 1
        finally:
            await batcher.stop()


class TestConfigEnvOverrides:
    def test_env_int_falls_back_to_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MLI_TEST_KNOB", raising=False)
        assert _env_int("MLI_TEST_KNOB", 7) == 7

    def test_env_int_reads_the_environment(self, monkeypatch):
        monkeypatch.setenv("MLI_TEST_KNOB", "42")
        assert _env_int("MLI_TEST_KNOB", 7) == 42

    def test_env_int_rejects_a_non_integer_loudly(self, monkeypatch):
        monkeypatch.setenv("MLI_TEST_KNOB", "not-a-number")
        with pytest.raises(ValueError, match="must be an integer"):
            _env_int("MLI_TEST_KNOB", 7)
