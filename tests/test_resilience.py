"""Regression tests for batch-loop resilience and liveness reporting.

Before these fixes, a single request carrying the wrong number of features
raised inside `np.vstack` (outside the guarded block), which propagated out of
`_batch_loop` and killed the background task permanently. `enqueue()` kept
accepting work into a queue nothing drained, so every later request hung until
its deadline — while /health still reported 200.

Every await here is bounded: a regression must fail fast, not hang the suite.
"""

import asyncio

import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app.config import FEATURE_DIM
from app.inference.batcher import InferenceBatcher
from app.inference.predictor import Predictor
from app.main import create_app

TIMEOUT = 2.0

GOOD = np.array([[0.5] * FEATURE_DIM], dtype=np.float32)
MALFORMED = np.array([[0.5] * 3], dtype=np.float32)


def _working_predictor() -> Predictor:
    model = MagicMock()
    model.predict_proba.side_effect = lambda x: np.column_stack([1 - x[:, 0], x[:, 0]])
    return Predictor(model)


@pytest_asyncio.fixture
async def app():
    with patch("app.model.loader.load_model", return_value=MagicMock()):
        application = create_app()
        predictor = _working_predictor()
        batcher = InferenceBatcher(predictor, batch_window_ms=5, queue_max_size=100)
        batcher.start()
        application.state.batcher = batcher
        yield application
        await batcher.stop()


class TestApiInputValidation:
    @pytest.mark.asyncio
    async def test_wrong_feature_count_is_rejected_before_the_batcher(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
            assert resp.status_code == 422

            # The malformed request never reached the pipeline, so the service
            # still serves valid traffic.
            assert app.state.batcher.is_healthy
            ok = await client.post("/predict", json={"features": [0.1] * FEATURE_DIM})
            assert ok.status_code == 200

    @pytest.mark.asyncio
    async def test_too_many_features_is_rejected(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/predict", json={"features": [0.1] * (FEATURE_DIM + 1)}
            )
            assert resp.status_code == 422


class TestBatchLoopSurvivesFailures:
    @pytest.mark.asyncio
    async def test_mismatched_shapes_do_not_kill_the_loop(self):
        """The original crash: mixed feature widths blow up in np.vstack."""
        batcher = InferenceBatcher(
            _working_predictor(), max_batch_size=32, batch_window_ms=50, queue_max_size=100
        )
        batcher.start()
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    batcher.enqueue(GOOD),
                    batcher.enqueue(MALFORMED),
                    batcher.enqueue(GOOD),
                    return_exceptions=True,
                ),
                timeout=TIMEOUT,
            )
            # Every caller is answered — the failure is fast, not a hang.
            assert len(results) == 3
            assert all(isinstance(r, BaseException) for r in results)

            # And the loop is still alive to serve the next request.
            assert batcher.is_healthy
            recovered = await asyncio.wait_for(batcher.enqueue(GOOD), timeout=TIMEOUT)
            assert isinstance(recovered, float)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_predictor_exception_does_not_kill_the_loop(self):
        model = MagicMock()
        model.predict_proba.side_effect = RuntimeError("model exploded")
        batcher = InferenceBatcher(
            Predictor(model), max_batch_size=8, batch_window_ms=20, queue_max_size=100
        )
        batcher.start()
        try:
            with pytest.raises(RuntimeError, match="model exploded"):
                await asyncio.wait_for(batcher.enqueue(GOOD), timeout=TIMEOUT)

            assert batcher.is_healthy

            # Recover the model; the same batcher keeps serving.
            model.predict_proba.side_effect = lambda x: np.column_stack(
                [1 - x[:, 0], x[:, 0]]
            )
            recovered = await asyncio.wait_for(batcher.enqueue(GOOD), timeout=TIMEOUT)
            assert isinstance(recovered, float)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_short_prediction_array_fails_batch_instead_of_hanging(self):
        """A predictor returning fewer rows than the batch must not strand callers."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.1, 0.9]])  # always one row
        batcher = InferenceBatcher(
            Predictor(model), max_batch_size=8, batch_window_ms=50, queue_max_size=100
        )
        batcher.start()
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    batcher.enqueue(GOOD),
                    batcher.enqueue(GOOD),
                    batcher.enqueue(GOOD),
                    return_exceptions=True,
                ),
                timeout=TIMEOUT,
            )
            assert all(isinstance(r, BaseException) for r in results)
            assert batcher.is_healthy
        finally:
            await batcher.stop()


class TestHealthReportsLiveness:
    @pytest.mark.asyncio
    async def test_health_is_ok_while_loop_runs(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_reports_503_when_batch_loop_is_dead(self, app):
        batcher = app.state.batcher
        batcher._task.cancel()
        await asyncio.sleep(0.01)  # let the cancellation land

        assert not batcher.is_healthy

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_health_reports_503_before_startup(self):
        with patch("app.model.loader.load_model", return_value=MagicMock()):
            application = create_app()  # no batcher on app.state yet
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 503
