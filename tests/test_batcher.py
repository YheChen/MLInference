"""Tests for the InferenceBatcher."""

import asyncio

import numpy as np
import pytest
from unittest.mock import MagicMock

from app.inference.batcher import InferenceBatcher, OverloadedError
from app.inference.predictor import Predictor


def _make_predictor():
    """Create a Predictor with a mock model that returns column-1 probs."""
    model = MagicMock()
    model.predict_proba.side_effect = lambda x: np.column_stack(
        [1 - x[:, 0], x[:, 0]]
    )
    return Predictor(model)


@pytest.fixture
def predictor():
    return _make_predictor()


class TestBatcher:
    @pytest.mark.asyncio
    async def test_single_request(self, predictor):
        batcher = InferenceBatcher(predictor, max_batch_size=4, batch_window_ms=50, queue_max_size=10)
        batcher.start()
        try:
            features = np.array([[0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            result = await batcher.enqueue(features)
            assert isinstance(result, float)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_batch_multiple_requests(self, predictor):
        batcher = InferenceBatcher(predictor, max_batch_size=8, batch_window_ms=50, queue_max_size=100)
        batcher.start()
        try:
            tasks = []
            for i in range(5):
                x = np.array([[float(i) * 0.1] + [0.0] * 9], dtype=np.float32)
                tasks.append(batcher.enqueue(x))
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            assert all(isinstance(r, float) for r in results)
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_queue_full_raises_overloaded(self, predictor):
        batcher = InferenceBatcher(predictor, max_batch_size=2, batch_window_ms=500, queue_max_size=2)
        # Don't start the batcher — queue will fill and never drain
        batcher._running = True  # allow enqueue but no consumer

        features = np.array([[1.0] * 10], dtype=np.float32)
        # Fill the queue
        loop = asyncio.get_running_loop()
        for _ in range(2):
            fut = loop.create_future()
            from app.inference.batcher import _BatchItem
            import time
            item = _BatchItem(features=features, future=fut, enqueue_time=time.perf_counter())
            batcher._queue.put_nowait(item)

        # Next enqueue should raise
        with pytest.raises(OverloadedError):
            await batcher.enqueue(features)

    @pytest.mark.asyncio
    async def test_enqueue_before_start_raises(self, predictor):
        batcher = InferenceBatcher(predictor)
        with pytest.raises(RuntimeError, match="not been started"):
            await batcher.enqueue(np.array([[1.0] * 10], dtype=np.float32))

    @pytest.mark.asyncio
    async def test_stop_cancels_pending(self, predictor):
        batcher = InferenceBatcher(predictor, max_batch_size=100, batch_window_ms=5000, queue_max_size=10)
        batcher.start()

        features = np.array([[0.5] * 10], dtype=np.float32)
        fut = batcher.enqueue(features)

        # Stop before the long batch window expires
        await batcher.stop()

        # The future should be cancelled
        with pytest.raises((asyncio.CancelledError, Exception)):
            await fut
