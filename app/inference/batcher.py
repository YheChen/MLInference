from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from app.config import BATCH_MAX_SIZE, BATCH_WINDOW_MS, QUEUE_MAX_SIZE
from app.inference.predictor import Predictor
from app.metrics.prometheus import (
	BATCH_LATENCY_SECONDS,
	BATCH_SIZE,
	QUEUE_DEPTH,
	QUEUE_REJECTIONS,
)
from app.utils.logging import get_logger

logger = get_logger()


class OverloadedError(RuntimeError):
	"""Raised when the inference queue is saturated."""


@dataclass
class _BatchItem:
	features: np.ndarray
	future: asyncio.Future
	enqueue_time: float


class InferenceBatcher:
	def __init__(
		self,
		predictor: Predictor,
		*,
		max_batch_size: int = BATCH_MAX_SIZE,
		batch_window_ms: int = BATCH_WINDOW_MS,
		queue_max_size: int = QUEUE_MAX_SIZE,
	) -> None:
		self._predictor = predictor
		self._max_batch_size = max_batch_size
		self._batch_window_seconds = batch_window_ms / 1000
		self._queue: asyncio.Queue[_BatchItem] = asyncio.Queue(maxsize=queue_max_size)
		self._task: asyncio.Task | None = None
		self._running = False

	@property
	def queue_size(self) -> int:
		return self._queue.qsize()

	@property
	def is_healthy(self) -> bool:
		"""True while the background batch loop is alive and draining the queue.

		A dead loop is indistinguishable from a healthy one at the queue level —
		`enqueue()` still accepts work, it just never completes — so /health has
		to inspect the task itself for the liveness probe to mean anything.
		"""
		return self._running and self._task is not None and not self._task.done()

	def start(self) -> None:
		if self._running:
			return
		self._running = True
		loop = asyncio.get_running_loop()
		self._task = loop.create_task(self._batch_loop())

	async def stop(self) -> None:
		if not self._running:
			return
		self._running = False
		if self._task:
			self._task.cancel()
			try:
				await self._task
			except asyncio.CancelledError:
				pass
		await self._flush_queue_with_cancellation()

	async def enqueue(self, features: np.ndarray) -> float:
		if not self._running:
			raise RuntimeError("Batcher has not been started")

		loop = asyncio.get_running_loop()
		future: asyncio.Future = loop.create_future()
		item = _BatchItem(features=features, future=future, enqueue_time=time.perf_counter())

		try:
			self._queue.put_nowait(item)
			QUEUE_DEPTH.set(self._queue.qsize())
		except asyncio.QueueFull as exc:
			QUEUE_REJECTIONS.inc()
			raise OverloadedError("Inference queue is full") from exc

		return await future

	async def _batch_loop(self) -> None:
		while self._running:
			# `batch` is filled in place so that if we are cancelled mid-collection
			# the items already pulled off the queue can still be failed cleanly
			# instead of being dropped with their callers left awaiting.
			batch: List[_BatchItem] = []
			try:
				await self._collect_batch(batch)
				await self._execute_batch(batch)
			except asyncio.CancelledError:
				self._fail_batch(batch, asyncio.CancelledError())
				raise
			except Exception as exc:  # noqa: BLE001 - the loop must outlive any one batch
				# Without this, a single malformed batch kills the background task
				# for good: `enqueue()` keeps accepting work into a queue nothing
				# drains, and every subsequent request hangs until it times out.
				self._fail_batch(batch, exc)
				logger.error(
					"batch_failed",
					extra={
						"batch_size": len(batch),
						"queue_depth": self._queue.qsize(),
						"detail": f"{type(exc).__name__}: {exc}",
					},
				)

	async def _collect_batch(self, batch: List[_BatchItem]) -> None:
		"""Drain up to `max_batch_size` items, or whatever arrives in the window."""
		batch.append(await self._queue.get())
		deadline = time.perf_counter() + self._batch_window_seconds

		while len(batch) < self._max_batch_size:
			remaining = deadline - time.perf_counter()
			if remaining <= 0:
				break
			try:
				next_item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
			except asyncio.TimeoutError:
				break
			batch.append(next_item)

	async def _execute_batch(self, batch: List[_BatchItem]) -> None:
		if not batch:
			return

		BATCH_SIZE.observe(len(batch))
		start = time.perf_counter()

		# np.vstack belongs inside the guard: mismatched feature counts raise here,
		# before the model is ever called.
		try:
			batch_array = np.vstack([item.features for item in batch])
			predictions = await asyncio.to_thread(self._predictor.predict, batch_array)
		except Exception as exc:
			self._fail_batch(batch, exc)
			QUEUE_DEPTH.set(self._queue.qsize())
			return

		inference_latency = time.perf_counter() - start
		BATCH_LATENCY_SECONDS.observe(inference_latency)

		if len(predictions) != len(batch):
			# Never leave a caller awaiting a future that will never be resolved.
			self._fail_batch(
				batch,
				RuntimeError(
					f"Predictor returned {len(predictions)} results "
					f"for a batch of {len(batch)}"
				),
			)
			QUEUE_DEPTH.set(self._queue.qsize())
			return

		for value, item in zip(predictions, batch):
			if not item.future.done():
				item.future.set_result(float(value))

		QUEUE_DEPTH.set(self._queue.qsize())

	@staticmethod
	def _fail_batch(batch: List[_BatchItem], exc: BaseException) -> None:
		"""Resolve every still-pending future so no caller hangs."""
		for item in batch:
			if not item.future.done():
				item.future.set_exception(exc)

	async def _flush_queue_with_cancellation(self) -> None:
		while True:
			try:
				item = self._queue.get_nowait()
			except asyncio.QueueEmpty:
				break
			if not item.future.done():
				item.future.set_exception(asyncio.CancelledError())
		QUEUE_DEPTH.set(self._queue.qsize())
