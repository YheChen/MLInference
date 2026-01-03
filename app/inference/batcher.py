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
		try:
			while self._running:
				item = await self._queue.get()
				batch: List[_BatchItem] = [item]
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

				await self._execute_batch(batch)
		except asyncio.CancelledError:
			pass

	async def _execute_batch(self, batch: List[_BatchItem]) -> None:
		if not batch:
			return

		batch_array = np.vstack([item.features for item in batch])
		BATCH_SIZE.observe(len(batch))

		start = time.perf_counter()
		try:
			predictions = await asyncio.to_thread(self._predictor.predict, batch_array)
		except Exception as exc:  # pragma: no cover - surfaced to callers
			for item in batch:
				if not item.future.done():
					item.future.set_exception(exc)
			QUEUE_DEPTH.set(self._queue.qsize())
			return

		inference_latency = time.perf_counter() - start
		BATCH_LATENCY_SECONDS.observe(inference_latency)

		for value, item in zip(predictions, batch):
			if not item.future.done():
				item.future.set_result(float(value))

		QUEUE_DEPTH.set(self._queue.qsize())

	async def _flush_queue_with_cancellation(self) -> None:
		while True:
			try:
				item = self._queue.get_nowait()
			except asyncio.QueueEmpty:
				break
			if not item.future.done():
				item.future.set_exception(asyncio.CancelledError())
		QUEUE_DEPTH.set(self._queue.qsize())
