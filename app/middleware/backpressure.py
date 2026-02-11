from __future__ import annotations

from fastapi import Request, status
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import QUEUE_HIGH_WATERMARK
from app.metrics.prometheus import QUEUE_REJECTIONS
from app.utils.logging import get_logger

logger = get_logger()


class BackpressureMiddleware(BaseHTTPMiddleware):
	"""Rejects traffic early when the inference queue is already stressed."""

	def __init__(
		self,
		app,
		*,
		queue_attr: str = "batcher",
		high_watermark: int = QUEUE_HIGH_WATERMARK,
	) -> None:
		super().__init__(app)
		self._queue_attr = queue_attr
		self._high_watermark = high_watermark

	async def dispatch(
		self, request: Request, call_next: RequestResponseEndpoint
	) -> Response:
		if request.url.path != "/predict":
			return await call_next(request)

		batcher = getattr(request.app.state, self._queue_attr, None)
		if batcher is None:
			return await call_next(request)

		if batcher.queue_size >= self._high_watermark:
			QUEUE_REJECTIONS.inc()
			logger.warning(
				"backpressure_reject",
				extra={
					"path": request.url.path,
					"queue_depth": batcher.queue_size,
					"detail": "queue at high watermark",
				},
			)
			return JSONResponse(
				status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
				content={"detail": "Server overloaded"},
			)

		return await call_next(request)
