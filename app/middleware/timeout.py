from __future__ import annotations

import asyncio

from fastapi import Request, status
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import REQUEST_TIMEOUT_MS
from app.metrics.prometheus import REQUEST_TIMEOUTS


class TimeoutMiddleware(BaseHTTPMiddleware):
	"""Enforces a hard ceiling on request latency to protect SLAs."""

	def __init__(self, app, *, timeout_ms: int = REQUEST_TIMEOUT_MS) -> None:
		super().__init__(app)
		self._timeout_seconds = timeout_ms / 1000

	async def dispatch(
		self, request: Request, call_next: RequestResponseEndpoint
	) -> Response:
		try:
			return await asyncio.wait_for(
				call_next(request), timeout=self._timeout_seconds
			)
		except asyncio.TimeoutError:
			REQUEST_TIMEOUTS.inc()
			return JSONResponse(
				status_code=status.HTTP_504_GATEWAY_TIMEOUT,
				content={"detail": "Request timed out"},
			)
