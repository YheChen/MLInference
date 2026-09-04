from __future__ import annotations

import time

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.metrics.prometheus import REQUEST_LATENCY_SECONDS, REQUESTS_TOTAL


class MetricsMiddleware(BaseHTTPMiddleware):
	"""Records latency and outcome for every /predict request.

	Registered last so it is the outermost middleware, which is the whole point:
	requests shed by backpressure (503) or killed by the timeout (504) never
	reach the route handler, so a histogram observed inside the handler silently
	omits exactly the traffic you need to see during an overload event.
	"""

	async def dispatch(
		self, request: Request, call_next: RequestResponseEndpoint
	) -> Response:
		if request.url.path != "/predict":
			return await call_next(request)

		start = time.perf_counter()
		status_code = 500
		try:
			response = await call_next(request)
			status_code = response.status_code
			return response
		finally:
			REQUEST_LATENCY_SECONDS.observe(time.perf_counter() - start)
			REQUESTS_TOTAL.labels(status=str(status_code)).inc()
