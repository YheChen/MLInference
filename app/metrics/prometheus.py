from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REQUEST_LATENCY_SECONDS = Histogram(
    "inference_request_latency_seconds",
    "End-to-end latency per inference request",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)

BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Number of requests processed per batch",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

BATCH_LATENCY_SECONDS = Histogram(
    "inference_batch_latency_seconds",
    "Time spent running model inference for a batch",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25),
)

QUEUE_DEPTH = Gauge(
    "inference_queue_depth",
    "Current depth of the inference pre-batching queue",
)

QUEUE_REJECTIONS = Counter(
    "inference_queue_rejections_total",
    "Total requests rejected due to load shedding/backpressure",
)

REQUEST_TIMEOUTS = Counter(
    "inference_request_timeouts_total",
    "Total requests terminated by the timeout middleware",
)

metrics_router = APIRouter()


@metrics_router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
