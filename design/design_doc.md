# Design Document — ML Inference Service

## 1. Problem Statement

Build a production-style ML inference service that demonstrates systems engineering discipline: tail-latency control, micro-batching, overload handling, and observability. The ML model is intentionally simple (logistic regression); the focus is on the infrastructure surrounding it.

**Target SLA**: p95 end-to-end request latency ≤ 50 ms under moderate load (50 concurrent users).

---

## 2. Goals and Non-Goals

### Goals

- Serve predictions over HTTP with low, predictable tail latency.
- Batch individual requests to amortise per-inference overhead.
- Shed load gracefully when the system is saturated (no tail collapse).
- Expose Prometheus metrics sufficient to diagnose latency and throughput issues.
- Provide structured JSON logs for operational visibility.

### Non-Goals

- ML model accuracy or training sophistication.
- Multi-model serving or A/B testing.
- Horizontal auto-scaling (service runs as a single process).
- GPU inference or model quantisation.

---

## 3. Architecture Overview

```
  Client
    │
    ▼
┌──────────────────────────────────────┐
│  FastAPI  (single uvicorn worker)    │
│  ┌──────────────────────────────┐    │
│  │ TimeoutMiddleware  (100 ms)  │    │
│  │ BackpressureMiddleware (80%) │    │
│  └──────────┬───────────────────┘    │
│             │                        │
│  ┌──────────▼───────────────────┐    │
│  │ /predict handler             │    │
│  │  → enqueue(features) → Future│    │
│  └──────────┬───────────────────┘    │
│             │                        │
│  ┌──────────▼───────────────────┐    │
│  │ InferenceBatcher             │    │
│  │  bounded asyncio.Queue(2000) │    │
│  │  batch_window = 5 ms         │    │
│  │  max_batch_size = 32         │    │
│  └──────────┬───────────────────┘    │
│             │                        │
│  ┌──────────▼───────────────────┐    │
│  │ Predictor  (to_thread)       │    │
│  │  sklearn LogisticRegression  │    │
│  └──────────────────────────────┘    │
│                                      │
│  /metrics  →  Prometheus exposition  │
│  /health   →  liveness probe         │
└──────────────────────────────────────┘
```

### Component responsibilities

| Component                  | Role                                                                                                                                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TimeoutMiddleware**      | Hard 100 ms deadline on every request. Returns `504` if breached. Prevents slow requests from consuming resources indefinitely.                                                                |
| **BackpressureMiddleware** | Monitors batcher queue depth. Rejects new `/predict` requests with `503` once the queue crosses 80 % of capacity. Sheds load before the queue is completely full.                              |
| **InferenceBatcher**       | Background `asyncio.Task` that collects incoming requests into batches. Uses a 5 ms window and max-batch-size of 32. Each caller awaits an `asyncio.Future` resolved when its batch completes. |
| **Predictor**              | Thin wrapper around the sklearn model. Runs synchronous `predict_proba()` in `asyncio.to_thread` to avoid blocking the event loop.                                                             |
| **Prometheus metrics**     | Histograms for request latency, batch size, and batch inference time. Gauge for queue depth. Counters for rejections and timeouts.                                                             |
| **Structured logger**      | JSON-formatted logs with request ID, latency, queue depth, and status code.                                                                                                                    |

---

## 4. Key Design Decisions

### 4.1 Why micro-batching?

Calling `model.predict_proba()` once per request is wasteful — sklearn (and most ML frameworks) are faster on batched numpy arrays due to vectorised operations. Batching amortises the Python → C boundary crossing and thread dispatch overhead.

**Trade-off**: batching introduces a _window delay_ (up to 5 ms) where early-arriving requests wait for the batch to fill. We keep this window small so the latency cost is bounded.

### 4.2 Why a bounded queue?

An unbounded queue hides overload: the queue grows without limit, memory climbs, and _every_ request eventually times out. A bounded queue (2 000 slots) makes overload _visible_: when the queue is full, we reject immediately with `503` rather than accepting work we cannot serve within SLA.

### 4.3 Why a high-watermark (80 %)?

If we only reject at 100 % capacity, requests already in the queue may still breach the timeout. By shedding load at 80 %, the remaining 400 slots provide headroom for in-flight requests to complete within the 100 ms budget.

### 4.4 Why a hard timeout instead of retry/exponential back-off?

From the client's perspective, a fast failure is preferable to a slow one. The 100 ms timeout guarantees worst-case latency: the client gets a `504` and can retry immediately rather than waiting an unbounded amount of time.

### 4.5 Why `asyncio.to_thread` instead of a process pool?

The sklearn model is lightweight and GIL-releasing during numpy computation. `to_thread` avoids the serialisation overhead of `ProcessPoolExecutor` and keeps memory usage low. For GPU models or CPU-heavy workloads, a process pool or dedicated inference server (Triton, TF Serving) would be more appropriate.

### 4.6 Why single-worker uvicorn?

Simplicity. A single worker keeps all state (model, batcher queue, Prometheus counters) in one process. Horizontal scaling is handled by running multiple containers behind a load balancer, each with one worker. This sidesteps shared-memory counters and cross-process queue coordination.

---

## 5. Alternatives Considered

| Alternative                            | Why rejected                                                                                                                     |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **gRPC instead of REST**               | REST is simpler to demo and debug with curl. gRPC would reduce serialisation overhead at high RPS but adds client complexity.    |
| **Triton / TF Serving**                | Overkill for a sklearn model. The goal is to demonstrate the _engineering_ around inference, not to use an off-the-shelf server. |
| **Redis-backed queue**                 | Adds an external dependency. `asyncio.Queue` is lock-free in the event loop and sufficient for a single-process service.         |
| **Per-request thread**                 | No batching benefit; thread-pool exhaustion under load.                                                                          |
| **Adaptive batching (dynamic window)** | More complex; hard to reason about latency bounds. Fixed window + max-batch-size is simple and predictable.                      |

---

## 6. Failure Modes

| Failure                         | Symptom                                               | Mitigation                                                                                   |
| ------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Model file missing at startup   | App crashes immediately.                              | Fail-fast; obvious in logs. Train model before deploy.                                       |
| Sustained over-capacity traffic | Queue fills → 503 rejections rise.                    | By design: backpressure middleware sheds load. Upstream should back off or scale replicas.   |
| Model inference becomes slow    | Batch latency rises → timeout middleware fires `504`. | Monitor `inference_batch_latency_seconds`. Investigate model/data drift.                     |
| Memory leak in batcher          | Queue depth stays near max even under low load.       | Monitor `inference_queue_depth`. If consistently high, check for futures not being resolved. |
| Prometheus scrape fails         | Metrics stop updating. No impact on inference.        | Alert on Prometheus target health.                                                           |

---

## 7. Metrics & Observability

### Prometheus metrics

| Metric                              | Type      | Purpose                                                           |
| ----------------------------------- | --------- | ----------------------------------------------------------------- |
| `inference_request_latency_seconds` | Histogram | End-to-end latency per request (includes queue wait + inference). |
| `inference_batch_size`              | Histogram | Requests per batch — shows batching efficiency.                   |
| `inference_batch_latency_seconds`   | Histogram | Model inference time per batch (excludes queue wait).             |
| `inference_queue_depth`             | Gauge     | Current queue occupancy — primary overload indicator.             |
| `inference_queue_rejections_total`  | Counter   | Requests rejected by backpressure or queue-full.                  |
| `inference_request_timeouts_total`  | Counter   | Requests killed by the timeout middleware.                        |

### Key PromQL queries

```promql
# Tail latency
histogram_quantile(0.99, rate(inference_request_latency_seconds_bucket[2m]))

# Batching efficiency
rate(inference_batch_size_sum[1m]) / rate(inference_batch_size_count[1m])

# Overload signal
rate(inference_queue_rejections_total[1m]) > 0
```

### Structured log fields

`timestamp`, `level`, `request_id`, `method`, `path`, `status_code`, `latency_ms`, `queue_depth`, `batch_size`, `detail`.

---

## 8. Load Test Results

| Scenario | Users | Duration | p50  | p95   | p99   | Throughput | 503s | 504s |
| -------- | ----- | -------- | ---- | ----- | ----- | ---------- | ---- | ---- |
| steady   | 50    | 2 min    | 7 ms | 14 ms | 29 ms | ~458 req/s | 0    | 0    |

p95 = 14 ms — well within the 50 ms SLA target.

---

## 9. Future Work

- **Horizontal scaling**: run multiple containers behind a reverse proxy; add instance labels to Prometheus metrics.
- **Adaptive batching**: dynamically adjust the batch window based on queue depth and observed latency.
- **Model versioning**: support hot-swapping models without downtime.
- **Grafana dashboard**: pre-built dashboard JSON for the metrics above.
- **Integration tests**: automated end-to-end test that starts the server, runs a short Locust scenario, and asserts p95 < 50 ms in CI.
