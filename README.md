# ML Inference Service

A production-style ML inference service built to demonstrate **latency engineering, batching, overload handling, and observability** — not ML complexity.

The model is intentionally trivial (logistic regression). The interesting work is in the infrastructure: bounded queues, micro-batching, backpressure middleware, hard timeouts, Prometheus metrics, and structured logging.

---

## Architecture

```
                ┌────────────┐
  HTTP POST ──▶│  FastAPI    │
  /predict     │  Middleware │  ← TimeoutMiddleware (100 ms hard cap)
               │             │  ← BackpressureMiddleware (queue high-watermark)
               └─────┬──────┘
                     │
               ┌─────▼──────┐
               │  Batcher    │  bounded async queue (2 000 slots)
               │  (5 ms win, │  drains into batches of up to 32
               │   batch 32) │
               └─────┬──────┘
                     │
               ┌─────▼──────┐
               │  Predictor  │  sklearn LogisticRegression
               │  (thread)   │  runs in asyncio.to_thread
               └─────┬──────┘
                     │
               ┌─────▼──────┐
               │  Prometheus │  latency histograms, queue depth,
               │  /metrics   │  rejection & timeout counters
               └─────────────┘
```

### Request flow

1. Middleware enforces a **100 ms hard timeout** and rejects requests early when queue depth exceeds the **high-watermark (80 % of 2 000)**.
2. The `/predict` handler pushes a per-request `Future` onto the batcher's bounded `asyncio.Queue`.
3. A background loop drains the queue into batches (up to 32 items or a 5 ms window, whichever comes first) and runs inference in a thread.
4. Results are dispatched back to each waiting `Future`.

---

## Design Decisions

| Decision                    | Rationale                                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Bounded queue (2 000)**   | Prevents unbounded memory growth under load; enables clear back-pressure signal.                                            |
| **High-watermark at 80 %**  | Middleware sheds load _before_ the queue is completely full, so enqueued requests still finish within SLA.                  |
| **5 ms batch window**       | Balances throughput (larger batches) vs. latency (requests waiting in the window).                                          |
| **Hard 100 ms timeout**     | Guarantees worst-case client latency; any request stuck in the queue past the deadline gets a `504`.                        |
| **Structured JSON logging** | One JSON object per log line — trivial to ingest into ELK / CloudWatch / any aggregator.                                    |
| **Prometheus histograms**   | Histogram buckets let you compute arbitrary percentiles server-side with `histogram_quantile`.                              |
| **Single uvicorn worker**   | Keeps the demo simple and avoids shared-state complexity; real deployments would scale horizontally behind a load balancer. |

---

## Quick Start

### 1. Train the model

```bash
python -m training.train
# → Saved model to app/model/model.pkl
```

### 2. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Smoke test

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.3, 0.5, 1.2, -0.7, 0.0, 0.8, -1.1, 0.4, 0.6]}'
# → {"pred": 0.732...}
```

### 4. Run with Docker Compose (app + Prometheus)

```bash
docker compose -f docker/docker-compose.yml up --build
```

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`

Verify Prometheus is scraping the app:  
Prometheus → Status → Targets → `ml-inference` should be **UP**.

---

## Load Testing

Three pre-built scenarios via Locust, selected with the `SCENARIO` env var:

| Scenario     | Purpose                                                     | Example Command                                                                       |
| ------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **steady**   | Baseline latency under moderate load                        | `SCENARIO=steady locust -f load_test/locustfile.py --headless -u 50 -r 10 -t 2m`      |
| **ramp**     | Linear ramp 0 → 300 users; detects latency degradation      | `SCENARIO=ramp locust -f load_test/locustfile.py --headless -t 5m`                    |
| **overload** | Push past queue capacity; validates backpressure + shedding | `SCENARIO=overload locust -f load_test/locustfile.py --headless -u 1000 -r 100 -t 2m` |

After a run, export latency data with:

```bash
# CSV export (built into Locust --headless)
SCENARIO=steady locust -f load_test/locustfile.py --headless -u 50 -r 10 -t 2m --csv=results/steady

# Or pull percentiles from Prometheus
python load_test/report_latency.py --prom http://localhost:9090 --range 2m
```

---

## Benchmarks

> **Hardware**: Apple M-series / 16 GB (single uvicorn worker)  
> **Model**: sklearn LogisticRegression, 10 features  
> **Test**: `SCENARIO=steady`, 50 users, 2 min

| Metric           | Target         | Observed          |
| ---------------- | -------------- | ----------------- |
| p50              | —              | _run and fill in_ |
| p95              | ≤ 50 ms        | _run and fill in_ |
| p99              | —              | _run and fill in_ |
| Throughput       | —              | _run and fill in_ |
| Rejections (503) | 0 under steady | _run and fill in_ |
| Timeouts (504)   | 0 under steady | _run and fill in_ |

> Fill these in after running the load test. Use `python load_test/report_latency.py` to auto-generate the numbers from Prometheus.

---

## Prometheus Queries

Paste these into Prometheus (`http://localhost:9090/graph`) or Grafana:

```promql
# p50 / p95 / p99 end-to-end request latency
histogram_quantile(0.50, rate(inference_request_latency_seconds_bucket[2m]))
histogram_quantile(0.95, rate(inference_request_latency_seconds_bucket[2m]))
histogram_quantile(0.99, rate(inference_request_latency_seconds_bucket[2m]))

# Request throughput (req/s)
sum(rate(inference_request_latency_seconds_count[1m]))

# Queue depth (instantaneous)
inference_queue_depth

# Rejection rate
rate(inference_queue_rejections_total[1m])

# Timeout rate
rate(inference_request_timeouts_total[1m])

# Average batch size
rate(inference_batch_size_sum[1m]) / rate(inference_batch_size_count[1m])

# Batch inference latency (model time only)
histogram_quantile(0.95, rate(inference_batch_latency_seconds_bucket[2m]))
```

---

## Operational Playbook

### SLO

| SLI                 | Target                  |
| ------------------- | ----------------------- |
| p95 request latency | ≤ 50 ms                 |
| Error rate (5xx)    | < 1 % under normal load |
| Availability        | `/health` returns 200   |

### Overload Signals

| Signal              | Metric                                      | Action                                                                                       |
| ------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Queue depth rising  | `inference_queue_depth` > 1 600             | Back-pressure middleware starts rejecting (automatic). Scale horizontally or reduce traffic. |
| Rejections spiking  | `inference_queue_rejections_total` rate > 0 | Upstream is sending more than the service can absorb. Add replicas or increase queue size.   |
| Timeouts spiking    | `inference_request_timeouts_total` rate > 0 | Batches are taking too long. Check model latency (`inference_batch_latency_seconds`).        |
| p95 approaching SLA | `histogram_quantile(0.95, ...)` > 40 ms     | Warning zone. Investigate queue depth and batch latency.                                     |

### Failure Modes

| Failure                | Behavior                                                  | Mitigation                                                        |
| ---------------------- | --------------------------------------------------------- | ----------------------------------------------------------------- |
| Model file missing     | Startup crashes (fail-fast).                              | Ensure `python -m training.train` ran before starting the server. |
| Queue full             | `enqueue()` raises `OverloadedError` → client sees `503`. | Expected under extreme load — load shedding by design.            |
| Inference thread hangs | Timeout middleware returns `504` after 100 ms.            | Investigate model or data issues.                                 |
| Prometheus down        | App continues serving; metrics are lost.                  | Prometheus is decoupled — no impact on inference.                 |

### Log Fields (structured JSON)

Every log line includes:

| Field         | Description                                              |
| ------------- | -------------------------------------------------------- |
| `timestamp`   | ISO-8601                                                 |
| `level`       | INFO / WARNING / ERROR                                   |
| `request_id`  | 12-char hex, unique per request                          |
| `method`      | HTTP method                                              |
| `path`        | Request path                                             |
| `status_code` | HTTP response status                                     |
| `latency_ms`  | End-to-end wall time                                     |
| `queue_depth` | Batcher queue depth at response time                     |
| `detail`      | Human-readable context (e.g., "queue at high watermark") |

---

## How to Reproduce (full checklist)

```bash
# 1. Clone & install
git clone <repo-url> && cd MLInference
python3 -m pip install -r requirements.txt

# 2. Train the model
python -m training.train

# 3. Start the server
uvicorn app.main:app --port 8000 &

# 4. Run baseline load test
SCENARIO=steady locust -f load_test/locustfile.py --headless -u 50 -r 10 -t 2m --csv=results/steady

# 5. Pull percentiles from Prometheus (if running)
python load_test/report_latency.py

# 6. (Optional) Docker Compose with Prometheus
docker compose -f docker/docker-compose.yml up --build
```

---

## Project Structure

```
app/
  main.py              – FastAPI app factory + lifecycle
  config.py            – Tuning knobs (batch size, queue bounds, timeout)
  api/routes.py        – /predict, /health endpoints
  inference/batcher.py – Async micro-batching with bounded queue
  inference/predictor.py – Thin model wrapper
  model/loader.py      – Loads model.pkl at startup
  metrics/prometheus.py – Histogram / Gauge / Counter definitions
  middleware/backpressure.py – Early rejection at queue high-watermark
  middleware/timeout.py – Hard 100 ms deadline per request
  utils/logging.py     – Structured JSON logger
docker/
  Dockerfile           – Multi-stage build, trains model at build time
  docker-compose.yml   – App + Prometheus
  prometheus.yml       – Scrape config
load_test/
  locustfile.py        – steady / ramp / overload scenarios
  report_latency.py    – Pull p50/p95/p99 from Prometheus
training/
  train.py             – Generates synthetic data + trains sklearn model
```
