"""
Locust load-test scenarios for the ML Inference Service.

Usage (headless examples):
  # Steady-state:  low concurrency, verifies baseline latency
  SCENARIO=steady locust -f load_test/locustfile.py --headless -u 50 -r 10 -t 2m

  # Ramp:  linear ramp from 0 → 300 users over 3 min, then hold
  SCENARIO=ramp locust -f load_test/locustfile.py --headless -t 5m

  # Overload:  push well past queue capacity to exercise backpressure / shedding
  SCENARIO=overload locust -f load_test/locustfile.py --headless -u 1000 -r 100 -t 2m

Environment variables:
  SCENARIO          steady | ramp | overload  (default: steady)
  FEATURE_DIM       Number of features per request (default: 10)
  TARGET_HOST       Base URL (default: http://localhost:8000)
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import List

from locust import HttpUser, LoadTestShape, between, constant_pacing, task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

SCENARIO: str = os.getenv("SCENARIO", "steady").strip().lower()

FEATURE_DIM = _env_int("FEATURE_DIM", 10)
FEATURE_MIN = _env_float("FEATURE_MIN", -2.0)
FEATURE_MAX = _env_float("FEATURE_MAX", 2.0)
REQUEST_TIMEOUT_S = _env_float("REQUEST_TIMEOUT_SECONDS", 2.0)

_rng = random.Random(0)


def _random_features(dim: int = FEATURE_DIM) -> List[float]:
    span = FEATURE_MAX - FEATURE_MIN
    return [FEATURE_MIN + _rng.random() * span for _ in range(dim)]


# ---------------------------------------------------------------------------
# Scenario-specific parameters
# ---------------------------------------------------------------------------

_SCENARIO_PARAMS = {
    #                wait_min  wait_max  allow_503
    "steady":       (0.05,     0.15,     False),
    "ramp":         (0.02,     0.08,     True),
    "overload":     (0.001,    0.005,    True),
}

_wait_min, _wait_max, _allow_503 = _SCENARIO_PARAMS.get(
    SCENARIO, _SCENARIO_PARAMS["steady"]
)


# ---------------------------------------------------------------------------
# User class
# ---------------------------------------------------------------------------

class InferenceUser(HttpUser):
    """Simulates a client calling /predict and /health."""

    host = os.getenv("TARGET_HOST", "http://localhost:8000")
    wait_time = between(_wait_min, _wait_max)

    @task(20)
    def predict(self) -> None:
        payload = json.dumps({"features": _random_features()})
        with self.client.post(
            "/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_S,
            name="/predict",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503 and _allow_503:
                # Expected under overload / ramp
                resp.success()
            else:
                resp.failure(f"status {resp.status_code}")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health", timeout=REQUEST_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Ramp LoadTestShape  (only active when SCENARIO=ramp)
# ---------------------------------------------------------------------------

class RampShape(LoadTestShape):
    """
    Linear ramp:  0 → peak_users over ramp_duration, then hold for hold_duration.
    Ignored unless SCENARIO=ramp.
    """

    peak_users = _env_int("RAMP_PEAK_USERS", 300)
    ramp_duration = _env_int("RAMP_DURATION_S", 180)    # 3 min ramp
    hold_duration = _env_int("RAMP_HOLD_S", 120)        # 2 min hold
    spawn_rate = _env_int("RAMP_SPAWN_RATE", 10)

    use_common_options = True   # respect -t flag as overall cap

    def tick(self):
        if SCENARIO != "ramp":
            # Returning None on the first tick disables the shape entirely
            # and lets Locust fall back to the default -u / -r behaviour.
            return None

        run_time = self.get_run_time()
        total = self.ramp_duration + self.hold_duration

        if run_time > total:
            return None  # stop

        if run_time < self.ramp_duration:
            users = max(1, math.ceil(self.peak_users * run_time / self.ramp_duration))
        else:
            users = self.peak_users

        return users, self.spawn_rate
