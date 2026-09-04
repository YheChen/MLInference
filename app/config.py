"""Tuning knobs.

Every value can be overridden with an environment variable of the same name;
see .env.example for the full list and the defaults documented below.
"""

import os
from pathlib import Path

# Repo root, so the service starts correctly from any working directory
# rather than only from the project root.
_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


MODEL_PATH = os.getenv("MODEL_PATH") or str(_ROOT / "app" / "model" / "model.pkl")

# Number of features the trained model expects. Requests are validated against
# this at the API boundary so malformed input can never reach the batcher.
FEATURE_DIM = _env_int("FEATURE_DIM", 10)

BATCH_MAX_SIZE = _env_int("BATCH_MAX_SIZE", 32)
BATCH_WINDOW_MS = _env_int("BATCH_WINDOW_MS", 5)

QUEUE_MAX_SIZE = _env_int("QUEUE_MAX_SIZE", 2000)
QUEUE_HIGH_WATERMARK_RATIO = _env_float("QUEUE_HIGH_WATERMARK_RATIO", 0.8)
QUEUE_HIGH_WATERMARK = int(QUEUE_MAX_SIZE * QUEUE_HIGH_WATERMARK_RATIO)

REQUEST_TIMEOUT_MS = _env_int("REQUEST_TIMEOUT_MS", 100)  # hard timeout
