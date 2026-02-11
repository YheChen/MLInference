"""
Structured JSON logging for the ML Inference Service.

Every log line is a single JSON object so it can be ingested by any log
aggregator (ELK, CloudWatch, etc.) without extra parsing.
"""

from __future__ import annotations

import logging
import json
import sys
import time
import uuid
from typing import Any, Dict

# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach extra fields set via `extra={...}` on the log call
        for key in ("request_id", "method", "path", "status_code",
                     "latency_ms", "queue_depth", "batch_size", "detail"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root application logger."""
    logger = logging.getLogger("ml_inference")
    if logger.handlers:
        return logger  # already configured

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("ml_inference")


def generate_request_id() -> str:
    return uuid.uuid4().hex[:12]
