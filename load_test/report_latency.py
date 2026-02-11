#!/usr/bin/env python3
"""
Fetch latency percentiles from a running Prometheus instance and print
a Markdown-friendly benchmark table.

Usage:
    python load_test/report_latency.py                     # defaults
    python load_test/report_latency.py --prom http://localhost:9090 --range 5m
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Dict, Optional

DEFAULT_PROM = "http://localhost:9090"
DEFAULT_RANGE = "5m"

METRIC = "inference_request_latency_seconds"

QUANTILES = [0.50, 0.95, 0.99]


def _query(prom_url: str, expr: str) -> Optional[float]:
    """Execute an instant PromQL query and return the scalar value."""
    url = f"{prom_url}/api/v1/query?query={urllib.request.quote(expr)}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"  ⚠  query failed: {exc}", file=sys.stderr)
        return None

    results = data.get("data", {}).get("result", [])
    if not results:
        return None
    return float(results[0]["value"][1])


def fetch_percentiles(prom_url: str, range_window: str) -> Dict[str, Optional[float]]:
    """Return a dict like {'p50': 0.012, 'p95': 0.038, 'p99': 0.072}."""
    out: Dict[str, Optional[float]] = {}
    for q in QUANTILES:
        label = f"p{int(q * 100)}"
        expr = f'histogram_quantile({q}, rate({METRIC}_bucket[{range_window}]))'
        val = _query(prom_url, expr)
        out[label] = val
    return out


def _fmt(val: Optional[float]) -> str:
    if val is None:
        return "n/a"
    return f"{val * 1000:.1f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch latency percentiles from Prometheus")
    parser.add_argument("--prom", default=DEFAULT_PROM, help="Prometheus base URL")
    parser.add_argument("--range", default=DEFAULT_RANGE, dest="range_window", help="PromQL rate window")
    args = parser.parse_args()

    pcts = fetch_percentiles(args.prom, args.range_window)

    total_expr = f"sum(rate({METRIC}_count[{args.range_window}]))"
    rps = _query(args.prom, total_expr)

    print()
    print("## Latency Report")
    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    for label, val in pcts.items():
        print(f"| {label} | {_fmt(val)} |")
    if rps is not None:
        print(f"| throughput | {rps:.0f} req/s |")
    print()

    # SLA check
    p95 = pcts.get("p95")
    if p95 is not None:
        if p95 <= 0.050:
            print("✅  p95 ≤ 50 ms — SLA met")
        else:
            print(f"❌  p95 = {p95*1000:.1f} ms — exceeds 50 ms SLA target")
    print()


if __name__ == "__main__":
    main()
