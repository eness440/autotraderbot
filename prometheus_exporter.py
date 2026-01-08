"""
prometheus_exporter.py
----------------------

Minimal Prometheus exporter for the AutoTraderBot.  Exposes key
runtime and API metrics over HTTP in the Prometheus exposition format.
By default binds to port 9100; override via the ``PROMETHEUS_PORT``
environment variable.  The exporter reads from ``metrics/runtime_status.json``
and ``metrics/metrics.json``.  If these files are absent, defaults are
used.

This script is intentionally light‑weight and does not depend on the
``prometheus_client`` library.  To run it as a background service::

    python prometheus_exporter.py

Metrics exposed:

- ``autotrader_open_trades``: number of open positions.
- ``autotrader_daily_pnl``: current realised PnL for the day.
- ``autotrader_kill_switch``: 1 if kill switch is ON, 0 otherwise.
- ``autotrader_api_calls_total``: cumulative API call count.
- ``autotrader_api_errors_total``: cumulative API error count.
- ``autotrader_api_retries_total``: cumulative API retry count.
- ``autotrader_total_latency_seconds``: cumulative API latency.

You can extend ``collect_metrics`` to include more fields (e.g.
provider status) as needed.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List


def collect_metrics() -> List[str]:
    """
    Collect metrics from JSON files and return them as lines in
    Prometheus exposition format.  Missing values default to 0.
    """
    metrics_lines: List[str] = []
    # Runtime status
    status_path = Path("metrics/runtime_status.json")
    status = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}
    open_trades = status.get("open_trades") or 0
    daily_pnl = status.get("daily_pnl") or 0
    kill_switch = status.get("kill_switch") or "OFF"
    kill_val = 1 if str(kill_switch).upper() == "ON" else 0
    metrics_lines.append(f"autotrader_open_trades {open_trades}\n")
    metrics_lines.append(f"autotrader_daily_pnl {daily_pnl}\n")
    metrics_lines.append(f"autotrader_kill_switch {kill_val}\n")
    # API metrics
    mm_path = Path("metrics/metrics.json")
    mm = {}
    if mm_path.exists():
        try:
            mm = json.loads(mm_path.read_text(encoding="utf-8"))
        except Exception:
            mm = {}
    api_calls = mm.get("api_calls") or 0
    api_errors = mm.get("api_errors") or 0
    api_retries = mm.get("api_retries") or 0
    total_latency = mm.get("total_latency") or 0
    metrics_lines.append(f"autotrader_api_calls_total {api_calls}\n")
    metrics_lines.append(f"autotrader_api_errors_total {api_errors}\n")
    metrics_lines.append(f"autotrader_api_retries_total {api_retries}\n")
    metrics_lines.append(f"autotrader_total_latency_seconds {total_latency}\n")
    return metrics_lines


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # type: ignore[override]
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return
        body = "".join(collect_metrics()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server() -> None:
    port = int(os.getenv("PROMETHEUS_PORT", "9100"))
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"[prometheus_exporter] Serving metrics on port {port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[prometheus_exporter] Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()