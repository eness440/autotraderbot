"""
watchdog.py
-----------

This module implements a simple fail-safe watchdog that monitors system
resources (CPU and memory usage) and environment integrity.  When usage
exceeds configured thresholds or suspicious conditions (e.g. missing API
keys) are detected, the watchdog triggers a callback which can pause the
bot, notify an operator, or take other corrective action.

The watchdog is designed to run in a separate thread or process and
periodically sample the system.  It can be integrated into the bot by
starting it at initialisation and registering appropriate callbacks.

Example usage::

    from watchdog import Watchdog
    def on_alert(msg):
        print("Watchdog alert:", msg)
    wd = Watchdog(memory_threshold=0.9, cpu_threshold=0.9, interval=5, callback=on_alert)
    wd.start()
    # ... run bot ...
    wd.stop()

"""

from __future__ import annotations

import os
import threading
import time
import logging
from typing import Callable, Optional

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # pragma: no cover

logger = logging.getLogger(__name__)


class Watchdog:
    """Monitors resource usage, environment integrity and connectivity.

    The watchdog periodically samples system statistics, checks for missing
    environment variables and optionally verifies network connectivity by
    pinging a URL.  When a threshold is exceeded or an anomaly is detected
    (such as missing API keys), the registered callback is invoked with a
    descriptive message.
    """

    def __init__(
        self,
        memory_threshold: float = 0.9,
        cpu_threshold: float = 0.9,
        interval: float = 10.0,
        callback: Optional[Callable[[str], None]] = None,
        env_keys: Optional[list[str]] = None,
        ping_url: Optional[str] = None,
    ) -> None:
        """Initialise the watchdog.

        :param memory_threshold: fraction of total memory usage above which to alert
        :param cpu_threshold: fraction of total CPU usage above which to alert
        :param interval: interval in seconds between checks
        :param callback: function to call with an alert message when a threshold is breached
        :param env_keys: list of environment variable names that must be present; missing keys trigger alerts
        :param ping_url: optional URL to test network connectivity; failure triggers an alert
        """
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.interval = interval
        self.callback = callback
        self.env_keys = env_keys or []
        self.ping_url = ping_url
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _check(self) -> None:
        while not self._stop_event.is_set():
            alerts = []
            # Check environment variables
            for key in self.env_keys:
                if os.getenv(key) is None:
                    alerts.append(f"Missing required env var: {key}")
            # Check network connectivity
            if self.ping_url:
                try:
                    # Use urllib to avoid external dependencies
                    import urllib.request

                    with urllib.request.urlopen(self.ping_url, timeout=5) as resp:
                        if resp.status >= 400:
                            alerts.append(f"Ping URL returned status {resp.status}")
                except Exception as exc:
                    alerts.append(f"Network connectivity check failed: {exc}")
            if psutil is not None:
                try:
                    mem = psutil.virtual_memory()
                    mem_ratio = mem.percent / 100.0
                    if mem_ratio > self.memory_threshold:
                        alerts.append(f"High memory usage: {mem_ratio:.2%}")
                    # Measure CPU usage over a one‑second interval to avoid spurious peaks
                    cpu_ratio = psutil.cpu_percent(interval=1.0) / 100.0
                    if cpu_ratio > self.cpu_threshold:
                        alerts.append(f"High CPU usage: {cpu_ratio:.2%}")
                except Exception as exc:
                    logger.debug("psutil error: %s", exc)
            else:
                # psutil not available; use os.getloadavg as fallback on Unix
                try:
                    load1, _, _ = os.getloadavg()
                    cpu_ratio = load1 / os.cpu_count()
                    if cpu_ratio > self.cpu_threshold:
                        alerts.append(f"High load average: {cpu_ratio:.2%}")
                except Exception:
                    pass
            if alerts and self.callback:
                for msg in alerts:
                    try:
                        self.callback(msg)
                    except Exception as exc:
                        logger.warning("Watchdog callback error: %s", exc)
            time.sleep(self.interval)

    def start(self) -> None:
        """Start monitoring in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._check, name="WatchdogThread", daemon=True)
        self._thread.start()
        logger.info("Watchdog started")

    def stop(self) -> None:
        """Stop monitoring and join the thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("Watchdog stopped")