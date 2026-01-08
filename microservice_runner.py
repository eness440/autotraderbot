"""
microservice_runner.py
----------------------

This module provides a simple wrapper for running the bot in a microservice
architecture.  While the main codebase is organised as a monolithic
application, the underlying logic can be decomposed into separate services
(data collection, signal generation, order execution).  This runner
implements a cooperative, in‑memory queue based approach for
demonstration purposes.

The classes ``DataCollector``, ``SignalGenerator`` and ``OrderExecutor``
are imported from ``microservices.py``.  They each expose ``step()``
methods that process tasks from asynchronous queues.  The
``MicroserviceManager`` instantiates these services and runs them in
threads.  Each service communicates via Python lists used as queues.

To enable microservice mode set the environment variable
``USE_MICROSERVICES=1`` before starting ``main_bot_async.py``.  When
enabled, the main loop will start the microservice manager and
periodically feed it with the required inputs.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List

try:
    from microservices import DataCollector, SignalGenerator, OrderExecutor  # type: ignore
except Exception:
    # Define dummy microservices if missing
    class DataCollector:
        def __init__(self, in_queue: List[Any], out_queue: List[Any]):
            self.in_queue = in_queue
            self.out_queue = out_queue
        def step(self):
            # Consume price fetch tasks and produce market data
            if self.in_queue:
                symbol = self.in_queue.pop(0)
                self.out_queue.append({"symbol": symbol, "price": 0.0})
    class SignalGenerator:
        def __init__(self, in_queue: List[Any], out_queue: List[Any]):
            self.in_queue = in_queue
            self.out_queue = out_queue
        def step(self):
            if self.in_queue:
                data = self.in_queue.pop(0)
                self.out_queue.append({"symbol": data["symbol"], "signal": 0})
    class OrderExecutor:
        def __init__(self, in_queue: List[Any]):
            self.in_queue = in_queue
        def step(self):
            if self.in_queue:
                order = self.in_queue.pop(0)
                # In real implementation, send order via ccxt
                print(f"[ORDER] Executed {order}")

logger = logging.getLogger(__name__)


class MicroserviceManager:
    """Manage and run microservices using shared in‑memory queues.

    This class attempts to adapt between the skeleton microservices defined
    in ``microservices.py`` (which expect callables for subscribe/publish)
    and the simple list‑based fallback used for demonstration.  If the
    imported ``DataCollector`` class defines a ``publish`` parameter in its
    ``__init__`` signature, callables are wired up to internal lists.  If
    not, the legacy list‑based interface is used.  This prevents errors
    like ``'list' object is not callable`` when running with the skeleton
    microservices.
    """

    def __init__(self) -> None:
        # Shared queues for passing messages between services
        self.q_price_req: List[Any] = []  # price requests (used only by fallback)
        self.q_price_data: List[Any] = []  # price data from collector → generator
        self.q_orders: List[Any] = []  # signals/orders from generator → executor
        # Attempt to adapt to skeleton microservices that accept callables
        try:
            import inspect
            sig = inspect.signature(DataCollector.__init__)
            param_names = [p.name for p in list(sig.parameters.values())[1:]]  # skip 'self'
            # Skeleton DataCollector declares a 'publish' parameter
            if 'publish' in param_names and 'in_queue' not in param_names:
                # Wire callables to internal queues
                def publish_price(data: Any) -> None:
                    self.q_price_data.append(data)

                def subscribe_price() -> Any:
                    if self.q_price_data:
                        return self.q_price_data.pop(0)
                    return None

                def publish_signal(sig: Any) -> None:
                    self.q_orders.append(sig)

                def subscribe_order() -> Any:
                    if self.q_orders:
                        return self.q_orders.pop(0)
                    return None

                # Instantiate skeleton services with callables
                self.collector = DataCollector(publish_price)
                # SignalGenerator in skeleton expects (subscribe, publish)
                self.generator = SignalGenerator(subscribe_price, publish_signal)
                # OrderExecutor in skeleton expects subscribe only
                self.executor = OrderExecutor(subscribe_order)
            else:
                # Fallback to list‑based interface (imported classes expect queues)
                self.collector = DataCollector(self.q_price_req, self.q_price_data)  # type: ignore
                # In fallback mode, generator writes signals to q_orders directly
                self.generator = SignalGenerator(self.q_price_data, self.q_orders)  # type: ignore
                self.executor = OrderExecutor(self.q_orders)  # type: ignore
        except Exception:
            # In case of inspection failure, assume fallback interface
            self.collector = DataCollector(self.q_price_req, self.q_price_data)  # type: ignore
            self.generator = SignalGenerator(self.q_price_data, self.q_orders)  # type: ignore
            self.executor = OrderExecutor(self.q_orders)  # type: ignore
        self._stop_event = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run_service, args=(self.collector,), daemon=True).start()
        threading.Thread(target=self._run_service, args=(self.generator,), daemon=True).start()
        threading.Thread(target=self._run_service, args=(self.executor,), daemon=True).start()
        logger.info("Microservices started.")

    def stop(self) -> None:
        self._stop_event.set()

    def _run_service(self, service: Any) -> None:
        while not self._stop_event.is_set():
            try:
                service.step()
            except Exception as exc:
                logger.error("Microservice error: %s", exc)
            time.sleep(0.1)

    def add_price_request(self, symbol: str) -> None:
        # Only meaningful for list‑based fallback; skeleton microservices
        # continuously produce data and signals.
        self.q_price_req.append(symbol)

    def add_order(self, order: Dict[str, Any]) -> None:
        self.q_orders.append(order)


def run_microservices() -> MicroserviceManager:
    """Instantiate and start microservices, returning the manager."""
    mgr = MicroserviceManager()
    mgr.start()
    return mgr


if __name__ == "__main__":
    mgr = run_microservices()
    # Example demonstration: periodically request prices and process signals
    symbols = ["BTC/USDT", "ETH/USDT"]
    try:
        while True:
            for s in symbols:
                mgr.add_price_request(s)
            time.sleep(5)
    except KeyboardInterrupt:
        mgr.stop()