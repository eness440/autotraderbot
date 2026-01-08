"""
microservices.py
----------------

This module outlines a microservice architecture for the trading bot.
Although the current codebase is monolithic, migrating to a microservice
structure can improve scalability and reliability.  Each service is
responsible for a distinct aspect of the trading pipeline and communicates
via message queues or asynchronous events.  The classes below are
skeletons; real implementations would include networking code, state
management and robust error handling.

Services:

* ``DataCollector``: subscribes to market data streams (WebSocket) and
  publishes cleaned price and order-book data to a message queue.
* ``SignalGenerator``: consumes data from the queue, runs technical
  indicators, sentiment models and AI predictors to generate trading
  signals, then publishes these to a queue.
* ``OrderExecutor``: listens for signals and executes them on the
  exchange via CCXT or native APIs.  Applies risk management rules.
* ``ModelTrainer``: periodically retrains models (BiLSTM, RL, etc.)
  using recent data and publishes updated weights to a model registry.

This module is not used directly by the current bot but serves as a
starting point for developers looking to refactor the system into
microservices.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class BaseService(threading.Thread):
    """Base class for background services in a microservice architecture."""

    def __init__(self, name: str, interval: float = 1.0) -> None:
        super().__init__(name=name, daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        while not self.stopped():
            try:
                self.step()
            except Exception as exc:
                logger.warning("Service %s encountered error: %s", self.name, exc)
            time.sleep(self.interval)

    def step(self) -> None:
        raise NotImplementedError


class DataCollector(BaseService):
    """Collects real‑time market data and publishes to a queue.

    This implementation attempts to subscribe to WebSocket price streams
    if the optional ``websockets`` dependency is available.  Symbols are
    read from the environment variable ``MS_SYMBOLS`` (comma‑separated)
    or default to ``BTCUSDT``.  On each ``step()`` the collector fetches
    one update from the asynchronous price stream and emits a dict
    containing the symbol and price.  If a WebSocket connection cannot
    be established (for example because the ``websockets`` library is
    missing or network access is unavailable), the collector falls back
    to a simple counter‑based price value using the system clock.  This
    design keeps the service operational in constrained environments.
    """

    def __init__(self, publish: Callable[[Any], None], interval: float = 1.0) -> None:
        super().__init__(name="DataCollector", interval=interval)
        self.publish = publish
        # Read comma‑separated symbols from environment; default to BTCUSDT
        import os
        syms_env = os.getenv("MS_SYMBOLS", "BTCUSDT")
        # Normalise symbols: remove spaces and convert to lower case (Binance format)
        self.symbols = [s.strip().lower() for s in syms_env.split(",") if s.strip()]
        if not self.symbols:
            self.symbols = ["btcusdt"]
        # Prepare an async generator for WebSocket price stream.  This will
        # be initialised lazily upon the first call to ``step()``.
        self._price_gen = None  # type: Any
        # Keep track of the last known price for each symbol.  If all API
        # requests fail, the most recent valid price will be re‑used instead
        # of generating dummy data.  This prevents the fallback from
        # producing monotonically increasing pseudo‑prices.
        self._last_prices: dict[str, float] = {sym: None for sym in self.symbols}

    def _init_stream(self) -> None:
        """Initialise the asynchronous price stream generator lazily.

        If the ``websocket_data_provider.stream_prices`` function is
        available and the ``websockets`` library is installed, creates
        an async generator.  Otherwise leaves ``self._price_gen`` as
        ``None``, indicating that the service should use its fallback
        mechanism.
        """
        if self._price_gen is not None:
            return
        try:
            from websocket_data_provider import stream_prices  # type: ignore
            import importlib
            # Check if websockets library is available by importing
            importlib.import_module("websockets")
            # Convert symbols to Binance lower‑case ticker format
            self._price_gen = stream_prices(self.symbols)
        except Exception:
            # WebSocket streaming not available
            self._price_gen = None

    def _get_next_price(self) -> Optional[Any]:
        """Synchronously fetch the next price update from the async generator.

        Returns a dict of the form ``{"symbol": <str>, "price": <float>}``
        if an update is available, or ``None`` if streaming is unavailable.
        """
        self._init_stream()
        if self._price_gen is None:
            return None
        try:
            import asyncio
            # Create a new temporary event loop for the async generator
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Retrieve the next (symbol, price) tuple
                sym, price = loop.run_until_complete(self._price_gen.__anext__())
                return {"symbol": sym, "price": float(price)}
            except StopAsyncIteration:
                # Stream ended; reset generator to None
                self._price_gen = None
                return None
            finally:
                loop.close()
        except Exception:
            return None

    def step(self) -> None:
        # Attempt to fetch real‑time price from WebSocket stream
        data = self._get_next_price()
        if data is None:
            # No WebSocket data available.  Try to fetch the latest price
            # using the authenticated Binance API via ccxt.  If API keys
            # are not configured or ccxt is unavailable, fall back to a
            # public REST endpoint.  As a final resort, reuse the last
            # known price instead of generating monotonic dummy values.
            import os
            symbol = self.symbols[0]
            symbol_norm = symbol.replace("/", "").upper()
            fetched_price: Optional[float] = None
            # Attempt to fetch price via ccxt using API credentials
            try:
                import ccxt  # type: ignore
                api_key = os.getenv("BINANCE_API_KEY")
                # Support both BINANCE_API_SECRET and BINANCE_SECRET_KEY for backward compatibility.
                secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET_KEY")
                if api_key and secret:
                    exchange = ccxt.binance({
                        'apiKey': api_key,
                        'secret': secret,
                        'enableRateLimit': True,
                    })
                    ticker = exchange.fetch_ticker(symbol_norm)
                    price_val = ticker.get('last')
                    if isinstance(price_val, (int, float)) and price_val > 0:
                        fetched_price = float(price_val)
            except Exception:
                fetched_price = None
            # If ccxt failed, try Binance public REST API
            if fetched_price is None:
                base_url = os.getenv("PRICE_FETCH_BASE_URL", "https://api.binance.com/api/v3")
                endpoint = f"{base_url}/ticker/price?symbol={symbol_norm}"
                try:
                    import requests  # type: ignore
                    resp = requests.get(endpoint, timeout=5)
                    if resp.status_code == 200:
                        price_str = resp.json().get("price")
                        if price_str is not None:
                            fetched_price = float(price_str)
                except Exception:
                    fetched_price = None
            # If all external fetches failed, use the last known price
            if fetched_price is None:
                last_price = self._last_prices.get(symbol)
                if last_price is not None:
                    fetched_price = float(last_price)
            if fetched_price is not None:
                data = {"symbol": symbol, "price": fetched_price}
        # If a valid price was obtained (either from WebSocket, REST or last price),
        # publish it and update last price cache
        if data is not None:
            try:
                self._last_prices[data["symbol"]] = float(data["price"])
            except Exception:
                pass
            try:
                self.publish(data)
            except Exception as exc:
                logger.warning("DataCollector publish failed: %s", exc)


class SignalGenerator(BaseService):
    """Generates trading signals from incoming price data.

    This service subscribes to price updates from the ``DataCollector``
    via the provided ``subscribe`` callable.  It maintains a short
    history of recent prices for each symbol and computes a simple
    Relative Strength Index (RSI) to detect overbought/oversold
    conditions.  RSI values below a configurable buy threshold trigger
    a ``buy`` signal; values above a sell threshold trigger a ``sell``
    signal; otherwise the action is ``hold``.  The computed signal and
    an associated confidence score are published via the ``publish``
    callable.
    """

    def __init__(self, subscribe: Callable[[], Optional[Any]], publish: Callable[[Any], None], interval: float = 1.0) -> None:
        super().__init__(name="SignalGenerator", interval=interval)
        self.subscribe = subscribe
        self.publish = publish
        # Price history per symbol for RSI calculation
        self.history: dict[str, list[float]] = {}
        # Thresholds can be overridden via environment variables
        import os
        try:
            self.buy_threshold = float(os.getenv("MS_RSI_BUY", "30.0"))
            self.sell_threshold = float(os.getenv("MS_RSI_SELL", "70.0"))
        except Exception:
            self.buy_threshold = 30.0
            self.sell_threshold = 70.0
        # Read moving average window lengths for the MA crossover strategy.
        # Short and long windows are configurable via environment variables
        # ``MS_MA_SHORT`` and ``MS_MA_LONG``.  Defaults are 5 and 15,
        # respectively.  If values are invalid or reversed (long < short),
        # they will be corrected to sensible defaults.  These values are
        # used to compute simple moving averages in ``step``.
        try:
            self.ma_short = int(os.getenv("MS_MA_SHORT", "5"))
        except Exception:
            self.ma_short = 5
        try:
            self.ma_long = int(os.getenv("MS_MA_LONG", "15"))
        except Exception:
            self.ma_long = 15
        # Ensure long window is larger than short window
        if self.ma_long <= self.ma_short:
            self.ma_long = max(self.ma_short + 1, 15)

        # Momentum ayarları: kısa vadeli momentum sinyali için geçmiş bakış
        # süresi ve momentum eşiği. Bu değerler ortam değişkenlerinden
        # okunabilir. Eğer uygun değerler sağlanmazsa varsayılan olarak
        # 5 periyotluk bakış ve %1 momentum eşiği kullanılır.
        try:
            self.mom_lookback = int(os.getenv("MS_MOMENTUM_LOOKBACK", "5"))
        except Exception:
            self.mom_lookback = 5
        if self.mom_lookback < 1:
            self.mom_lookback = 5
        try:
            self.mom_threshold = float(os.getenv("MS_MOMENTUM_THRESHOLD", "0.01"))
        except Exception:
            self.mom_threshold = 0.01

        # ---------------------------
        # Volatilite rejimi ayarları
        # ---------------------------
        # Yüksek volatilite (trend piyasası) ve düşük volatilite
        # (yataya yakın piyasa) durumlarını tespit etmek için geçmiş
        # fiyat verilerinin standart sapması kullanılır.  Bu değerler
        # ortam değişkenlerinden okunur ve varsayılan bakış 10 periyot,
        # eşik 0.02 (yani %2) olarak ayarlanır.  vol_conf_factor ise
        # volatiliteye göre güven skoruna uygulanacak doğrusal katsayıdır.
        try:
            self.vol_lookback = int(os.getenv("MS_VOL_LOOKBACK", "10"))
        except Exception:
            self.vol_lookback = 10
        if self.vol_lookback < 2:
            self.vol_lookback = 10
        try:
            self.vol_threshold = float(os.getenv("MS_VOL_THRESHOLD", "0.02"))
        except Exception:
            self.vol_threshold = 0.02
        try:
            self.vol_conf_factor = float(os.getenv("MS_VOL_CONF_FACTOR", "0.15"))
        except Exception:
            self.vol_conf_factor = 0.15
        # Sınırlar: 0.0 <= factor <= 0.5
        if self.vol_conf_factor < 0.0:
            self.vol_conf_factor = 0.0
        if self.vol_conf_factor > 0.5:
            self.vol_conf_factor = 0.5

    @staticmethod
    def _compute_rsi(prices: list[float]) -> float:
        """Compute a simple RSI from a list of prices.

        This implementation calculates RSI using the average gain and
        loss over the last 14 periods.  If there are fewer than 15
        prices, returns 50.0 (neutral).  The RSI is returned on the
        standard 0–100 scale.
        """
        n = 14
        if len(prices) < n + 1:
            return 50.0
        gains = []
        losses = []
        for i in range(-n, 0):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                gains.append(diff)
            else:
                losses.append(abs(diff))
        avg_gain = sum(gains) / n if gains else 0.0
        avg_loss = sum(losses) / n if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def step(self) -> None:
        # Retrieve latest market data from collector
        data = self.subscribe()
        if not data or not isinstance(data, dict):
            return
        symbol = data.get("symbol")
        price = data.get("price")
        if not isinstance(symbol, str) or not isinstance(price, (int, float)):
            return
        # Maintain price history for the symbol
        hist = self.history.setdefault(symbol, [])
        hist.append(float(price))
        # Keep only the last 50 prices to bound memory
        if len(hist) > 50:
            self.history[symbol] = hist[-50:]
        # Compute RSI
        rsi = self._compute_rsi(hist)
        # Compute simple moving averages for crossover if enough data
        short_ma = None  # type: Optional[float]
        long_ma = None  # type: Optional[float]
        try:
            if len(hist) >= self.ma_short:
                short_ma = sum(hist[-self.ma_short:]) / float(self.ma_short)
            if len(hist) >= self.ma_long:
                long_ma = sum(hist[-self.ma_long:]) / float(self.ma_long)
        except Exception:
            short_ma = None
            long_ma = None
        # Determine primary signal using MA crossover if both MAs available
        action = "hold"
        ma_signal = None
        if short_ma is not None and long_ma is not None:
            if short_ma > long_ma:
                ma_signal = "buy"
            elif short_ma < long_ma:
                ma_signal = "sell"
        # Determine RSI‑based signal
        rsi_signal = None
        if rsi <= self.buy_threshold:
            rsi_signal = "buy"
        elif rsi >= self.sell_threshold:
            rsi_signal = "sell"
        else:
            rsi_signal = "hold"
        # Combine MA and RSI signals.  Prefer MA crossover if available; if
        # both signals exist and agree, boost confidence.  If they conflict,
        # fall back to RSI to avoid false crosses.
        if ma_signal is not None:
            if rsi_signal == "hold":
                action = ma_signal
            elif ma_signal == rsi_signal:
                action = ma_signal
            else:
                # If conflict, choose RSI signal but mark confidence lower
                action = rsi_signal
        else:
            action = rsi_signal
        # --- Momentum sinyali ---
        # Fiyat serisinin kısa bir süre içindeki yüzdesel değişimi momentum
        # ölçüsü olarak kullanılır. Momentum eşikleri ortam değişkenlerinden
        # okunur ve ``self.mom_threshold`` ile karşılaştırılır. Bakış
        # süresi ``self.mom_lookback`` tanımlıdır.
        momentum_signal: Optional[str] = None
        mom_value: float = 0.0
        if len(hist) >= self.mom_lookback:
            try:
                mom_value = (hist[-1] - hist[-self.mom_lookback]) / float(hist[-self.mom_lookback])
                if mom_value > self.mom_threshold:
                    momentum_signal = "buy"
                elif mom_value < -self.mom_threshold:
                    momentum_signal = "sell"
                else:
                    momentum_signal = "hold"
            except Exception:
                momentum_signal = None
        # --- Sinyallerin birleştirilmesi ---
        # Varsayılan aksiyon önceki RSI/MA birleşimiyle belirlenen ``action``
        final_action = action
        # Toplanan sinyaller listesi (hold olmayanlar)
        signals_list: list[str] = []
        if momentum_signal and momentum_signal != "hold":
            signals_list.append(momentum_signal)
        if ma_signal:
            signals_list.append(ma_signal)
        if rsi_signal and rsi_signal != "hold":
            signals_list.append(rsi_signal)
        if signals_list:
            buys = signals_list.count("buy")
            sells = signals_list.count("sell")
            if buys > sells:
                final_action = "buy"
            elif sells > buys:
                final_action = "sell"
            else:
                final_action = action  # eşitlik durumunda orijinal karara sadık kal
        else:
            final_action = action
        # --- Güven skoru hesapla ---
        # RSI ve MA tabanlı güven skorları
        conf_rsi = abs(rsi - 50.0) / 50.0
        conf_ma = 0.0
        try:
            if short_ma is not None and long_ma is not None and long_ma != 0.0:
                conf_ma = abs(short_ma - long_ma) / abs(long_ma)
        except Exception:
            conf_ma = 0.0
        base_confidence: float = 0.0
        if ma_signal is not None and rsi_signal == ma_signal:
            base_confidence = min(1.0, (conf_rsi + conf_ma) / 2.0 * 1.5)
        else:
            base_confidence = max(conf_rsi, conf_ma)
        # Momentum güveni: momentumun eşik ile göreceli büyüklüğü
        conf_mom = 0.0
        try:
            conf_mom = abs(mom_value) / max(self.mom_threshold, 1e-9)
            if conf_mom > 1.0:
                conf_mom = 1.0
        except Exception:
            conf_mom = 0.0

        # -----------------------------
        # Nihai güven skoru hesaplama
        # -----------------------------
        # Temel güven skoru RSI/MA ve momentum ortalamasının sınırlı
        # değeridir.  Daha sonra volatilite rejimine göre bir çarpan
        # uygulanacaktır.
        confidence = min(1.0, (base_confidence + conf_mom) / 2.0)

        # Volatilite rejimi analizi: son ``vol_lookback`` periyottaki
        # yüzdesel getirilerin standart sapmasını hesapla.  Bu değer
        # ``self.vol_threshold`` eşiğinin üzerindeyse piyasa trend
        # halinde kabul edilir ve güven skoru artırılır; aksi halde
        # yavaş piyasa olarak değerlendirilir ve güven skoru azaltılır.
        trending_flag = None  # type: Optional[bool]
        try:
            if len(hist) >= self.vol_lookback + 1:
                import statistics
                rets: list[float] = []
                # Hesaplama: (P_t - P_{t-1}) / P_{t-1}
                for i in range(-self.vol_lookback, 0):
                    prev_idx = i - 1
                    prev_price = hist[prev_idx]
                    curr_price = hist[i]
                    if prev_price != 0:
                        rets.append((curr_price - prev_price) / prev_price)
                if rets:
                    vol_std = statistics.stdev(rets) if len(rets) > 1 else abs(rets[0])
                    trending_flag = vol_std >= self.vol_threshold
        except Exception:
            trending_flag = None
        # Güveni volatiliteye göre ayarla
        try:
            if trending_flag is not None:
                if trending_flag:
                    # Trend piyasası: güveni artır
                    confidence *= (1.0 + self.vol_conf_factor)
                else:
                    # Yatay piyasa: güveni düşür
                    confidence *= max(0.0, (1.0 - self.vol_conf_factor))
                if confidence > 1.0:
                    confidence = 1.0
        except Exception:
            pass
        # --- Payload oluştur ve yayınla ---
        payload = {
            "symbol": symbol,
            "action": final_action,
            "confidence": round(confidence, 3),
            "rsi": round(rsi, 2),
        }
        if short_ma is not None:
            payload["short_ma"] = round(short_ma, 4)
        if long_ma is not None:
            payload["long_ma"] = round(long_ma, 4)
        if momentum_signal:
            payload["momentum"] = round(mom_value, 4)
        # Volatilite rejimi ve momentum bilgilerini ekle
        if trending_flag is not None:
            payload["vol_trending"] = bool(trending_flag)
        try:
            self.publish(payload)
        except Exception as exc:
            logger.warning("SignalGenerator publish failed: %s", exc)


class OrderExecutor(BaseService):
    """Executes trading signals by placing orders on the exchange.

    This service listens for signals (dicts with keys ``symbol`` and
    ``action``) from the preceding ``SignalGenerator``.  For each
    received signal it will either place an entry order (for ``buy``)
    or exit order (for ``sell``).  If ``ENABLE_ORDER_EXECUTION`` is
    unset or ``0`` in the environment, the service operates in dry‑run
    mode and only logs what it would do.  Real order placement is
    delegated to the ``safe_submit_entry_plan`` and
    ``safe_submit_exit_plan`` functions from ``safe_order_wrapper``.
    """

    def __init__(self, subscribe: Callable[[], Optional[Any]], interval: float = 1.0) -> None:
        super().__init__(name="OrderExecutor", interval=interval)
        self.subscribe = subscribe
        # Determine whether to execute orders or just log them
        import os
        val = os.getenv("ENABLE_ORDER_EXECUTION", "0").strip().lower()
        self._execute = val in ("1", "true", "yes")
        # Lazy import of safe order functions to avoid heavy dependencies
        self._entry_fn = None
        self._exit_fn = None

    def _load_order_funcs(self) -> None:
        """Import order execution functions on demand."""
        if self._entry_fn is None or self._exit_fn is None:
            try:
                from safe_order_wrapper import safe_submit_entry_plan, safe_submit_exit_plan  # type: ignore
                self._entry_fn = safe_submit_entry_plan
                self._exit_fn = safe_submit_exit_plan
            except Exception as exc:
                logger.warning("safe_order_wrapper import failed: %s", exc)
                self._entry_fn = None
                self._exit_fn = None

    def step(self) -> None:
        signal = self.subscribe()
        if not signal or not isinstance(signal, dict):
            return
        symbol = signal.get("symbol")
        action = signal.get("action")
        confidence = signal.get("confidence")
        if not symbol or not action:
            return
        # Format log message
        msg = f"Signal received for {symbol}: action={action} confidence={confidence}"
        if not self._execute:
            logger.info("[DRY] %s", msg)
            return
        # Attempt to execute order
        self._load_order_funcs()
        if self._entry_fn is None or self._exit_fn is None:
            logger.info("[DRY] %s (order functions unavailable)", msg)
            return
        try:
            if action == "buy":
                # For demonstration we send a fixed notional size (e.g. 0.001 BTC)
                order = self._entry_fn(symbol, 0.001)  # type: ignore[arg-type]
                logger.info("Executed entry order for %s: %s", symbol, order)
            elif action == "sell":
                order = self._exit_fn(symbol)  # type: ignore[arg-type]
                logger.info("Executed exit order for %s: %s", symbol, order)
            else:
                # hold - do nothing
                logger.debug("No action for %s (hold)", symbol)
        except Exception as exc:
            logger.error("Order execution failed for %s: %s", symbol, exc)


class ModelTrainer(BaseService):
    """Periodically retrains AI models and publishes new weights."""

    def __init__(self, publish: Callable[[str, Any], None], interval: float = 3600.0) -> None:
        super().__init__(name="ModelTrainer", interval=interval)
        self.publish = publish

    def step(self) -> None:
        # TODO: retrain models (BiLSTM, RL) and publish updated weights
        weights = {"bilstm": "new_weights"}
        self.publish("bilstm", weights)