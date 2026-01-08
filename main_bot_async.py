# -*- coding: utf-8 -*-
"""
main_bot_async.py (async-native + AutoUpdater entegre + BiLSTM/RL füzyon)
- 100+ sembolü eşzamanlı analiz eder (symbol-lock + order-queue)
- Kararları controller_async.decide_batch ile toplu alır
- BiLSTM ve RL Agent tahminlerini (metrics/ai_predictions.json) okuyup güven skoruna entegre eder
- Order planı → güvenli async wrapper ile (DRY-RUN) bağlandı
- AutoUpdater arka planda bakım, risk kalibrasyonu, log temizliği vb. işlemleri yürütür
- 03:00 ve 15:00'te otomatik tetiklenen veri→dataset→RL→BiLSTM eğitim zinciri

GÜNCELLEME:
- Borsa çağrıları (fetch_balance, fetch_positions vb.) call(...) ile sarıldı (rate-limit güvenli).
- Stop-loss hesabı risk_manager.compute_stop_loss(...) ile tek kaynaktan yönetiliyor.
- [FIX] Askıda kalan emir temizliği (Zaman aşımı + Fiyat sapması) eklendi.
- [FIX] Bakiye kontrolü free_balance üzerinden zorunlu kılındı.
"""

import asyncio
import math
from datetime import datetime, timezone

# Prefer explicit local timezone for user-facing timestamps. Europe/Istanbul
# is used as the project default (matches the user's locale).
try:
    from zoneinfo import ZoneInfo  # py3.9+
    _LOCAL_TZ = ZoneInfo("Europe/Istanbul")
except Exception:
    _LOCAL_TZ = None
from collections.abc import Coroutine
import ccxt
import json
import threading
import subprocess
import schedule
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Runtime status reporter (lazy import)
#
# Many modules import this file, so avoid raising on import failure.  At
# runtime, ``_update_runtime_status`` will write key metrics into
# metrics/runtime_status.json and metrics/dashboard.json using atomic
# writes.  If the module cannot be imported (e.g. during unit tests),
# fallback to a no‑op implementation.
try:
    from .runtime_status import update_status as _update_runtime_status  # type: ignore
except Exception:
    def _update_runtime_status(**kwargs):  # type: ignore
        return

# ---------------------------------------------------------------------------
# Load environment variables from a .env file if present.  Using
# ``load_dotenv(find_dotenv())`` searches parent directories to locate
# the nearest .env file.  Wrapping in a try/except keeps the bot
# functional even if python-dotenv is not installed.
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv())
except Exception:
    # If dotenv is not available, environment variables must be provided by the OS
    pass

# ---------------------------------------------------------------------------
# Canary model support
#
# When a new BiLSTM model is trained, the training script publishes a
# ``models/model_update.json`` manifest with a ``canary`` flag and a
# timestamp.  During the canary period, the bot should reduce capital
# allocation for trades that rely on the updated model.  The helper
# function below reads the manifest and returns a multiplier to scale
# risk exposure.  This allows safe hot‑reloading of experimental
# models on live capital.

# Duration (hours) for which the canary risk reduction applies.  After
# this period has elapsed since the model was published, full capital
# allocation is restored.  Adjust as needed.
CANARY_PERIOD_HOURS: float = 12.0

# Fraction of normal wallet allocation to use during the canary period.
# For example, 0.3 means allocate 30% of the usual size.  Values
# should be in (0, 1].
CANARY_RISK_MULTIPLIER: float = 0.3

def _get_canary_multiplier() -> float:
    """Return a risk multiplier based on the canary model state.

    Reads the ``models/model_update.json`` manifest and checks whether
    the current BiLSTM model is marked as ``canary``.  If so and the
    publication timestamp is within ``CANARY_PERIOD_HOURS``, return
    ``CANARY_RISK_MULTIPLIER`` to reduce risk.  Otherwise, return 1.0.

    Returns:
        float: Multiplier in (0, 1] to scale risk; 1.0 means no reduction.
    """
    try:
        # Resolve the path to the manifest relative to this file's parent
        base = Path(__file__).resolve().parent
        manifest_path = base / "models" / "model_update.json"
        if not manifest_path.exists():
            return 1.0
        # Parse the manifest JSON
        txt = manifest_path.read_text(encoding="utf-8")
        data = json.loads(txt)
        # Only apply canary reduction if the manifest has 'canary' true
        if not data.get("canary", False):
            return 1.0
        published_iso = data.get("published_at")
        if not isinstance(published_iso, str):
            return 1.0
        try:
            pub_dt = datetime.fromisoformat(published_iso)
        except Exception:
            return 1.0
        # Compute age in hours
        now = datetime.now(timezone.utc)
        age_sec = (now - pub_dt).total_seconds()
        age_hours = age_sec / 3600.0
        if age_hours <= CANARY_PERIOD_HOURS:
            return CANARY_RISK_MULTIPLIER
        return 1.0
    except Exception:
        return 1.0
# Metrics and circuit breaker integration.  Metrics capture PnL,
# exposure, latency and error counts, while the circuit breaker halts
# new trades under abnormal conditions.  These imports are placed near
# the top to avoid circular dependencies later.
try:
    # Prefer absolute imports so this module can be executed directly
    # (python -m main_bot_async) outside a package context.
    from metrics_manager import update_from_exchange, check_alerts  # type: ignore
except Exception:
    try:
        from .metrics_manager import update_from_exchange, check_alerts  # type: ignore
    except Exception:
        # Fallback no‑op functions if metrics_manager is missing
        async def update_from_exchange(*args, **kwargs):  # type: ignore
            return
        def check_alerts(*args, **kwargs):  # type: ignore
            return
try:
    from circuit_breaker import record_anomaly  # type: ignore
except Exception:
    try:
        from .circuit_breaker import record_anomaly  # type: ignore
    except Exception:
        def record_anomaly(*args, **kwargs):  # type: ignore
            return
import time as time_module
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# -----------------------------------------------------------------------------
# Global config cache
#
# Several helpers in this module consult _CONFIG_DATA before the config is
# loaded later on. Initialise it here to avoid NameError and allow safe
# lookups with defaults.
_CONFIG_DATA: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Anomali Tespiti
#
# Son fiyat ile güncel fiyat arasındaki farkın ATR'nin belirli bir katını
# aşması durumunda işlemi pas geçmek için kullanılır.  Burada, her sembol
# için en son fiyatı tutan global bir sözlük ve anomali kontrol fonksiyonu
# bulunmaktadır.  anomaly_detector modülü bulunamazsa işlevsizlikten dolayı
# hata vermemesi için fallback fonksiyonu tanımlanır.
try:
    from anomaly_detector import is_anomalous_price  # type: ignore
except Exception:
    try:
        from .anomaly_detector import is_anomalous_price  # type: ignore
    except Exception:
        def is_anomalous_price(last_price: float | None, current_price: float | None, atr_value: float | None, threshold: float = 3.0) -> bool:  # type: ignore
            return False

# Her sembol için son analiz edilen fiyatı saklar.  Bu, `_analyze_one`
# fonksiyonunda güncellenir ve `anomaly_detector` ile beraber kullanılır.
LAST_PRICES: Dict[str, float] = {}

# Son 60 fiyat değerini saklamak için bir fiyat geçmişi sözlüğü.  Meta
# strateji seçici için gereklidir.  Her sembol için bir liste tutulur ve
# en fazla ``META_LOOKBACK`` adet son fiyat saklanır.  ``PRICE_HISTORY``
# yalnızca ``_analyze_one`` fonksiyonunda güncellenir ve hiçbir yerde
# sıfırlanmaz; böylece her sembol için kümülatif bir kayda sahip oluruz.
PRICE_HISTORY: Dict[str, list[float]] = {}

# Botun duraklatılıp devam ettirilebilmesi için global bayrak.  Telegram
# komutları ``/pause`` ve ``/resume`` ile değiştirilecektir.  Ticaret
# döngüsü bu bayrağı kontrol ederek işlem yapmayı durdurabilir.
BOT_PAUSED: bool = False
# Records the reason why the bot is currently paused.  When the bot enters
# a paused state (e.g. due to a watchdog alert or user command), this
# variable stores a short description of the trigger.  It is logged each
# time the main loop wakes up so operators can diagnose why the bot is
# sleeping.
PAUSE_REASON: str = ""

#
# Background task management
#
# A common source of resource leaks in asynchronous applications is
# forgetting to cancel background tasks when shutting down.  This can
# lead to lingering tasks that continue to run after the main loop has
# exited, resulting in warnings such as ``"Task was destroyed but it is
# pending!"`` and excessive CPU/memory usage.  To avoid this, we
# maintain a global list of all background tasks created via
# ``create_bg_task``.  When a kill signal is detected or the program
# exits normally, all tasks in this list are cancelled and awaited to
# ensure they have cleaned up correctly.  Exchange WebSocket
# connections are also closed during shutdown.

# List of all background tasks created by this module.  Tasks are
# appended by ``create_bg_task`` and cancelled by ``_cancel_bg_tasks``.
BG_TASKS: list[asyncio.Task] = []

def create_bg_task(coro: Coroutine) -> asyncio.Task:
    """Create and track a background task.

    This helper wraps ``asyncio.create_task`` and appends the created
    task to the global ``BG_TASKS`` list.  All background tasks
    launched via this function will be cancelled when the bot exits or
    when a kill signal is triggered.  If the task raises an exception,
    it will not be automatically rethrown; however, exceptions should
    be handled within the task itself.

    Args:
        coro: The coroutine to schedule as a task.

    Returns:
        The created ``asyncio.Task`` instance.
    """
    task = asyncio.create_task(coro)
    BG_TASKS.append(task)
    return task

async def _cancel_bg_tasks() -> None:
    """Cancel all tracked background tasks and wait for them to finish.

    Iterate over the tasks recorded in ``BG_TASKS``, issue a
    ``cancel()`` request to each one and then ``await`` them
    concurrently.  Any exceptions raised during cancellation are
    suppressed (returned as part of the ``gather`` result) to prevent
    unhandled exceptions during shutdown.  After this function
    completes, ``BG_TASKS`` will be cleared.
    """
    # Make a copy of the list to avoid modifying it while iterating
    tasks = list(BG_TASKS)
    BG_TASKS.clear()
    for t in tasks:
        try:
            t.cancel()
        except Exception:
            pass
    if tasks:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            # Suppress exceptions; tasks may already be cancelled or finished
            pass

# Karaliste dosyası.  Bu dosyadaki semboller aktif sembol listesinden
# çıkarılır.  ``/blacklist`` komutuyla güncellenir.  Dosya JSON formatında
# basit bir liste içerir, örneğin ["BTC/USDT", "DOGE/USDT"].
BLACKLIST_FILE: Path = Path("data") / "blacklist.json"

# Meta strateji seçiminde isteğe bağlı transformer tahmincisi.  Eğer
# modeller/price_transformer.pt mevcutsa, bu transformer yüklenir;
# aksi hâlde ``SimpleTransformerPredictor`` kullanılır.  Bu nesne
# ``meta_strategy_selector.select_strategy`` fonksiyonuna aktarılabilir.
try:
    from transformer_predictor import TransformerPricePredictor  # type: ignore
    _tmp_transformer = TransformerPricePredictor(
        model_path=Path("models") / "price_transformer.pt",
        context_length=30,
        fallback_to_ma=False,
    )
    META_TRANSFORMER = _tmp_transformer if getattr(_tmp_transformer, "is_available", False) else None
except Exception:
    META_TRANSFORMER = None

# Kullanıcı tarafından belirlenebilen risk faktörü.  Varsayılan 1.0'dır ve
# ``/risk`` komutuyla 0–1 arası bir değere ayarlanabilir.  Bu değer,
# kill‑switch tarafından uygulanan risk azaltma faktörü ile çarpılır.
RISK_FACTOR_USER: float = 1.0

# -----------------------------------------------------------------------------
# Pair Trading Cooldown Tracking
#
# To avoid rapid successive pair trades on the same coin pair (which can lead to
# unnecessary churn and slippage), maintain a dictionary of the last time a
# trade was signalled for each pair.  When a new signal arrives for a given
# pair, it will only be considered if at least ``PAIR_TRADE_COOLDOWN_SECONDS``
# seconds have elapsed since the previous signal.  The key is a tuple of the
# two symbols sorted alphabetically, making the lookup order‑independent.
_PAIR_TRADE_LAST_TIME: Dict[Tuple[str, str], float] = {}
# Cooldown interval (in seconds) for pair trading signals.  Default is 1800
# seconds (30 minutes).  This can be tuned to allow more or less frequent
# re‑entries.
PAIR_TRADE_COOLDOWN_SECONDS: int = 1800

# -----------------------------------------------------------------------------
# Microcap Filter
#
# Many very low‑priced tokens (often costing fractions of a cent) tend to
# suffer from poor liquidity, large spreads and strict minimum order sizes on
# OKX.  Attempting to trade these so‑called "microcap" coins frequently leads
# to repeated "insufficient margin" or "max size" errors and triggers long
# adaptation loops.  To mitigate this, the bot skips any symbol whose
# current price is below ``MICROCAP_PRICE_THRESHOLD``.  Adjust this value as
# needed to trade lower‑priced markets.
MICROCAP_PRICE_THRESHOLD: float = 0.001

# ---------------------------------------------------------------------------
# Options metrics configuration
#
# Only fetch options metrics for these underlying assets.  Many
# cryptocurrencies lack a liquid options market on Deribit.  Fetching
# metrics for unsupported assets can result in repeated connection errors
# or rate limit issues.  Restrict to BTC and ETH by default.  Extend
# this set if other assets gain options liquidity in the future.
OPTIONS_SUPPORTED_BASES: set[str] = {"BTC", "ETH"}

# ---------------------------------------------------------------------------
# Live data state
#
# The following dictionaries accumulate live data from WebSocket streams
# so that the trading logic and risk manager can inspect them in real time.
# Each dict maps an underlying (e.g. "BTC") or symbol (e.g. "BTCUSDT")
# to the latest metrics.  These structures are updated by background tasks
# started in ``main()`` when the corresponding WebSocket stream is enabled via
# environment variables.  Keys will be created on demand.
#
# - OPTIONS_LIVE_METRICS: holds option metrics from Deribit (implied volatility,
#   open interest, put/call ratio)
# - LIQUIDATION_INTENSITY: accumulates notional size of liquidation events on
#   each symbol since the last risk evaluation cycle.  Callers should reset
#   counts periodically if required.
# - WHALE_ALERTS: collects large transfer events for monitored addresses.
OPTIONS_LIVE_METRICS: Dict[str, Dict[str, float]] = {}
LIQUIDATION_INTENSITY: Dict[str, float] = {}
WHALE_ALERTS: Dict[str, list] = {}

# ---------------------------------------------------------------------------
# Social sentiment snapshot (cached)
#
# The bot maintains a background sentiment scheduler that writes
# metrics/social_sentiment.json.  In previous versions, per-symbol analysis
# used hardcoded placeholder values. Phase-2 wires the real metrics into the
# decision inputs with safe, deterministic degrade behaviour.
_SOCIAL_SENTIMENT_CACHE: dict = {"ts": 0.0, "data": {}}

def _read_social_sentiment_cached(ttl_sec: float = 60.0) -> dict:
    """Read metrics/social_sentiment.json with a small in-process cache."""
    import time
    from pathlib import Path
    import json

    now = time.time()
    try:
        if (now - float(_SOCIAL_SENTIMENT_CACHE.get("ts", 0.0))) < float(ttl_sec):
            data = _SOCIAL_SENTIMENT_CACHE.get("data")
            return data if isinstance(data, dict) else {}
    except Exception:
        pass

    path = Path("metrics") / "social_sentiment.json"
    data: dict = {}
    try:
        if path.exists():
            txt = path.read_text(encoding="utf-8").strip()
            if txt:
                parsed = json.loads(txt)
                if isinstance(parsed, dict):
                    data = parsed
    except Exception:
        data = {}
    _SOCIAL_SENTIMENT_CACHE["ts"] = now
    _SOCIAL_SENTIMENT_CACHE["data"] = data
    return data


def _sentiment_snapshot_for_decision() -> dict:
    """Build a compact sentiment dict for controller_async.decide_batch."""
    data = _read_social_sentiment_cached()
    # Fear & Greed: updater stores normalized [0,1]. Controller expects 0..100.
    fng_norm = data.get("fng_value")
    try:
        fng_norm_f = float(fng_norm) if fng_norm is not None else None
    except Exception:
        fng_norm_f = None
    if fng_norm_f is not None:
        fng_norm_f = max(0.0, min(1.0, fng_norm_f))
        fear_greed_0_100 = fng_norm_f * 100.0
    else:
        fear_greed_0_100 = None

    # Global sentiment (already 0..1). Use as social_score.
    try:
        social_score = float(data.get("global_sentiment", 0.5))
    except Exception:
        social_score = 0.5
    social_score = max(0.0, min(1.0, social_score))

    # Availability: at least one real signal present
    sources = ("tweet_sentiment", "reddit_sentiment", "news_sentiment", "lunarcrush_sentiment")
    available_count = 0
    for k in sources:
        if data.get(k) is not None:
            available_count += 1
    social_available = available_count > 0

    return {
        # Derivatives placeholders stay None unless you wire real OKX metrics.
        "funding": None,
        "oi_change": None,
        "fear_greed": fear_greed_0_100,
        "social_score": social_score,
        "social_available": social_available,
        "social_sources": available_count,
    }

def _load_blacklist() -> list[str]:
    """Karaliste dosyasını yükle ve içindeki sembolleri döndür.

    Eğer dosya yoksa veya içeriği boşsa, boş liste döndürülür.
    """
    try:
        if not BLACKLIST_FILE.exists():
            return []
        txt = BLACKLIST_FILE.read_text(encoding="utf-8").strip()
        if not txt:
            return []
        data = json.loads(txt)
        if isinstance(data, list):
            return [str(s).strip().upper() for s in data if isinstance(s, str)]
    except Exception:
        pass
    return []

def _save_blacklist(symbols: list[str]) -> None:
    """Karalisteyi dosyaya yaz.  Yazma hataları sessizce yutulur."""
    try:
        BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with BLACKLIST_FILE.open("w", encoding="utf-8") as f:
            json.dump(symbols, f, indent=2)
    except Exception:
        pass

def _handle_user_command(command: str, args: str) -> None:
    """Telegram komutlarını işler.

    Desteklenen komutlar:

    * ``pause``: Botu duraklatır.
    * ``resume``: Duraklatılmış botu devam ettirir.
    * ``blacklist <SYM1,SYM2,...>``: Verilen sembolleri karalisteye ekler.
    * ``unblacklist <SYM1,SYM2,...>``: Belirtilen sembolleri karalisteden çıkarır.
    * ``risk <value>``: Kullanıcı risk katsayısını ayarlar (0–1 arası).  Bu
      katsayı, çok kademeli kill‑switch tarafından uygulanan risk azaltma
      faktörü ile çarpılarak toplam risk ayarını etkiler.

    Bilinmeyen komutlar log'a yazılır.
    """
    global BOT_PAUSED, RISK_FACTOR_USER
    cmd = command.strip().lower() if isinstance(command, str) else ""
    arg_str = args.strip() if isinstance(args, str) else ""
    if cmd == "pause":
        BOT_PAUSED = True
        # Record the pause reason for diagnostics
        PAUSE_REASON = "user command"
        try:
            from notification import send_notification  # type: ignore
            send_notification("Trading bot paused by user command.")
        except Exception:
            pass
        log.info("User command: bot paused")
    elif cmd == "resume":
        BOT_PAUSED = False
        PAUSE_REASON = ""
        try:
            from notification import send_notification  # type: ignore
            send_notification("Trading bot resumed by user command.")
        except Exception:
            pass
        log.info("User command: bot resumed")
    elif cmd == "blacklist":
        syms = [s.strip().upper() for s in arg_str.split(",") if s.strip()]
        if syms:
            bl = set(_load_blacklist())
            for s in syms:
                bl.add(s)
            _save_blacklist(sorted(list(bl)))
            log.info(f"User command: added to blacklist {syms}")
        else:
            log.info("User command: blacklist called with no symbols")
    elif cmd == "unblacklist":
        syms = [s.strip().upper() for s in arg_str.split(",") if s.strip()]
        if syms:
            bl = set(_load_blacklist())
            for s in syms:
                if s in bl:
                    bl.remove(s)
            _save_blacklist(sorted(list(bl)))
            log.info(f"User command: removed from blacklist {syms}")
        else:
            log.info("User command: unblacklist called with no symbols")
    elif cmd == "risk":
        try:
            val = float(arg_str)
            if 0.0 <= val <= 1.0:
                RISK_FACTOR_USER = val
                log.info(f"User command: risk factor set to {val}")
            else:
                log.warning("risk command: value out of range [0,1]")
        except Exception:
            log.warning("risk command: invalid value")
    else:
        log.info(f"Unknown user command: {cmd} {arg_str}")

def _watchdog_callback(message: str) -> None:
    """Handle watchdog alerts by pausing the bot and recording the reason.

    When the watchdog detects a high CPU/memory condition or other anomaly it
    invokes this callback with a descriptive message.  We set the global
    ``BOT_PAUSED`` flag and store the message in ``PAUSE_REASON`` so that
    subsequent loop iterations can display why the bot is sleeping.  A
    notification is sent to alert the operator.
    """
    global BOT_PAUSED, PAUSE_REASON
    BOT_PAUSED = True
    PAUSE_REASON = message
    log.error(f"[WATCHDOG] {message} → Bot paused for safety")
    try:
        from notification import send_notification  # type: ignore
        send_notification(f"🚨 Watchdog alert: {message}")
    except Exception:
        pass

def _update_price_history(symbol: str, price: float, max_len: int = 60) -> None:
    """PRICE_HISTORY sözlüğünü günceller.

    Her sembol için en fazla ``max_len`` değer tutar.  Eski değerler
    listeden çıkarılır.  Hata durumları sessizce yutulur.
    """
    try:
        hist = PRICE_HISTORY.get(symbol)
        if hist is None:
            PRICE_HISTORY[symbol] = [float(price)]
        else:
            hist.append(float(price))
            if len(hist) > max_len:
                PRICE_HISTORY[symbol] = hist[-max_len:]
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# AUTO-UPDATER ENTEGRASYONU
# ──────────────────────────────────────────────────────────────────────────────
try:
    from auto_updater import start_background
    # Her 60 dakikada bir otomatik sistem taraması ve güncelleme
    start_background(interval_min=60)
except Exception as e:
    print(f"[WARN] AutoUpdater başlatılamadı: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# BOT BİLEŞENLERİ
# ──────────────────────────────────────────────────────────────────────────────
from settings import OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, OKX_USE_TESTNET
from logger import get_logger
from retry_utils import retry
from state_manager import is_killed, pause_loop_sleep, DailyKillSwitch
from analyzer import get_multi_timeframe_analysis, get_multi_timeframe_analysis_async
from portfolio_tools import calculate_correlation
from controller_async import decide_batch, is_llm_tech_only_mode
from risk_manager import (
    calculate_position_size, get_entry_levels, get_exit_levels,
    calculate_tiered_leverage_and_allocation, apply_dynamic_wallet_cap,
    adjust_risk_for_volatility, compute_stop_loss,
    manage_trailing_stop,
)
from safe_order_wrapper import (
    safe_submit_entry_plan, safe_submit_exit_plan,
    normalize_okx_symbol, safe_set_leverage, adaptive_order_sleep, call,
    safe_update_stop_loss,
)
from trade_logger import log_trade_open, log_trade_close, get_daily_realized_pnl

# ---------------------------------------------------------------------------
# Concurrency guards
#
# OKX (especially demo/testnet) can rate-limit aggressively when many symbols
# are queried in parallel. Limit concurrency for ticker fetching and per-loop
# analyses to avoid bursts that trigger 429/rate-limit responses.
# ---------------------------------------------------------------------------
try:
    MAX_TICKER_CONCURRENCY = int(os.getenv("MAX_TICKER_CONCURRENCY", "8"))
except Exception:
    MAX_TICKER_CONCURRENCY = 8
try:
    MAX_ANALYZE_CONCURRENCY = int(os.getenv("MAX_ANALYZE_CONCURRENCY", "8"))
except Exception:
    MAX_ANALYZE_CONCURRENCY = 8

# Semaphores are safe to create at import time; they only coordinate awaits.
_TICKER_SEM = asyncio.Semaphore(max(1, MAX_TICKER_CONCURRENCY))
_ANALYZE_SEM = asyncio.Semaphore(max(1, MAX_ANALYZE_CONCURRENCY))


# ---------------------------------------------------------------------------
# Background services are started in the main entry point (see __main__ block)
# to avoid spawning threads at import time.  The sentiment scheduler and stop
# order watchdog will be launched as daemon threads when the bot is run
# directly.
# ---------------------------------------------------------------------------

log = get_logger("autotrader.async_main")
def _env_flag(key: str, default: str = "0") -> bool:
    return str(os.getenv(key, default)).strip().lower() in ("1", "true", "yes", "y", "on")

BOT_DEBUG = _env_flag("BOT_DEBUG", "0")
DEBUG_TA = BOT_DEBUG or _env_flag("DEBUG_TA", "0")
DEBUG_ANALYZE = BOT_DEBUG or _env_flag("DEBUG_ANALYZE", "0")
LOG_PRICE_LINES = _env_flag("LOG_PRICE_LINES", "1")

# ──────────────────────────────────────────────────────────────────────────────
# Sembol listesi yükleme (opsiyonel symbols_okx.json, yoksa default 5 sembol)
# ──────────────────────────────────────────────────────────────────────────────
SYMBOLS_FILE = Path("symbols_okx.json")


def _load_target_symbols(max_symbols: int = 100):
    """
    Eğer proje kökünde symbols_okx.json varsa, içindeki string listeden
    en fazla max_symbols kadarını yükler.
    Dosya yoksa veya bozuksa varsayılan 5 sembole döner.
    """
    try:
        if SYMBOLS_FILE.exists():
            txt = SYMBOLS_FILE.read_text(encoding="utf-8").strip()
            if txt:
                data = json.loads(txt)
                if isinstance(data, list):
                    syms = [str(s).strip() for s in data if isinstance(s, str)]
                    syms = [s for s in syms if s]
                    if syms:
                        if len(syms) > max_symbols:
                            syms = syms[:max_symbols]
                        log.info(f"{len(syms)} sembol symbols_okx.json dosyasından yüklendi.")
                        return syms
                # 🔹 Düzeltme: Formatı da destekle ({"symbols": [...]})
                elif isinstance(data, dict) and isinstance(data.get("symbols"), list):
                    syms = [str(s).strip() for s in data["symbols"] if isinstance(s, str)]
                    syms = [s for s in syms if s]
                    if syms:
                        if len(syms) > max_symbols:
                            syms = syms[:max_symbols]
                        log.info(f"{len(syms)} sembol symbols_okx.json dosyasından yüklendi.")
                        return syms
            log.warning("symbols_okx.json boş veya geçersiz formatta, varsayılan semboller kullanılacak.")
    except Exception as e:
        log.warning(f"symbols_okx.json okunamadı: {e}. Varsayılan semboller kullanılacak.")

    default_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT']
    log.info(f"symbols_okx.json bulunamadı, varsayılan sembol seti kullanılacak: {default_symbols}")
    return default_symbols


# ──────────────────────────────────────────────────────────────────────────────
# Parametreler
# ──────────────────────────────────────────────────────────────────────────────
TARGET_SYMBOLS = _load_target_symbols()
# OKX market doğrulaması sonrası kullanılacak aktif liste
ACTIVE_SYMBOLS = list(TARGET_SYMBOLS)

MIN_CONFIDENCE_FOR_TRADE = 0.65
BALANCE_START = 2000.0
KILLER = DailyKillSwitch(daily_limit_pct=-0.10, cooldown_hours=24)

# Çok kademeli kill‑switch entegrasyonu.  Bu, günlük realized PnL'in belli
# yüzdeleri aştığı durumlarda portföy riskini azaltır, yeni işlemleri
# durdurur veya botu tamamen kapatır.  Eğer bu modül yüklenemezse
# fallback olarak None kullanılır.
try:
    # Prefer the enhanced kill‑switch if available; fall back to MultiLevel
    try:
        from .enhanced_kill_switch import EnhancedKillSwitch  # type: ignore
    except Exception:
        from enhanced_kill_switch import EnhancedKillSwitch  # type: ignore
    KillSwitchClass = EnhancedKillSwitch
except Exception:
    try:
        try:
            from .multi_kill_switch import MultiLevelKillSwitch  # type: ignore
        except Exception:
            from multi_kill_switch import MultiLevelKillSwitch  # type: ignore
        KillSwitchClass = MultiLevelKillSwitch
    except Exception:
        KillSwitchClass = None  # type: ignore

if KillSwitchClass is not None:
    # Varsayılan eşik ve aksiyonlar: -3% → risk azalt, -5% → yeni işlem yok, -7% → botu durdur
    # Load MA-based kill switch parameters from config.json if available
    ks_extra = {}
    try:
        cfg_path = Path(__file__).resolve().parent / 'config.json'
        if cfg_path.exists():
            cfg_data = json.loads(cfg_path.read_text(encoding='utf-8'))
            ks_cfg = cfg_data.get('kill_switch') or cfg_data.get('killswitch') or {}
            if isinstance(ks_cfg, dict):
                mw = ks_cfg.get('ma_window')
                md = ks_cfg.get('ma_drawdown_pct')
                if mw is not None:
                    ks_extra['ma_window'] = int(mw)
                if md is not None:
                    ks_extra['ma_drawdown_pct'] = float(md)
    except Exception:
        ks_extra = {}

    # Only EnhancedKillSwitch supports MA params
    if getattr(KillSwitchClass, '__name__', '') != 'EnhancedKillSwitch':
        ks_extra = {}

    KILLER_MULTI = KillSwitchClass(
        limits=[(-0.03, "reduce"), (-0.05, "halt"), (-0.07, "stop")],
        cooldown_hours={"reduce": 6, "halt": 12, "stop": 24},
        **ks_extra,
    )
else:
    KILLER_MULTI = None

# Circuit breaker and orderbook modules
try:
    from .circuit_breaker import should_halt as circuit_should_halt  # type: ignore
except Exception:
    # fallback: never halt
    def circuit_should_halt(total_balance: float) -> bool:  # type: ignore
        return False

try:
    from .orderbook_analyzer import calculate_imbalance  # type: ignore
except Exception:
    async def calculate_imbalance(exchange, symbol, depth: int = 20):  # type: ignore
        return None

# Performans tabanlı sembol cooldown kontrolü için
from pathlib import Path
import json

# Cooldown dosya yolu
COOLDOWNS_FILE = Path("data") / "cooldowns.json"

def _is_symbol_in_cooldown(symbol: str) -> bool:
    """
    performans analizi sonucu oluşturulan cooldowns.json dosyasından,
    verilen sembolün hâlâ cooldown altında olup olmadığını kontrol eder.
    Dosya formatı: {"SYM/USDT": "2025-11-30T12:00:00+00:00", ...}

    Returns True if symbol should be skipped.
    """
    try:
        if not COOLDOWNS_FILE.exists():
            return False
        txt = COOLDOWNS_FILE.read_text(encoding="utf-8").strip()
        if not txt:
            return False
        data = json.loads(txt)
        if not isinstance(data, dict):
            return False
        until_str = data.get(symbol)
        if not until_str:
            return False
        try:
            until_dt = datetime.fromisoformat(str(until_str))
        except Exception:
            return False
        now = datetime.now(timezone.utc)
        return now < until_dt
    except Exception:
        return False
WRAPPER_DRY_RUN = False  # True = sadece simülasyon (emir göndermez)

# trade_parameters.dry_run override: config.json içindeki "trade_parameters" 
# alanında "dry_run": true ise bu değer kullanılır. Böylece tüm emir
# fonksiyonları dry-run modunda çalışır ve borsaya emir gönderilmez.
try:
    # _CONFIG_DATA yukarıda okundu
    tp = _CONFIG_DATA.get("trade_parameters") or {}
    if "dry_run" in tp and tp["dry_run"] is not None:
        WRAPPER_DRY_RUN = bool(tp["dry_run"])
except Exception:
    pass

# ----
# Atomic SL/TP Kullanımı
# OKX borsasında giriş emri ile birlikte attachAlgoOrds kullanarak atomik
# stop-loss ve take-profit emirleri eklenebiliyor. Ancak bu özellik
# bazı durumlarda emirlerin reddedilmesine veya stop/TP emirlerinin
# tetiklenmemesine yol açabilir. Kullanıcı raporlarına göre stop-loss
# emirlerinin çalışmadığı durumlar yaşandığından, atomik SL/TP
# kullanımını bu bayrak ile kontrol ediyoruz. USE_ATOMIC_SLTP=False
# olduğunda safe_submit_entry_plan fonksiyonuna sl_price ve tp_price
# değerleri None olarak iletilir ve SL/TP emirleri yalnızca
# safe_submit_exit_plan tarafından oluşturulur.
USE_ATOMIC_SLTP = False

# Dinamik bakiye (gerçek USDT bakiyesi buraya yansıtılır)
CURRENT_BALANCE: float = BALANCE_START

# Daha korumacı sabit stop-loss (fallback). Bu değer config.json içindeki
# trade_parameters.stop_loss_pct ile override edilebilir.
STOP_LOSS_PERCENT = 0.005  # 0.5% sabit stop-loss

# Maksimum slipaj toleransı (örneğin %0.1). Limit emirler için kullanılacak.
MAX_SLIPPAGE_PERCENT = 0.001

# TP için hedef kâr aralığı (fallback modda, ATR yoksa). Bu değerler
# trade_parameters.take_profit_multiplier ile stop-loss yüzdesine göre
# çarpılarak override edilebilir.
TP_MIN_PCT = 0.01  # %1
TP_MAX_PCT = 0.05  # %5

# Pozisyon bazlı ekstra koruma (mini kill-switch)
MAX_TRADE_DRAWDOWN_PCT = 0.015  # örnek: %1.5 zarar olursa pozisyonu zorla kapat
# Çok uzak limit emirlerini temizlemek için mesafe eşiği
MAX_ORDER_DISTANCE_PCT = 0.02   # [FIX] Fiyat %2 uzaklaşırsa emri iptal et

# Füzyon (BiLSTM / RL / LLM) varsayılan ağırlıkları; config.json'dan da okunur
DEFAULT_HYBRID_WEIGHTS = {"chatgpt": 0.5, "deepseek": 0.5, "bilstm": 0.30, "ppo_rl": 0.20}
CONFIG_FILE = Path("config.json")
AI_PRED_LOG_FILE = Path("metrics/ai_predictions.json")

def load_config(force: bool = False) -> Dict[str, Any]:
    """Load config.json into the global cache.

    - Uses a single source of truth (_CONFIG_DATA)
    - Provides explicit logging on JSON errors instead of silently
      falling back to {}
    """
    global _CONFIG_DATA
    if _CONFIG_DATA and not force:
        return _CONFIG_DATA
    if not CONFIG_FILE.exists():
        _CONFIG_DATA = {}
        return _CONFIG_DATA
    try:
        _CONFIG_DATA = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        # Keep the process running but make the failure obvious.
        try:
            log.exception(f"[CONFIG] config.json okunamadı veya bozuk: {e}")
        except Exception:
            print(f"[CONFIG][ERROR] config.json okunamadı veya bozuk: {e}")
        _CONFIG_DATA = {}
    return _CONFIG_DATA


# Yüklenen config (risk profil, cooldown, max open vs)
load_config(force=True)

# Trade cooldown ve açık pozisyon limiti config'ten yüklenir
TRADE_COOLDOWN_MIN = int(_CONFIG_DATA.get("trade_cooldown_min", 0))
MAX_OPEN_POSITIONS = int(_CONFIG_DATA.get("max_open_positions", 0))
# İstenirse açık pozisyon limitini tamamen devre dışı bırak (varsayılan: limit yok).
# max_open_positions>0 olsa bile enforce_max_open_positions=False ise limit uygulanmaz.
ENFORCE_MAX_OPEN_POSITIONS = bool(_CONFIG_DATA.get("enforce_max_open_positions", False))

# Limit the total number of symbols analysed per loop.  Analysing too
# many symbols concurrently can overwhelm CPU resources and trigger the
# watchdog.  Adjust via the ``MAX_SYMBOLS_TO_ANALYZE`` environment
# variable; if unset the default of 15 is used.  Set to 0 to disable
# slicing (analyse all symbols).
try:
    _ms = os.getenv("MAX_SYMBOLS_TO_ANALYZE")
    if _ms is not None:
        MAX_SYMBOLS_TO_ANALYZE = max(0, int(float(_ms)))
    else:
        # Default to analyse a larger universe per loop.  Processing too few symbols
        # can miss opportunities, so the default has been raised to scan up to 100
        # symbols.  Set to 0 to disable slicing entirely.
        MAX_SYMBOLS_TO_ANALYZE = 100
except Exception:
    MAX_SYMBOLS_TO_ANALYZE = 15

# ------------------------------------------------------------------------------
# Trade parametreleri override: trade_parameters alanı var ise
# STOP_LOSS_PERCENT, TP_MIN_PCT/TP_MAX_PCT ve dry-run ayarını güncelle.
try:
    trade_params = _CONFIG_DATA.get("trade_parameters") or {}
    # stop-loss yüzdesi override
    sl_pct = trade_params.get("stop_loss_pct")
    if sl_pct is not None:
        try:
            STOP_LOSS_PERCENT = float(sl_pct)
        except Exception:
            pass
    # take profit multiplier override: TP aralığını stop-loss yüzdesi ile çarp
    tp_mult = trade_params.get("take_profit_multiplier")
    if tp_mult is not None:
        try:
            mult = float(tp_mult)
            # Eğer STOP_LOSS_PERCENT değiştirildiyse yeni değeri kullan
            TP_MIN_PCT = float(STOP_LOSS_PERCENT) * mult
            TP_MAX_PCT = float(STOP_LOSS_PERCENT) * mult
        except Exception:
            pass
    # global dry-run override
    wrapper_dry = trade_params.get("dry_run")
    if wrapper_dry is not None:
        try:
            # WRAPPER_DRY_RUN değişkeni aşağıda tanımlanıyor; burada sadece değer
            # değişkenini ayarlayıp, tanımını kaldırmayın
            pass
        except Exception:
            pass
except Exception:
    pass

# ------------------------------------------------------------------------------
# Portföy kategorileri ve sınırlar
#
# config.json içindeki "category_limits" alanından alınan maksimum açık
# pozisyon sayısı ve sector_map.json dosyasından yüklenen sembol→kategori
# eşlemesiyle, belirli sektörlerde aşırı yığılmayı engellemek için yeni bir
# koruma mekanizması eklenmiştir.  "sector_map.json" dosyası projenin kök
# dizininde bulunmalıdır ve coin sembollerini (örneğin BTC) sektör
# isimlerine (örneğin "Layer1", "DeFi") eşler.

CATEGORY_LIMITS: Dict[str, int] = {}
try:
    cl = _CONFIG_DATA.get("category_limits")
    if isinstance(cl, dict):
        for k, v in cl.items():
            try:
                CATEGORY_LIMITS[str(k)] = int(v)
            except Exception:
                continue
except Exception:
    pass

SECTOR_MAP: Dict[str, str] = {}
try:
    sec_path = Path(__file__).resolve().parent / "sector_map.json"
    if sec_path.exists():
        with sec_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # normalize keys to upper case without suffix (e.g. BTC/USDT → BTC)
            for key, val in data.items():
                if not isinstance(key, str):
                    continue
                sym = key.split("/")[0].upper()
                SECTOR_MAP[sym] = str(val)
except Exception:
    pass

def _get_symbol_category(symbol: str) -> Optional[str]:
    """
    Returns the sector/category for a given trading symbol based on
    sector_map.json.  Symbols are normalized by taking the base
    asset (e.g. 'BTC/USDT' → 'BTC') and converting to upper case.
    If the symbol is not found or the map is empty, returns None.
    """
    try:
        if not SECTOR_MAP:
            return None
        base = str(symbol).split("/")[0].upper()
        return SECTOR_MAP.get(base)
    except Exception:
        return None

symbol_locks: Dict[str, asyncio.Lock] = {}
order_queue: asyncio.Queue = asyncio.Queue()

# Analiz tarafı için eşzamanlılık limiti (aynı anda en fazla N sembol)
# Limit the number of concurrent symbol analyses.  A lower number reduces
# CPU utilisation and prevents the watchdog from pausing the bot during
# heavy initialisation or market analysis.  You can override this via the
# environment variable ``ANALYSIS_CONCURRENCY``.  Default: 4.
try:
    ANALYSIS_CONCURRENCY = int(os.getenv("ANALYSIS_CONCURRENCY", "4"))
except Exception:
    ANALYSIS_CONCURRENCY = 4
analysis_sema = asyncio.Semaphore(ANALYSIS_CONCURRENCY)

# Trailing TP için pozisyon durumu
position_state: Dict[str, Dict[str, float]] = {}

# ──────────────────────────────────────────────────────────────────────────────
# OHLC METRICS YAZIMI (build_dataset.py için veri üretimi)
# ──────────────────────────────────────────────────────────────────────────────
METRICS_DIR = Path("metrics")
OHLC_FILE = METRICS_DIR / "ohlc_history.json"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
_ohlc_lock = threading.Lock()


def _safe_load_json_list(path: Path):
    try:
        if path.exists():
            txt = path.read_text(encoding="utf-8").strip()
            if not txt:
                return []
            obj = json.loads(txt)
            # Hem [] hem {} biçimini destekle
            if isinstance(obj, dict):
                return obj.get("rows", [])
            return obj
    except Exception:
        pass
    return []

# --------------------------------------------------------------------------
# Dinamik Hibrit Ağırlıklandırma
#
# Her modelin son performansına göre hibrit ağırlıkları hafifçe ayarlanır.
# - BiLSTM: last_accuracy metriği > 0.5 ise ağırlığı artırılır, <0.5 ise azaltılır.
# - RL: Modelin en son güncellenme tarihi eskidikçe ağırlığı azaltılır.
# LLM modelleri için henüz bir performans metriği olmadığından sabit kalır.
# Ağırlıklar yeniden normalize edilir, böylece toplam etkisi aynı kalır.

def _get_dynamic_hybrid_weights(base_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Hibrit ağırlıkları dinamik olarak ayarlar.

    Args:
        base_weights: config veya runtime ayarlarından gelen nominal ağırlıklar.

    Returns:
        Yeni ağırlıklar sözlüğü. Sum(base_weights) korunur.
    """
    weights = dict(base_weights) if base_weights else {}
    try:
        metrics_dir = Path("metrics")
        bilstm_path = metrics_dir / "bilstm_metrics.json"
        rl_path = metrics_dir / "rl_metrics.json"
        # Baseline total weight to preserve
        original_sum = 0.0
        for k in ("chatgpt", "deepseek", "bilstm", "ppo_rl"):
            if k in weights:
                original_sum += float(weights[k])
        # Adjust BiLSTM weight based on accuracy
        if "bilstm" in weights and bilstm_path.exists():
            try:
                data = json.loads(bilstm_path.read_text(encoding="utf-8"))
                acc = data.get("last_accuracy")
                if isinstance(acc, (float, int)):
                    # scale factor: centered at 1.0 when acc=0.6, ranges ~0.8–1.2 for 0.5–0.7
                    factor = 1.0 + ((acc - 0.6) * 1.0)
                    # Clamp factor
                    factor = max(0.6, min(1.4, factor))
                    weights["bilstm"] = weights.get("bilstm", 0.0) * factor
            except Exception:
                pass
        # Adjust RL weight based on recency
        if "ppo_rl" in weights and rl_path.exists():
            try:
                data = json.loads(rl_path.read_text(encoding="utf-8"))
                last_update = data.get("last_update")
                if isinstance(last_update, str):
                    try:
                        last_dt = datetime.fromisoformat(last_update)
                        age_days = (datetime.now(timezone.utc) - last_dt).days
                        # Factor: 1.2 when fresh (<7 days), down to 0.7 when stale (>30 days)
                        if age_days <= 7:
                            factor = 1.2
                        elif age_days <= 14:
                            factor = 1.1
                        elif age_days <= 21:
                            factor = 0.9
                        elif age_days <= 30:
                            factor = 0.8
                        else:
                            factor = 0.7
                        weights["ppo_rl"] = weights.get("ppo_rl", 0.0) * factor
                    except Exception:
                        pass
            except Exception:
                pass
        # Normalize weights to preserve original sum
        new_sum = 0.0
        for k in ("chatgpt", "deepseek", "bilstm", "ppo_rl"):
            w = weights.get(k)
            if w is not None:
                new_sum += float(w)
        if new_sum > 0 and original_sum > 0:
            scale = original_sum / new_sum
            for k in weights:
                weights[k] = float(weights[k]) * scale
    except Exception:
        pass
    return weights


def _safe_save_json_list(path: Path, rows):
    try:
        payload = {"rows": rows}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def append_ohlc(symbol: str, ts_iso: str, o: float, h: float, l: float, c: float, v: float):
    """Son 5m mumu metrics/ohlc_history.json içine ekler (idempotent değil; build_dataset window’lu kullanır)."""
    with _ohlc_lock:
        rows = _safe_load_json_list(OHLC_FILE)
        rows.append({
            "symbol": symbol.replace("/", ""),
            "ts": ts_iso,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(v),
        })
        # Büyüme kontrolü (son 300k kayıt)
        if len(rows) > 300_000:
            rows = rows[-300_000:]
        _safe_save_json_list(OHLC_FILE, rows)

# --------------------------------------------------------------------------
# Trade Cooldown ve Açık Pozisyon Limit Kontrolleri
#
# Her sembol için son işlem açılış zamanını kontrol ederek belirli bir süre
# boyunca yeni pozisyon açılmasını engellemek için yardımcı fonksiyonlar.
# Ayrıca, aynı anda açık/planned pozisyon sayısını sınırlamak için
# MAX_OPEN_POSITIONS parametresi kullanılabilir.

TRADE_LOG_PATH = Path("trade_log.json")

def _get_last_trade_open_time(symbol: str) -> Optional[datetime]:
    """
    trade_log.json içinden ilgili sembol için en son açık trade'in
    açılış zamanını (datetime) döndürür. Bulunamazsa None.
    """
    try:
        if not TRADE_LOG_PATH.exists():
            return None
        txt = TRADE_LOG_PATH.read_text(encoding="utf-8").strip()
        if not txt:
            return None
        data = json.loads(txt)
        if not isinstance(data, list):
            return None
        for row in reversed(data):
            if str(row.get("symbol")) != symbol:
                continue
            ts = row.get("timestamp_open")
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except Exception:
                    pass
        return None
    except Exception:
        return None

def _should_skip_due_to_cooldown(symbol: str) -> bool:
    """
    Config'te tanımlı cooldown süresi boyunca aynı sembolde ikinci kez trade açmayı engeller.
    TRADE_COOLDOWN_MIN <= 0 ise hiçbir zaman skip etmez.
    """
    try:
        cooldown = int(TRADE_COOLDOWN_MIN)
        if cooldown <= 0:
            return False
        last_time = _get_last_trade_open_time(symbol)
        if last_time is None:
            return False
        diff = datetime.now(timezone.utc) - last_time
        # diff.total_seconds() kullanarak dakika hesapla
        minutes = diff.total_seconds() / 60.0
        return minutes < cooldown
    except Exception:
        return False


def _lock_for(sym: str) -> asyncio.Lock:
    """Her sembol için ayrı lock (çakışmayı engeller)."""
    if sym not in symbol_locks:
        symbol_locks[sym] = asyncio.Lock()
    return symbol_locks[sym]


# ──────────────────────────────────────────────────────────────────────────────
# Gerçek bakiye okuma (USDT) - SARMALANDI
# ──────────────────────────────────────────────────────────────────────────────
async def _get_balances(exchange, ccy: str = "USDT") -> (float, float):
    """Fetch (total, free) balance from OKX as reliably as possible.

    Notes:
    - ccxt `fetch_balance()` may return different shapes depending on OKX account type.
    - OKX v5 embeds reliable values under `balance["info"]["data"][0]["details"]`.
    - We only fall back to BALANCE_START as a last resort (legacy behavior).

    Returns:
        (total_balance, free_balance) as non-negative floats.
    """

    def _sf(x):
        try:
            if x is None:
                return None
            if isinstance(x, str):
                x = x.strip()
                if x == "":
                    return None
            v = float(x)
            if v != v:  # NaN
                return None
            return v
        except Exception:
            return None

    def _parse_okx_info(info):
        """Parse OKX raw payload (or ccxt 'info') for the requested currency."""
        if info is None:
            return None, None

        if isinstance(info, list) and info:
            if isinstance(info[0], dict):
                info = {"data": info}

        if not isinstance(info, dict):
            return None, None

        data = info.get("data")
        item0 = None
        if isinstance(data, list) and data:
            item0 = data[0] if isinstance(data[0], dict) else None
        elif isinstance(data, dict):
            item0 = data

        # Account balance: data[0].details[*]
        if isinstance(item0, dict):
            details = item0.get("details")
            if isinstance(details, list):
                for d in details:
                    if not isinstance(d, dict):
                        continue
                    if str(d.get("ccy", "")).upper() != str(ccy).upper():
                        continue
                    total_v = _sf(d.get("eq")) or _sf(d.get("cashBal")) or _sf(d.get("bal")) or _sf(d.get("totalEq"))
                    free_v = _sf(d.get("availEq")) or _sf(d.get("availBal")) or _sf(d.get("avail")) or _sf(d.get("free"))
                    return total_v, free_v

        # Funding/assets endpoints: data[*] may already be currency rows
        if isinstance(data, list):
            for d in data:
                if not isinstance(d, dict):
                    continue
                if str(d.get("ccy", "")).upper() != str(ccy).upper():
                    continue
                total_v = _sf(d.get("bal")) or _sf(d.get("cashBal")) or _sf(d.get("eq"))
                free_v = _sf(d.get("availBal")) or _sf(d.get("availEq")) or _sf(d.get("free"))
                return total_v, free_v

        return None, None

    async def _fetch_ccxt_balance():
        try:
            acc_type = None
            try:
                acc_type = (getattr(exchange, "options", {}) or {}).get("defaultType")
            except Exception:
                acc_type = None
            if not acc_type:
                acc_type = "swap"

            params = {"type": acc_type, "ccy": ccy}
            return await call(exchange.fetch_balance, params, label=f"FETCH_BALANCE[{acc_type}]")
        except Exception as e:
            log.warning(f"[BAL] fetch_balance hata: {e}")
            return None

    async def _fetch_okx_raw():
        candidates = [
            ("privateGetAccountBalance", {"ccy": ccy}),
            ("privateGetAssetBalances", {"ccy": ccy}),
        ]
        for meth_name, params in candidates:
            meth = getattr(exchange, meth_name, None)
            if not callable(meth):
                continue
            try:
                raw = await call(meth, params, label=f"RAW_{meth_name}")
                if isinstance(raw, dict) and raw:
                    return raw
            except Exception:
                continue
        return None

    bal = await _fetch_ccxt_balance()
    total = None
    free = None

    # 1) Standard ccxt shapes
    try:
        if isinstance(bal, dict):
            sub = bal.get(ccy)
            if isinstance(sub, dict):
                total = _sf(sub.get("total")) or _sf(sub.get("equity")) or _sf(sub.get("cash")) or _sf(sub.get("balance"))
                free = _sf(sub.get("free")) or _sf(sub.get("available")) or _sf(sub.get("avail")) or _sf(sub.get("availEq")) or _sf(sub.get("availBal"))

            tot_dict = bal.get("total")
            if total is None and isinstance(tot_dict, dict):
                total = _sf(tot_dict.get(ccy))

            free_dict = bal.get("free")
            if free is None and isinstance(free_dict, dict):
                free = _sf(free_dict.get(ccy))
    except Exception:
        total = None
        free = None

    # 2) OKX info parsing
    if (total is None or free is None) and isinstance(bal, dict):
        try:
            t2, f2 = _parse_okx_info(bal.get("info"))
            if total is None:
                total = t2
            if free is None:
                free = f2
        except Exception:
            pass

    # 3) Raw OKX endpoints
    if total is None or free is None:
        try:
            raw = await _fetch_okx_raw()
            if raw:
                t3, f3 = _parse_okx_info(raw)
                if total is None:
                    total = t3
                if free is None:
                    free = f3
        except Exception:
            pass

    # 4) Last resort
    if total is None:
        if not hasattr(_get_balances, "_warned_no_total"):
            _get_balances._warned_no_total = True
            log.warning("[BAL] USDT toplam bakiyesi bulunamadı → BALANCE_START fallback kullanılacak.")
        total = float(BALANCE_START)

    if free is None:
        free = total

    return max(0.0, float(total)), max(0.0, float(free))


async def _get_real_balance(exchange, ccy: str = "USDT") -> float:
    """
    Toplam bakiyeyi döndürür (ekranda gösterim ve kill-switch için).
    """
    total, _free = await _get_balances(exchange, ccy)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Config okuma (hibrit ağırlıklar)
# ──────────────────────────────────────────────────────────────────────────────
def _load_hybrid_weights() -> Dict[str, float]:
    try:
        cfg = load_config()
        if cfg:
            hw = dict(DEFAULT_HYBRID_WEIGHTS)
            # dashboard v6.3+: "hybrid_weights": {"chatgpt":..,"deepseek":..,"bilstm":..,"rl":..}
            if isinstance(cfg.get("hybrid_weights"), dict):
                hw.update(cfg["hybrid_weights"])
            # opsiyonel eski anahtarlar:
            if isinstance(cfg.get("bilstm_weight"), (int, float)):
                hw["bilstm"] = float(cfg.get("bilstm_weight"))
            if isinstance(cfg.get("rl_weight"), (int, float)):
                hw["ppo_rl"] = float(cfg.get("rl_weight"))
            return hw
    except Exception as e:
        log.warning(f"hybrid_weights okunamadı: {e}")
    return dict(DEFAULT_HYBRID_WEIGHTS)


# ──────────────────────────────────────────────────────────────────────────────
# AI Tahmin Okuyucu (BiLSTM / RL / LLM)
# ──────────────────────────────────────────────────────────────────────────────
def _get_latest_ai_scores(symbol: str) -> Dict[str, Optional[float]]:
    """
    metrics/ai_predictions.json içinden ilgili sembol için en son güven değerleri:
    Dönüş: {
      "bilstm": Optional[0..1],
      "ppo_rl": Optional[0..1],
      "chatgpt": Optional[0..1],
      "deepseek": Optional[0..1]
    }
    """
    out: Dict[str, Optional[float]] = {
        "bilstm": None,
        "ppo_rl": None,
        "chatgpt": None,
        "deepseek": None,
    }
    try:
        if not AI_PRED_LOG_FILE.exists():
            return out
        data = json.loads(AI_PRED_LOG_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return out

        # model alanına göre eşleme
        llm_chatgpt_aliases = {"chatgpt", "gpt", "gpt-4", "gpt-4o", "gpt-4o-mini"}
        llm_deepseek_aliases = {"deepseek", "deepseek-reasoner", "deepseek-chat"}
        rl_aliases = {"ppo_rl", "ppo-rl", "rl", "ppo", "dqn", "dqn_rl"}

        for row in reversed(data):
            if str(row.get("symbol")) != symbol:
                continue
            m = str(row.get("model", "")).lower()
            conf = row.get("confidence", None)
            try:
                conf_val = float(conf) if conf is not None else None
            except Exception:
                conf_val = None
            if conf_val is None:
                continue
            conf_val = max(0.0, min(1.0, conf_val))

            # Identify model types by allowing substring matches. Previous
            # implementation relied on exact set membership, which failed
            # to recognise composite model names such as "hybrid (gpt+deepseek)".
            # We treat any model containing "bilstm" as a BiLSTM score,
            # any model containing "gpt" or any known alias as ChatGPT,
            # any model containing "deepseek" as DeepSeek, and any model
            # containing "rl" or "ppo" as an RL score. Exact matches still
            # work for backwards compatibility.
            if out["bilstm"] is None and ("bilstm" in m or m in ["bilstm"]):
                out["bilstm"] = conf_val
            if out["ppo_rl"] is None and (m in rl_aliases or "ppo" in m or "rl" in m):
                out["ppo_rl"] = conf_val
            if out["chatgpt"] is None and (m in llm_chatgpt_aliases or "gpt" in m):
                out["chatgpt"] = conf_val
            if out["deepseek"] is None and (m in llm_deepseek_aliases or "deepseek" in m):
                out["deepseek"] = conf_val

            if all(v is not None for v in out.values()):
                break

    except Exception as e:
        log.warning(f"{symbol}: ai_predictions.json okunamadı: {e}")
    return out


def _fuse_confidence(
    base_conf: float,
    bilstm_conf: Optional[float],
    rl_conf: Optional[float],
    chatgpt_conf: Optional[float],
    deepseek_conf: Optional[float],
    weights: Dict[str, float],
) -> float:
    """
    Teknik+sentimentten gelen base_conf üzerine
    BiLSTM, RL ve LLM (ChatGPT + DeepSeek) katkısını ekler.
    Formül: base + w*(conf-0.5)*2  → [-1..+1] etkisi. Sonuç [0..1] clamp.
    """
    fused = float(base_conf)
    try:
        if chatgpt_conf is not None:
            fused += float(weights.get("chatgpt", DEFAULT_HYBRID_WEIGHTS["chatgpt"])) * ((chatgpt_conf - 0.5) * 2.0)
        if deepseek_conf is not None:
            fused += float(weights.get("deepseek", DEFAULT_HYBRID_WEIGHTS["deepseek"])) * ((deepseek_conf - 0.5) * 2.0)
        if bilstm_conf is not None:
            fused += float(weights.get("bilstm", DEFAULT_HYBRID_WEIGHTS["bilstm"])) * ((bilstm_conf - 0.5) * 2.0)
        if rl_conf is not None:
            fused += float(weights.get("ppo_rl", DEFAULT_HYBRID_WEIGHTS["ppo_rl"])) * ((rl_conf - 0.5) * 2.0)
    except Exception as e:
        log.warning(f"confidence füzyon hatası: {e}")
    return max(0.0, min(1.0, fused))


# ---------------------------------------------------------------------------
# Risk Penalty Calculation
#
# The following helper computes a penalty factor for a given symbol based on
# live metrics from optional data feeds (options, liquidation, whale alerts).
# The returned value is between 0 and 1; it represents the fraction of
# confidence to subtract from the fused confidence score when deciding
# whether to open a new position.  If no metrics are available or
# streaming is disabled, the penalty will be zero.
def _compute_risk_penalty(sym: str) -> float:
    """
    Compute a penalty factor based on optional real‑time metrics such as
    options implied volatility, large liquidation events and whale
    transfer alerts.  A higher penalty reduces the final confidence of
    entering a trade.  The returned value is in the range [0, 1] and
    represents the fraction of confidence to be subtracted (e.g. 0.2
    means reduce confidence by 20%).  If no metrics are available
    (e.g. WebSocket streams are disabled), the penalty is zero.

    Parameters
    ----------
    sym : str
        The trading symbol (e.g. "BTC/USDT").  The base asset is
        extracted to look up metrics in the global dictionaries.

    Returns
    -------
    float
        A penalty fraction between 0 and 1.
    """
    # Extract base asset (e.g. BTC) from a symbol like "BTC/USDT"
    try:
        base = sym.split("/")[0].upper() if "/" in sym else sym.upper()
    except Exception:
        base = sym.upper()
    penalty = 0.0
    # Options metrics: implied volatility and put/call ratio
    try:
        metrics = OPTIONS_LIVE_METRICS.get(base)
        if isinstance(metrics, dict):
            # High implied volatility (e.g. > 0.5) warrants a penalty.  Use
            # a gentle scaling: IV of 0.5 → penalty 0.1; IV of 1.0 → 0.2.
            iv = metrics.get("implied_volatility")
            if iv is not None:
                try:
                    iv_f = float(iv)
                    penalty += max(0.0, min(iv_f / 5.0, 0.3))
                except Exception:
                    pass
            # Put/call ratio significantly above 1 indicates bearish bias.
            pcr = metrics.get("put_call_ratio")
            if pcr is not None:
                try:
                    pcr_f = float(pcr)
                    if pcr_f > 1.0:
                        penalty += min((pcr_f - 1.0) * 0.1, 0.2)
                except Exception:
                    pass
    except Exception:
        pass
    # Liquidation intensity: sum of notional amounts.  Scale penalty by 50M.
    try:
        liq = LIQUIDATION_INTENSITY.get(base)
        if liq is not None:
            try:
                liq_f = float(liq)
                penalty += min(liq_f / 50_000_000.0, 0.3)
            except Exception:
                pass
        else:
            # Fallback: if no liquidation data is available, derive a penalty
            # from the on‑chain sentiment.  When sentiment < 0.5 (bearish),
            # increase the penalty up to 0.2.  A neutral sentiment yields
            # zero additional penalty.
            try:
                from onchain_analytics import get_onchain_sentiment  # type: ignore
                sent = get_onchain_sentiment(base)
                if sent is not None:
                    try:
                        s_f = float(sent)
                        if s_f < 0.5:
                            penalty += min((0.5 - s_f) * 0.4, 0.2)
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    # Whale alerts: any recent large transfers trigger a modest penalty.
    try:
        # Prefer events from the aggregated 'all' key; fall back to
        # chain‑specific keys (eth or base lowercased).  If any events
        # are present, add a small penalty to reduce confidence.
        events_all = WHALE_ALERTS.get("all")
        if not events_all:
            # fall back to ETH key
            events_all = WHALE_ALERTS.get("eth")
        if not events_all:
            # also check for base‑specific key (lowercase)
            events_all = WHALE_ALERTS.get(base.lower())
        if events_all and isinstance(events_all, list) and len(events_all) > 0:
            penalty += 0.1
    except Exception:
        pass
    # Clamp to [0, 1]
    if penalty > 1.0:
        penalty = 1.0
    return penalty


# ──────────────────────────────────────────────────────────────────────────────
# OKX init + Sembol doğrulama
# ──────────────────────────────────────────────────────────────────────────────
@retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError, Exception), tries=3, base_delay=0.8, max_delay=4.0)
def initialize_exchange():
    """OKX API bağlantısını başlatır (sandbox veya gerçek) ve sembolleri doğrular."""
    global ACTIVE_SYMBOLS

    if not OKX_API_KEY or not OKX_API_SECRET or not OKX_API_PASSPHRASE:
        raise RuntimeError("OKX API bilgileri .env dosyasında eksik!")
    # When constructing the OKX exchange, set the default market type to
    # USDT‑settled swap contracts.  This ensures that subsequent API
    # calls (including fetch_ticker and create_order) target the
    # perpetual swap markets rather than spot markets, which avoids
    # missing price/current_price issues for symbols defined in spot
    # format (e.g. "BTC/USDT").
    ex = ccxt.okx({
        'apiKey': OKX_API_KEY,
        'secret': OKX_API_SECRET,
        'password': OKX_API_PASSPHRASE,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
        },
    })
    ex.set_sandbox_mode(bool(OKX_USE_TESTNET))
    ex.load_markets()
    env = "Demo" if OKX_USE_TESTNET else "Live"
    log.info(f"OKX bağlantısı başarılı ({env}).")

    # ── Sembol / market doğrulama: sadece USDT linear SWAP olanları bırak ──
    try:
        markets = ex.markets or {}
        id_index = {}
        for m in markets.values():
            mid = m.get("id")
            if mid:
                id_index[mid] = m

        valid: list[str] = []
        invalid: list[str] = []
        filtered_out: list[str] = []

        # Build a mapping from base asset to its USDT linear swap instrument ID.
        # Some users may provide symbols in spot format (e.g. "MATIC/USDT"), but the
        # OKX API expects swap instrument IDs such as "MATIC-USDT-SWAP".  To make
        # the bot more robust, try to automatically map a spot symbol's base to
        # the corresponding swap market.  Only include markets where the quote
        # currency is USDT and the instrument is a linear swap (or contract)
        # settling in USDT (or None).
        swap_map: dict[str, str] = {}
        try:
            for mk in markets.values():
                try:
                    is_swap_mk = bool(mk.get("swap") or mk.get("contract") or mk.get("type") == "swap")
                    if not is_swap_mk:
                        continue
                    if mk.get("quote") != "USDT":
                        continue
                    if mk.get("settle") not in (None, "USDT"):
                        continue
                    base = mk.get("base")
                    mid = mk.get("id")
                    if base and mid:
                        # Preserve the first mapping for a base; this avoids
                        # overwriting if multiple contracts exist (e.g. different
                        # expiry dates).  The typical perpetual SWAP is listed
                        # first in OKX markets.
                        swap_map.setdefault(base, mid)
                except Exception:
                    continue
        except Exception:
            # If market parsing fails, leave swap_map empty and fall back to
            # normalize_okx_symbol
            swap_map = {}

        for s in TARGET_SYMBOLS:
            # Attempt to derive the instrument ID from the base asset.  This
            # allows spot symbols to be automatically converted to their swap
            # equivalents (e.g. "MATIC/USDT" → "MATIC-USDT-SWAP").
            okx_id: str | None = None
            try:
                base = s.split("/")[0]
                okx_id = swap_map.get(base)
            except Exception:
                okx_id = None
            if not okx_id:
                # Fall back to naive normalization (append "-SWAP").  This
                # typically works if TARGET_SYMBOLS already contains swap
                # formatted strings.
                okx_id = normalize_okx_symbol(s)
            # Lookup the market by instrument ID
            m = id_index.get(okx_id)
            if not m:
                invalid.append(s)
                continue

            # Sadece USDT-quoted, linear SWAP kontratları
            quote = m.get("quote")
            settle = m.get("settle")
            is_swap = bool(m.get("swap") or m.get("contract") or m.get("type") == "swap")
            if not is_swap or quote != "USDT":
                filtered_out.append(s)
                continue
            if settle not in (None, "USDT"):
                filtered_out.append(s)
                continue

            valid.append(s)

        if invalid:
            log.warning(f"[SYMBOL] OKX'te SWAP market id bulunamadı, listeden çıkarıldı: {invalid}")
        if filtered_out:
            log.warning(f"[SYMBOL] USDT-SWAP olmayan semboller listeden çıkarıldı: {filtered_out}")

        if valid:
            ACTIVE_SYMBOLS = valid
            log.info(f"[SYMBOL] {len(valid)} geçerli USDT-SWAP sembol ile devam edilecek.")
        else:
            log.error("[SYMBOL] Hiç geçerli sembol kalmadı, TARGET_SYMBOLS aynen kullanılacak.")
            ACTIVE_SYMBOLS = list(TARGET_SYMBOLS)
    except Exception as e:
        log.warning(f"[SYMBOL] Sembol doğrulama başarısız, TARGET_SYMBOLS kullanılacak: {e}")
        ACTIVE_SYMBOLS = list(TARGET_SYMBOLS)

    return ex


# ──────────────────────────────────────────────────────────────────────────────
# Analiz (her sembol için)
# ──────────────────────────────────────────────────────────────────────────────
async def _analyze_one(
    exchange,
    symbol,
    tf_main: str = "1h",
    lookback: int = 60,
    microcap_threshold: float = 0.001,
    tickers: dict | None = None,
):
    """Analyze a single symbol and ALWAYS return a dict.

    Rationale:
      - trading_loop_async builds batch_items by filtering dicts; returning None suppresses symbols.
      - We want a score for every symbol (even when skipped or data is partial).
    """
    result = {
        "symbol": symbol,
        # controller_async.decide_batch expects "price". Keep both keys.
        "price": None,
        "current_price": None,
        # controller_async expects the technical pack under "ta_pack".
        "ta_pack": {},
        "analysis": {},
        "skip_reason": None,
        "error": None,
    }

    try:
        # Retrieve the last traded price from the OKX perpetual swap market.
        # Convert the generic symbol (e.g. "BTC/USDT") into the OKX swap
        # instrument ID (e.g. "BTC-USDT-SWAP").  Use the generic call()
        # wrapper to fetch the ticker in a rate‑limit aware fashion.  If
        # the ticker or its 'last' field is missing or zero, treat the
        # price as None so downstream logic can gracefully skip the
        # symbol.
        
        # Prefer the *current loop* batched tickers (fresh) over the local cache.
        price = None
        if isinstance(tickers, dict) and tickers:
            # OKX swaps can appear as "BTC/USDT:USDT".  Try a few common keys.
            candidates = [
                symbol,
                f"{symbol}:USDT",
                normalize_okx_symbol(symbol),
            ]
            t = None
            for k in candidates:
                if k in tickers:
                    t = tickers.get(k)
                    break
            if t is None:
                # Fallback: prefix match (handles exchange-specific suffixes)
                for k, v in tickers.items():
                    if isinstance(k, str) and k.startswith(symbol):
                        t = v
                        break
            if isinstance(t, dict):
                try:
                    price = float(t.get("last") or t.get("close") or 0)
                except Exception:
                    price = None
                if price == 0:
                    price = None

        # If the batched tickers did not include this symbol, query the exchange directly.
        if price is None:
            okx_symbol = normalize_okx_symbol(symbol)
            async with _TICKER_SEM:
                ticker = await call(exchange.fetch_ticker, okx_symbol, label=f"PX-{symbol}")
            if isinstance(ticker, dict):
                try:
                    price = float(ticker.get("last") or 0)
                except Exception:
                    price = None
                if price == 0:
                    price = None

        # Keep a best-effort last price cache for other components (but never *prefer* it).
        if price is not None:
            LAST_PRICES[symbol] = price

        # Microcap / dust filter: still return a structured result (so downstream can score + log).
        if price is None:
            result["skip_reason"] = "no_price"
            return result

        # Keep both keys for downstream compatibility.
        result["current_price"] = price
        result["price"] = price
        if price < microcap_threshold:
            log.info(f"[MICROCAP_SKIP] {symbol}: price={price:.6f} < threshold={microcap_threshold} — skipping analysis")
            result["skip_reason"] = f"microcap_price<{microcap_threshold}"
            return result

        if LOG_PRICE_LINES:
            # Log both UTC and local time to avoid "stale price" confusion.
            ts_utc = datetime.now(timezone.utc)
            if _LOCAL_TZ is not None:
                ts_local = ts_utc.astimezone(_LOCAL_TZ)
                ts = f"{ts_utc.isoformat()} | local={ts_local.isoformat()}"
            else:
                ts = ts_utc.isoformat()
            if price is None:
                p_str = "None"
            elif price >= 1:
                p_str = f"{price:.2f}"
            elif price >= 0.01:
                p_str = f"{price:.4f}"
            else:
                p_str = f"{price:.6f}"
            log.info(f"{symbol}: current_price={p_str} @ {ts}")

        # Fetch multi-timeframe OHLCV in an async-safe way (ccxt.async_support).
        # The legacy sync analyzer would call exchange.fetch_ohlcv without await,
        # which silently produced empty dataframes and neutral TA.
        # [FIX] tf_fast ve tf_slow tanımları eklendi
        tf_fast = "15m"  # Hızlı timeframe
        tf_slow = "4h"   # Yavaş timeframe
        
        tfs = []
        for _tf in (tf_main, tf_fast, tf_slow):
            if _tf and _tf not in tfs:
                tfs.append(_tf)
        if not tfs:
            tfs = ["15m", "1h", "4h"]
        analysis = await get_multi_timeframe_analysis_async(exchange, symbol, timeframes=tfs, limit=lookback)
        
        # DEBUG: İlk 3 sembol için analysis içeriğini logla
        if not hasattr(_analyze_one, '_debug_count'):
            _analyze_one._debug_count = 0
        if DEBUG_ANALYZE and _analyze_one._debug_count < 3:
            log.info(f"[DEBUG_ANALYZE] {symbol}: analysis type={type(analysis)}, keys={list(analysis.keys()) if isinstance(analysis, dict) else 'N/A'}")
            _analyze_one._debug_count += 1
        
        ta_pack = analysis or {}
        # controller_async uses "ta_pack"; keep "analysis" as mirror for legacy paths.
        result["ta_pack"] = ta_pack
        result["analysis"] = ta_pack
        return result

    except Exception as e:
        # Do NOT raise; keep the pipeline running and let controller decide SKIP.
        error_msg = f"{type(e).__name__}: {e}"
        result["error"] = error_msg
        result["skip_reason"] = "analysis_exception"
        # [FIX] Exception'ı logla - sorun tespiti için kritik
        import traceback
        print(f"[ANALYZE_ERROR] {symbol}: {error_msg}")
        log.error(f"[ANALYZE_ERROR] {symbol}: {error_msg}")
        log.debug(f"[ANALYZE_TRACEBACK] {symbol}: {traceback.format_exc()}")
        return result

async def orders_worker(exchange):
    """Order kuyruğunu sırasıyla işler (async + lock güvenli)."""
    global CURRENT_BALANCE
    while True:
        sym, decision = await order_queue.get()
        # Acquire per-symbol lock to prevent concurrent entry orders for the same symbol.
        # This ensures that only one entry (and its associated exit plan) is processed at
        # any given time for a symbol, preventing duplicate or overlapping orders.
        lock = _lock_for(sym)
        await lock.acquire()
        try:
            price = float(decision["price"])
            master = float(decision.get("master_confidence", 0.0))
            lev = int(decision.get("lev", 1) or 1)
            base_decision = decision["base_decision"]  # 'long' | 'short'

            # ATR değeri (volatilite adaptasyonu için)
            atr_value = decision.get("atr")
            try:
                atr_value = float(atr_value) if atr_value is not None else None
            except Exception:
                atr_value = None

            # Güncel bakiye çek
            try:
                total_balance, free_balance = await _get_balances(exchange)
                CURRENT_BALANCE = total_balance
            except Exception:
                total_balance = CURRENT_BALANCE
                free_balance = CURRENT_BALANCE

            # --- Dinamik risk yönetimi ---
            risk_params = calculate_tiered_leverage_and_allocation(master)
            recommended_leverage = int(risk_params.get("leverage", lev))
            recommended_alloc = float(risk_params.get("wallet_allocation_percent", 0.0))

            # Eğer bu emir bir pair trading işleminden geliyorsa, risk
            # profili özel bir şekilde değerlendirilir.  Pair trading
            # stratejileri hedge özellikleri nedeniyle genellikle daha
            # düşük risk taşır ancak hem pozisyon boyutu hem de kaldıracın
            # piyasa koşullarına uyacak şekilde dinamik ayarlanması gerekir.
            try:
                meta_strategy = decision.get("meta_strategy")
                if meta_strategy == "pair_trading":
                    # Ölçekleme için sinyalin z-skorunu kullan.  Daha yüksek
                    # z-skorları, oran sapmasının büyüklüğünü gösterir ve
                    # stratejiye daha fazla sermaye ayırmayı haklı çıkarabilir.
                    try:
                        z_abs = abs(float(decision.get("zscore", 0.0)))
                    except Exception:
                        z_abs = 0.0
                    # Toplam eşleşmiş işlem çifti için kullanılacak cüzdan
                    # yüzdesi.  Temel 4% (0.04) ve z-skoru 2'nin üzerinde
                    # arttıkça her bir tam sayı için +0.02 eklenir; bu değer
                    # 6% (0.06) ile sınırlandırılır.  Böylece z=3 ise
                    # ekstra=0.02, z=5 ise ekstra=0.06 olur.  Toplam
                    # cüzdan kullanımı 10% ile sınırlıdır.
                    extra = 0.0
                    if z_abs > 2.0:
                        extra = (z_abs - 2.0) * 0.02
                        if extra > 0.06:
                            extra = 0.06
                    pair_total_alloc = 0.04 + extra
                    if pair_total_alloc > 0.10:
                        pair_total_alloc = 0.10
                    # Çift işlem iki bacak içerdiğinden, her bacak için
                    # cüzdan kullanımını yarıya bölün.  Her iki bacak için
                    # önerilen yüzdelik pay aynıdır.
                    alloc_per_leg = pair_total_alloc / 2.0
                    recommended_alloc = alloc_per_leg
                    # Fiyatı yüksek varlıklarda marjin gereksinimi artar;
                    # bu nedenle daha yüksek bir kaldıraç kullanılabilir.
                    # Fiyat eşiğine göre dinamik kaldıraç seviyelerini belirle.
                    try:
                        price_f = float(price)
                        if price_f >= 50000:
                            recommended_leverage = 5
                        elif price_f >= 20000:
                            recommended_leverage = 4
                        elif price_f >= 10000:
                            recommended_leverage = 3
                        elif price_f >= 5000:
                            recommended_leverage = 2
                        else:
                            # Düşük fiyatlı varlıklarda kaldıracı minimumda tut
                            recommended_leverage = 1
                    except Exception:
                        # Eğer fiyat okunamazsa mevcut değeri koru veya 1 olarak ayarla
                        recommended_leverage = max(1, int(recommended_leverage) if recommended_leverage else 1)
                    # Her ihtimale karşı kaldıraç aralığını 1–5 arasında sınırla
                    try:
                        recommended_leverage = int(recommended_leverage)
                    except Exception:
                        recommended_leverage = 1
                    if recommended_leverage < 1:
                        recommended_leverage = 1
                    if recommended_leverage > 5:
                        recommended_leverage = 5
            except Exception:
                # Hata durumunda cüzdan ve kaldıracı varsayılan değerlere bırak
                pass

            # Volatiliteye göre kaldıraç ve cüzdan yüzdesini ayarla
            recommended_leverage, recommended_alloc, vol_info = adjust_risk_for_volatility(
                recommended_leverage, recommended_alloc, atr_value, price
            )
            # Çok kademeli kill‑switch'in 'reduce' modu aktifse, cüzdan yüzdesini ölçekle.
            try:
                if isinstance(recommended_alloc, (int, float)):
                    recommended_alloc = float(recommended_alloc) * float(risk_reduction_factor)
            except Exception:
                pass
            if vol_info:
                try:
                    log.info(
                        f"{sym}: vol_category={vol_info.get('category')} "
                        f"atr_ratio={vol_info.get('atr_ratio'):.4f} "
                        f"risk_factor={vol_info.get('risk_factor'):.2f}"
                    )
                except Exception:
                    log.info(f"{sym}: vol_category={vol_info.get('category')} (volatilite bazlı risk ayarı uygulandı)")

            # Hesap büyüklüğüne göre cüzdan kullanımını sınırla
            wallet_allocation = apply_dynamic_wallet_cap(recommended_alloc, total_balance)
            # Cüzdan kullanımını belirli bir aralıkta sabitle.  Eğer multi
            # kill‑switch'te 'reduce' modu aktifse, bu aralığın alt ve üst
            # limitlerini de risk oranına göre ölçeklendir.
            if wallet_allocation is not None:
                try:
                    min_cap = 0.15
                    max_cap = 0.40
                    if risk_reduction_factor < 1.0:
                        min_cap *= float(risk_reduction_factor)
                        max_cap *= float(risk_reduction_factor)
                    wallet_allocation = max(min_cap, min(wallet_allocation, max_cap))
                except Exception:
                    # Hata durumunda varsayılan aralığı kullan
                    wallet_allocation = max(0.15, min(wallet_allocation, 0.40))

            # Kaldıraç tek kaynaktan: risk_params (fallback olarak eski lev)
            if recommended_leverage > 0:
                lev = recommended_leverage
            else:
                lev = max(1, lev)

            # OKX tarafında kaldıraç ayarı
            try:
                await safe_set_leverage(exchange, sym, lev)
            except Exception as _e:
                log.warning(f"{sym}: kaldıraç ayarlanamadı: {_e}")

            # [FIX] Pozisyon büyüklüğünü hesapla (FREE BALANCE üzerinden)
            size = calculate_position_size(free_balance, wallet_allocation, lev, price)

            if size <= 0:
                log.warning(f"{sym}: Yetersiz Free Balance ({free_balance:.2f}). İşlem atlandı.")
                continue

            # --- İşlem açılışını logla ---
            try:
                ai_score = decision.get("ai_score")
                tech_score = decision.get("tech_score")
                sent_score = decision.get("sent_score")
                atr_val = atr_value
                fgi_val = decision.get("fgi")
                adx_val = decision.get("adx")
                rsi_val = decision.get("rsi")
                ema_fast_val = decision.get("ema_fast")
                ema_slow_val = decision.get("ema_slow")
                # Seans adı ve piyasa rejimini belirle
                session_name = None
                try:
                    from .session_filter import get_current_session as _get_current_session  # type: ignore
                    sess = _get_current_session()
                    if sess and isinstance(sess, dict):
                        session_name = str(sess.get("name"))
                except Exception:
                    session_name = None
                regime_val = None
                try:
                    from pathlib import Path as _Path
                    import json as _json
                    reg_file = _Path(__file__).resolve().parent / "data" / "market_regime.json"
                    if reg_file.exists():
                        _txt = reg_file.read_text(encoding="utf-8")
                        _data = _json.loads(_txt) if _txt else {}
                        if isinstance(_data, dict):
                            regime_val = _data.get("REGIME")
                except Exception:
                    regime_val = None
                # Compute mid price (average of best bid/ask) at the moment of entry
                entry_mid_price: float | None = None
                try:
                    okx_sym = normalize_okx_symbol(sym)
                    ticker_mid = await call(exchange.fetch_ticker, okx_sym, label=f"MID-{sym}")
                    if ticker_mid:
                        bid_v = ticker_mid.get("bid")
                        ask_v = ticker_mid.get("ask")
                        if bid_v is not None and ask_v is not None:
                            try:
                                b_f = float(bid_v)
                                a_f = float(ask_v)
                                if b_f > 0 and a_f > 0:
                                    entry_mid_price = (b_f + a_f) / 2.0
                            except Exception:
                                entry_mid_price = None
                except Exception:
                    entry_mid_price = None
                log_trade_open(
                    symbol=sym,
                    side=base_decision,
                    entry_price=price,
                    size=size,
                    ai_score=ai_score,
                    tech_score=tech_score,
                    sent_score=sent_score,
                    master_confidence=master,
                    leverage=lev,
                    atr=atr_val,
                    fgi=fgi_val,
                    adx=adx_val,
                    rsi=rsi_val,
                    ema_fast=ema_fast_val,
                    ema_slow=ema_slow_val,
                    # Ek alanlar: RL skoru, cüzdan kullanım yüzdesi, risk_usd, tf ve base_decision
                    rl_score=decision.get("rl_score"),
                    wallet_allocation_percent=wallet_allocation,
                    risk_usd=None,
                    tf=decision.get("tf"),
                    base_decision=base_decision,
                    session_name=session_name,
                    regime=regime_val,
                    # Additional transparency fields.  These are captured
                    # from the decision dict and persisted alongside the trade.
                    ai_components=decision.get("ai_components"),
                    tech_signals_detail=decision.get("tech_signals_detail"),
                    final_weights=decision.get("final_weights"),
                    risk_veto=False,
                    # Persist the mid price at entry for slippage calculation
                    entry_mid_price=entry_mid_price,
                    # New calibration fields
                    raw_score=decision.get("raw_score"),
                    provider_flags=decision.get("provider_flags"),
                    vol_category=decision.get("vol_category"),
                )
            except Exception as _e:
                pass

            # Giriş planı (kademeli giriş miktarları)
            entries = get_entry_levels(size, master)
            # OKX minimum miktar kontrolü: bazı coinlerde minimum order boyutu 1 veya daha büyük
            # get_entry_levels fonksiyonu pozisyonu 3 parçaya bölebilir, ancak her parça
            # borsanın min_amount değerinin altında kalıyorsa emir reddedilir (örn. BNB/USDT için min=1)
            # Bu durumda, tüm pozisyonu tek parça olarak girmek daha güvenlidir.  
            try:
                okx_symbol = normalize_okx_symbol(sym)
                market_info = None
                # Exchange markets tablosundan min_amount'ı çek
                try:
                    market_info = exchange.market(okx_symbol)
                except Exception:
                    # market() fonksiyonu bazı ccxt sürümlerinde yoksa load_markets üzerinde çalış
                    markets = getattr(exchange, 'markets', None)
                    if markets and okx_symbol in markets:
                        market_info = markets[okx_symbol]
                min_amt = 0.0
                if market_info:
                    min_amt = float(
                        market_info.get('limits', {}).get('amount', {}).get('min', 0) or 0
                    )
                # Fallback minimum amounts for symbols where market_info is missing or incomplete
                # Some markets (e.g., BNB/USDT) require at least 1 unit per order, but
                # ccxt might not populate limits.amount.min correctly.  Provide manual
                # fallbacks for known symbols when min_amt is zero.  The keys should
                # match the trading symbol passed to this function (uppercase).  You
                # can extend this dictionary with other markets as needed.
                fallback_min_amounts = {
                    "BNB/USDT": 1.0,
                    "BNB-USDT-SWAP": 1.0,
                    "BNB/USDT:USDT": 1.0,
                    # add more symbol keys here if needed
                }
                if min_amt <= 0:
                    sym_key = sym.upper()
                    # try full key and normalized okx symbol
                    if sym_key in fallback_min_amounts:
                        min_amt = fallback_min_amounts[sym_key]
                    elif okx_symbol in fallback_min_amounts:
                        min_amt = fallback_min_amounts[okx_symbol]
                # Eğer minimum miktar belirlendiyse ve herhangi bir giriş parçası bu miktarın
                # altındaysa, tüm pozisyonu tek parça yap.  Ayrıca toplam pozisyon boyutu
                # borsanın minimum order miktarının altındaysa, işlem tamamen atlanacaktır.
                if min_amt > 0:
                    # Toplam pozisyon minimum miktarın altındaysa, trade'i atla
                    if size < min_amt:
                        log.info(f"{sym}: toplam boyut {size:.3f} min_amount={min_amt} altında, trade atlanıyor.")
                        # Atlamak için entries boş bırakıp bir flag ayarla
                        entries = []
                    elif any(e < min_amt for e in entries):
                        # Tek parça, mevcut boyutun 3 basamaklı yuvarlanmış hali
                        entries = [round(size, 3)]
            except Exception as _ee:
                # market bilgisi çekilemedi; entries olduğu gibi kalsın
                pass

            # Eğer hiçbir giriş yapılmayacaksa (min amount eşiği altında), bu
            # sembol için planlanan işlem atlanır.  entries boş olduğunda
            # direkt sıradaki sembole geçilir.
            if not entries:
                continue

            # --- Dinamik TP/SL hesaplama ---
            dynamic_mode = atr_value is not None and atr_value > 0

            # Tick precision
            try:
                market = exchange.market(normalize_okx_symbol(sym))
                tick_size = market.get("precision", {}).get("price")
            except Exception:
                tick_size = None

            # Kademeli çıkış planı ve stop-loss fiyatı
            if dynamic_mode:
                # Volatilite bazlı SL/TP katsayıları
                sl_mult = 1.0
                tp_mult = 2.5
                if vol_info:
                    try:
                        sl_mult = float(vol_info.get("sl_mult", sl_mult))
                        tp_mult = float(vol_info.get("tp_mult", tp_mult))
                    except Exception:
                        pass

                # GÜNCELLEME: Risk Manager üzerinden standart SL hesabı
                sl_price = compute_stop_loss(
                    side=base_decision,
                    entry_price=price,
                    atr=atr_value,
                    atr_mult=sl_mult,
                    tick_size=tick_size,
                    percent_fallback=STOP_LOSS_PERCENT,
                    last_price=None,  # exit plan içinde last kontrolü var
                )

                # Take-profit seviyeleri (kademeli)
                if base_decision == "long":
                    tp1_price = price + (atr_value * tp_mult * 0.5)
                    tp2_price = price + (atr_value * tp_mult)
                else:
                    tp1_price = price - (atr_value * tp_mult * 0.5)
                    tp2_price = price - (atr_value * tp_mult)

                tp_levels = [
                    {"price": tp1_price, "size": size * 0.5},
                    {"price": tp2_price, "size": size * 0.5},
                ]
            else:
                # Fallback: Sabit %0.5 stop-loss ve %1–5 arası dinamik take-profit
                sl_price = compute_stop_loss(
                    side=base_decision,
                    entry_price=price,
                    atr=None,
                    atr_mult=1.0,
                    tick_size=tick_size,
                    percent_fallback=STOP_LOSS_PERCENT,
                    last_price=None,
                )

                # master_confidence'a göre TP yüzdesi (1% ile 5% arası)
                try:
                    conf_norm = max(MIN_CONFIDENCE_FOR_TRADE, min(master, 1.0))
                    span = 1.0 - MIN_CONFIDENCE_FOR_TRADE
                    if span <= 0:
                        tp_pct = TP_MIN_PCT
                    else:
                        ratio = (conf_norm - MIN_CONFIDENCE_FOR_TRADE) / span
                        tp_pct = TP_MIN_PCT + ratio * (TP_MAX_PCT - TP_MIN_PCT)
                    tp_pct = max(TP_MIN_PCT, min(tp_pct, TP_MAX_PCT))
                except Exception:
                    tp_pct = TP_MIN_PCT

                if base_decision == "long":
                    tp_price = price * (1.0 + tp_pct)
                else:
                    tp_price = price * (1.0 - tp_pct)

                tp_levels = [{"price": tp_price, "size": float(size)}]

            # Tick hassasiyeti uygulanıyorsa, TP fiyatlarını yuvarla
            if tick_size:
                try:
                    for lvl in tp_levels:
                        lvl_price = lvl.get("price")
                        lvl["price"] = round(lvl_price / tick_size) * tick_size
                except Exception:
                    pass

            # Limit emirler için slipaj toleransına göre fiyat belirle
            limit_price = None
            try:
                if base_decision == "long":
                    limit_price = price * (1.0 + MAX_SLIPPAGE_PERCENT)
                else:
                    limit_price = price * (1.0 - MAX_SLIPPAGE_PERCENT)
            except Exception:
                limit_price = None

            # Trailing TP için başlangıç durumu kaydet
            try:
                last_tp_price = None
                if tp_levels:
                    last_tp_price = float(tp_levels[-1].get("price"))
                # Yeni pozisyon takibi: trailing stop için ATR tabanlı mesafe ve mevcut SL saklanır
                atr_sl_dist = None
                current_sl_val = None
                try:
                    if sl_price is not None:
                        # Fark: long için entry - sl; short için sl - entry
                        if base_decision == "long":
                            atr_sl_dist = float(price) - float(sl_price)
                        else:
                            atr_sl_dist = float(sl_price) - float(price)
                        current_sl_val = float(sl_price)
                except Exception:
                    atr_sl_dist = None
                    current_sl_val = None
                position_state[sym] = {
                    "entry_price": float(price),
                    "side": 1.0 if base_decision == "long" else -1.0,
                    "tp_price": last_tp_price,
                    "atr_sl_distance": atr_sl_dist,
                    "current_sl": current_sl_val,
                    "tick_size": tick_size,
                }
            except Exception:
                pass

            # [FIX] Atomic SL Hazırlığı: SL ve ilk TP fiyatını entry fonksiyonuna gönderiyoruz
            first_tp_price = tp_levels[0]["price"] if tp_levels else None

            # Atomic SL/TP kullanımı kontrolü. USE_ATOMIC_SLTP bayrağı
            # kapalıysa sl_price ve tp_price parametrelerini None ileterek
            # attachAlgo mekanizmasını devre dışı bırakıyoruz. Böylece SL/TP
            # emirleri yalnızca exit plan üzerinden gönderilecek.
            atomic_sl = sl_price if USE_ATOMIC_SLTP else None
            atomic_tp = first_tp_price if USE_ATOMIC_SLTP else None
            entry_result = await safe_submit_entry_plan(
                exchange=exchange, symbol=sym,
                base_side=base_decision, total_size=size,
                entry_sizes=entries, leverage=lev,
                dry_run=WRAPPER_DRY_RUN, tick_size=tick_size,
                limit_price=limit_price,
                sl_price=atomic_sl, tp_price=atomic_tp
            )

            # Çıkış emrini yalnızca giriş planı başarılı olduysa gönder.  safe_submit_entry_plan
            # ``status" alanı "ok" ise giriş emri en az bir bacağı ile gerçekleşmiştir.
            exit_result = None
            try:
                if entry_result and entry_result.get("status") == "ok":
                    exit_result = await safe_submit_exit_plan(
                        exchange=exchange, symbol=sym,
                        base_side=base_decision,
                        tp_levels=tp_levels, sl_price=sl_price,
                        dry_run=WRAPPER_DRY_RUN, tick_size=tick_size
                    )
                else:
                    log.warning(
                        f"[ENTRY_ABORT] {sym}: Entry failed, skipping TP/SL submission."
                    )
            except Exception:
                # If exit submission fails entirely, keep exit_result as None
                pass

            # --- İşlem kapanışını logla ---
            # Çıkış planı gönderildikten sonra pozisyonun gerçekten kapandığını
            # doğrulamak için bekle ve exit fiyatını borsadan al. Önceki
            # implementasyon doğrudan TP/SL hedefini logluyordu; pozisyon
            # henüz kapanmadan log_trade_close çağrılıyordu ve bu da PnL
            # raporlarının sahte görünmesine neden oluyordu.  Aşağıdaki
            # kod bir yedek çıkış fiyatı hesaplar ve ardından asenkron
            # olarak pozisyonun kapandığını doğrulayıp gerçek fiyat ile
            # log_trade_close çağırır.  Eğer bekleme sırasında bir hata
            # oluşursa, fallback fiyatla loglama yapılır.
            try:
                fallback_exit = None
                if tp_levels:
                    try:
                        fallback_exit = float(tp_levels[0].get("price"))
                    except Exception:
                        fallback_exit = None
                if fallback_exit is None and sl_price is not None:
                    try:
                        fallback_exit = float(sl_price)
                    except Exception:
                        fallback_exit = None
                if fallback_exit is None:
                    fallback_exit = price
                await wait_for_position_close_and_log(exchange, sym, fallback_exit)
            except Exception:
                # Hata durumunda fallback fiyat ile loglama yapılır
                try:
                    log_trade_close(sym, fallback_exit)
                except Exception:
                    pass

            sl_repr = f"{sl_price:.4f}"
            log.info(
                f"🧾 ORDER PLAN | {sym} | lev={lev}x | size={size:.6f} | entries={entries} | tp_levels={tp_levels} | sl_price={sl_repr}"
            )
            log.info(f"📦 ENTRY_RES={entry_result is not None} | EXIT_RES={exit_result is not None} (dry={WRAPPER_DRY_RUN})")

            # Order-level rate limit koruması
            try:
                await adaptive_order_sleep(exchange)
            except Exception:
                await asyncio.sleep(0.1)

        except Exception as e:
            log.exception(f"{sym}: order worker error: {e}")
        finally:
            # Release the symbol lock and mark the order as done
            try:
                lock.release()
            except Exception:
                pass
            order_queue.task_done()


# ──────────────────────────────────────────────────────────────────────────────
# Bekleyen emirleri izleme döngüsü (slipaj, trailing TP + ekstra koruma)
# ──────────────────────────────────────────────────────────────────────────────
async def monitor_pending_orders(exchange):
    """
    1. Askıda kalan (Zaman aşımı veya Fiyat sapması) giriş emirlerini iptal eder. [FIX 1]
    2. Kârdaki pozisyonlar için Trailing TP uygular.
    3. Zarar limiti aşılırsa hard-close yapar.
    """
    while True:
        try:
            await asyncio.sleep(30)

            if WRAPPER_DRY_RUN:
                continue

            # ─── A) ENTRY EMİR TEMİZLİĞİ (FIX 1) ───
            try:
                for sym in ACTIVE_SYMBOLS:
                    okx_sym = normalize_okx_symbol(sym)
                    orders = await call(exchange.fetch_open_orders, okx_sym, label=f"CHK_ORD_{sym}")
                    if not orders:
                        continue

                    ticker = await call(exchange.fetch_ticker, okx_sym)
                    curr_price = float(ticker['last'])

                    for o in orders:
                        # Sadece LIMIT giriş emirlerine bak (TP/SL hariç)
                        is_reduce = o.get('reduceOnly', False) or o.get('info', {}).get('reduceOnly') == 'true'
                        # Ayrıca type='limit' olmalı
                        if o['type'] == 'limit' and not is_reduce:

                            oprice = float(o['price'])
                            otime = o['timestamp']  # ms

                            # 1. Zaman Aşımı (30 dk = 1.800.000 ms)
                            if (time_module.time() * 1000) - otime > 1800000:
                                log.info(f"[TIMEOUT] {sym}: Emir 30 dakikadır dolmadı. İptal ediliyor.")
                                await call(exchange.cancel_order, o['id'], okx_sym)
                                continue

                            # 2. Fiyat Sapması (Drift > %2)
                            if curr_price > 0:
                                drift = abs(curr_price - oprice) / curr_price
                                if drift > MAX_ORDER_DISTANCE_PCT:
                                    log.info(f"[DRIFT] {sym}: Fiyat %{drift*100:.1f} uzaklaştı. Eski emir iptal.")
                                    await call(exchange.cancel_order, o['id'], okx_sym)

            except Exception as e:
                log.warning(f"Entry temizlik hatası: {e}")

            # ─── B) MEVCUT TRAILING VE HARD STOP MANTIĞI ───
            if not position_state:
                continue

            for sym, meta in list(position_state.items()):
                try:
                    side_flag = meta.get("side")
                    entry_price = meta.get("entry_price")
                    current_tp = meta.get("tp_price")
                    if not entry_price or not side_flag:
                        continue

                    okx_symbol = normalize_okx_symbol(sym)
                    try:
                        ticker = await call(exchange.fetch_ticker, okx_symbol, label=f"TICKER-{sym}")
                        last = ticker.get("last")
                        last = float(last)
                    except Exception as e:
                        log.warning(f"monitor_pending_orders fetch_ticker {sym} hata: {e}")
                        continue
                    if last <= 0:
                        continue

                    # Mevcut kâr yüzdesi
                    if side_flag > 0:
                        gain_pct = (last - entry_price) / entry_price
                        close_side = "sell"
                    else:
                        gain_pct = (entry_price - last) / entry_price
                        close_side = "buy"

                    # Pozisyon bazlı ekstra kill-switch: belirli zarar yüzdesini aştıysa marketten kapat
                    if gain_pct <= -MAX_TRADE_DRAWDOWN_PCT:
                        try:
                            positions = await call(exchange.fetch_positions, label=f"FETCH_POS-{sym}")
                        except Exception as e:
                            log.warning(f"{sym}: hard-stop pozisyon fetch hata: {e}")
                            continue

                        # GÜNCELLEME: None veya boş liste durumu için koruma
                        if not positions:
                            position_state.pop(sym, None)
                            continue

                        pos_size = None
                        for p in positions:
                            raw_sym = p.get("symbol")
                            if not raw_sym:
                                continue
                            if str(raw_sym).split(":")[0] == sym:
                                sz = p.get("contracts") or p.get("size") or p.get("positionAmt")
                                try:
                                    sz = float(sz)
                                except Exception:
                                    sz = 0.0
                                if abs(sz) > 0:
                                    pos_size = abs(sz)
                                    break

                        if not pos_size:
                            # Pozisyon yoksa state temizle
                            position_state.pop(sym, None)
                            continue

                        params = {"tdMode": "isolated", "ccy": "USDT"}
                        try:
                            await call(
                                exchange.create_order,
                                okx_symbol,
                                "market",
                                close_side,
                                pos_size,
                                None,
                                params,
                                label=f"HARD_STOP-{sym}",
                            )
                            position_state.pop(sym, None)
                            log.info(f"[HARD_STOP] {sym}: drawdown {gain_pct:.4f}, pozisyon market kapatıldı.")
                            continue
                        except Exception as e:
                            log.warning(f"[HARD_STOP_FAIL] {sym}: {e}")
                            # Devamında trailing TP mantığı çalışabilir

                    # Henüz min TP eşiği yoksa trailing yapma
                    if gain_pct <= TP_MIN_PCT:
                        continue

                    # Kârın belli oranını realize etmek için hedef TP yüzdesi
                    desired_pct = gain_pct * 0.6
                    if desired_pct < TP_MIN_PCT:
                        desired_pct = TP_MIN_PCT
                    if desired_pct > TP_MAX_PCT:
                        desired_pct = TP_MAX_PCT

                    if side_flag > 0:
                        new_tp = entry_price * (1.0 + desired_pct)
                        if new_tp <= last:
                            new_tp = last * 1.002
                        side_tp = "sell"
                    else:
                        new_tp = entry_price * (1.0 - desired_pct)
                        if new_tp >= last:
                            new_tp = last * 0.998
                        side_tp = "buy"

                    if current_tp:
                        diff_pct = abs(new_tp - current_tp) / current_tp
                        # Değişim çok küçükse emir elleme
                        if diff_pct < 0.005:
                            continue
                        # TP'yi asla geriye çekme
                        if side_flag > 0 and new_tp <= current_tp:
                            continue
                        if side_flag < 0 and new_tp >= current_tp:
                            continue

                    # Mevcut TP ve diğer limit emirlerini yönet
                    try:
                        orders = await call(exchange.fetch_open_orders, okx_symbol, label=f"OPEN_ORDERS-{sym}")
                    except Exception as e:
                        log.warning(f"monitor_pending_orders fetch_open_orders {sym} hata: {e}")
                        orders = []

                    for o in orders or []:
                        try:
                            o_price = o.get("price")
                            # Çok uzaktaki limit emirlerini iptal et (genel temizlik)
                            if o_price is not None and last > 0:
                                try:
                                    px = float(o_price)
                                    dist = abs(px - last) / last
                                except Exception:
                                    dist = 0.0
                                if dist > MAX_ORDER_DISTANCE_PCT:
                                    await call(exchange.cancel_order, o["id"], okx_symbol, label=f"ORDER_CLEAN-{sym}")
                                    log.info(f"[ORDER_CLEAN] {sym}: {px} fiyatlı limit emir last'tan çok uzak, iptal edildi.")
                                    continue

                            # TP tarafındaki mevcut limit emirlerini iptal et (yenisini gireceğiz)
                            if o.get("side") == side_tp and o.get("type") == "limit":
                                await call(exchange.cancel_order, o["id"], okx_symbol, label=f"CANCEL_TP-{sym}")
                        except Exception as e:
                            log.warning(f"{sym}: trailing TP/iptal hatası: {e}")

                    # Pozisyon boyutunu bul
                    try:
                        positions = await call(exchange.fetch_positions, label=f"FETCH_POS2-{sym}")
                    except Exception as e:
                        log.warning(f"{sym}: trailing pozisyon fetch hata: {e}")
                        continue

                    # GÜNCELLEME: None veya boş liste durumu için koruma
                    if not positions:
                        position_state.pop(sym, None)
                        continue

                    pos_size = None
                    for p in positions:
                        raw_sym = p.get("symbol")
                        if not raw_sym:
                            continue
                        if str(raw_sym).split(":")[0] == sym:
                            sz = p.get("contracts") or p.get("size") or p.get("positionAmt")
                            try:
                                sz = float(sz)
                            except Exception:
                                sz = 0.0
                            if abs(sz) > 0:
                                pos_size = abs(sz)
                                break
                    if not pos_size:
                        # Pozisyon kapanmışsa state temizle
                        position_state.pop(sym, None)
                        continue

                    params = {"tdMode": "isolated", "ccy": "USDT"}
                    try:
                        res = await call(
                            exchange.create_order,
                            okx_symbol,
                            "limit",
                            side_tp,
                            pos_size,
                            new_tp,
                            params,
                            label=f"TRAIL_TP-{sym}",
                        )
                        position_state[sym]["tp_price"] = new_tp
                        log.info(f"[TRAIL_TP] {sym}: TP güncellendi → {new_tp:.6f}")
                        # ---- TRAILING SL UPDATE ----
                        try:
                            atr_sl_dist = meta.get("atr_sl_distance")
                            current_sl = meta.get("current_sl")
                            tick_sz = meta.get("tick_size")
                            # Yalnızca geçerli değerler varsa güncelle
                            if atr_sl_dist and current_sl and isinstance(atr_sl_dist, (int, float)) and isinstance(current_sl, (int, float)):
                                new_sl = manage_trailing_stop(
                                    last,
                                    entry_price,
                                    current_sl,
                                    atr_sl_dist,
                                    is_long=(side_flag > 0)
                                )
                                # Tick hassasiyetine yuvarla
                                if tick_sz:
                                    try:
                                        new_sl = round(new_sl / tick_sz) * tick_sz
                                    except Exception:
                                        pass
                                # Sadece SL ileriyse güncelle (geri çekmez)
                                if (side_flag > 0 and new_sl > current_sl) or (side_flag < 0 and new_sl < current_sl):
                                    await safe_update_stop_loss(
                                        exchange,
                                        sym,
                                        new_sl,
                                        tick_size=tick_sz,
                                        base_side="long" if side_flag > 0 else "short",
                                    )
                                    position_state[sym]["current_sl"] = new_sl
                                    # meta referansı güncelle, böylece sonraki kontroller güncel SL'i görür
                                    meta["current_sl"] = new_sl
                                    log.info(f"[TRAIL_SL] {sym}: SL güncellendi → {new_sl:.6f}")
                        except Exception as e_sl:
                            log.warning(f"[TRAIL_SL_FAIL] {sym}: {e_sl}")
                    except Exception as e:
                        log.warning(f"[TRAIL_TP_FAIL] {sym}: {e}")

                except Exception as e:
                    log.warning(f"monitor_pending_orders iç hata {sym}: {e}")
                    continue

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning(f"monitor_pending_orders hatası: {e}")
            await asyncio.sleep(10)


# ──────────────────────────────────────────────────────────────────────────────
# OKX’ten gerçek açık pozisyonları okuma - SARMALANDI
# ──────────────────────────────────────────────────────────────────────────────
async def load_open_trades_from_okx(exchange) -> Dict[str, str]:
    """
    OKX üzerindeki gerçek açık pozisyonlardan {symbol: 'long'|'short'} sözlüğü üretir.
    Korelasyon filtresi bu sözlüğü kullanır.
    GÜNCELLEME: async def + await call(fetch_positions)
    """
    open_trades: Dict[str, str] = {}
    try:
        # GÜNCELLEME: await call ile güvenli fetch
        positions = await call(exchange.fetch_positions, label="FETCH_POS_LOOP")
    except Exception as e:
        log.warning(f"Açık pozisyonlar çekilemedi: {e}")
        return open_trades

    # GÜNCELLEME: None veya boş liste durumunda güvenli çıkış
    if not positions:
        log.warning("Açık pozisyon listesi boş veya None geldi, {} döndürülüyor.".format(open_trades))
        return open_trades

    for pos in positions:
        try:
            raw_sym = pos.get("symbol")
            if not raw_sym:
                continue

            # OKX swap sembolü genelde "ETH/USDT:USDT" gelir → "ETH/USDT"e indir
            sym = str(raw_sym).split(":")[0]
            if sym not in ACTIVE_SYMBOLS:
                continue

            size = pos.get("contracts") or pos.get("size") or pos.get("positionAmt")
            try:
                size = float(size)
            except Exception:
                size = 0.0

            if abs(size) <= 0.0:
                continue  # pozisyon yok

            side = pos.get("side")
            if not side:
                side = "long" if size > 0 else "short"

            side = str(side).lower()
            if side not in ("long", "short"):
                continue

            open_trades[sym] = side
        except Exception:
            continue

    return open_trades


# ──────────────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyon – Pozisyon Kapanışını Doğrula ve Kapatma Logu Yaz
#
# safe_submit_exit_plan çağrıldıktan sonra pozisyonun gerçekten kapandığını
# doğrulamak ve exit fiyatını borsadan almak için bu yardımcı fonksiyon
# kullanılabilir.  trade_logger.log_trade_close yalnızca pozisyon
# kapandığında ve gerçek fiyat elde edildiğinde çağrılır.  Eğer pozisyon
# belirtilen zaman dilimi içinde kapanmazsa, fallback_exit fiyatı
# kullanılır.  Bu yöntem, log kaydının doğru gerçeklikte olmasını sağlar.
async def wait_for_position_close_and_log(exchange, symbol: str, fallback_exit: float, timeout: float = 120.0, poll_interval: float = 2.0) -> None:
    """
    Wait until there is no open position for ``symbol`` on the exchange and
    then log the trade close using the latest available price.  If the
    position does not close within ``timeout`` seconds, a fallback exit
    price is used for logging.

    Args:
        exchange: ccxt exchange instance (async)
        symbol: trading symbol in bot format (e.g. "BTC/USDT")
        fallback_exit: price to use if we cannot fetch a real exit price
        timeout: maximum seconds to wait for the position to close
        poll_interval: how often (in seconds) to check the position status
    """
    start_ts = time_module.monotonic()
    # Wait until the position is closed or timeout expires
    while True:
        elapsed = time_module.monotonic() - start_ts
        if elapsed > timeout:
            break
        try:
            # fetch open positions for the symbol
            positions = await call(exchange.fetch_positions, label=f"CLS-POS-{symbol}")
            pos_exists = False
            for p in positions or []:
                try:
                    raw_sym = p.get("symbol")
                    if not raw_sym:
                        continue
                    sym_clean = str(raw_sym).split(":")[0]
                    if sym_clean != symbol:
                        continue
                    sz = p.get("contracts") or p.get("size") or p.get("positionAmt")
                    if sz is None:
                        continue
                    try:
                        sz_f = float(sz)
                    except Exception:
                        sz_f = 0.0
                    if abs(sz_f) > 0:
                        pos_exists = True
                        break
                except Exception:
                    continue
            if not pos_exists:
                break
        except Exception:
            # On any error, wait and retry
            pass
        await asyncio.sleep(poll_interval)

    # ----------------------------------------------------------------------
    # Improved exit metrics: compute VWAP, total fee and realised slippage.
    # Instead of relying on a single last fill or ticker price, we fetch
    # all fills executed since the trade was opened and compute a
    # volume‑weighted average price (VWAP).  We also sum the fees across
    # those fills.  Slippage is calculated against the mid price observed
    # at entry time.
    # ----------------------------------------------------------------------
    exit_price: float | None = None
    fee_val: float | None = None
    slippage_val: float | None = None
    try:
        import trade_logger  # type: ignore
        from trade_metrics import compute_vwap_and_fee, calculate_slippage  # type: ignore
        # Retrieve the open trade record to determine when it was opened
        record = trade_logger._open_trades.get(symbol)
        since_ts: int | None = None
        entry_side: str | None = None
        entry_mid: float | None = None
        if record:
            try:
                ts_open = record.get("timestamp_open")
                # Convert ISO8601 to unix ms; handle Z suffix and timezone aware strings
                if ts_open:
                    # Remove trailing Z if present and parse
                    ts_s = str(ts_open).replace("Z", "");
                    dt = datetime.fromisoformat(ts_s)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    since_ts = int(dt.timestamp() * 1000) - 1000  # subtract 1s buffer
            except Exception:
                since_ts = None
            try:
                entry_side = record.get("side")
            except Exception:
                entry_side = None
            try:
                # entry_mid_price stored during log_trade_open via extra_fields
                entry_mid = record.get("entry_mid_price")
            except Exception:
                entry_mid = None
        okx_symbol = normalize_okx_symbol(symbol)
        # Compute vwap and fee across fills since trade open
        vwap, total_fee, last_side = await compute_vwap_and_fee(exchange, okx_symbol, since_ts)
        if vwap is not None:
            exit_price = float(vwap)
            fee_val = total_fee
            # Use entry_side if available; fallback to last_side from fills
            side_for_slip = entry_side or last_side
            slippage_val = calculate_slippage(entry_mid, vwap, side_for_slip)
        else:
            # Fall back to previous logic if vwap is unavailable
            # Try ticker for exit price
            try:
                ticker = await call(exchange.fetch_ticker, okx_symbol, label=f"CLS-TICKER-{symbol}")
                if ticker:
                    ep = ticker.get("last") or ticker.get("close")
                    exit_price = float(ep) if ep is not None else None
            except Exception:
                exit_price = None
            # If still missing, use fallback_exit
            if exit_price is None:
                exit_price = fallback_exit
            # Fee: none in fallback
            fee_val = None
            # Slippage: approximate using entry_mid and exit_price
            if record and entry_mid is not None and exit_price is not None:
                slippage_val = calculate_slippage(entry_mid, exit_price, entry_side)
    except Exception:
        # In case of any unexpected error, revert to simplistic exit logic
        try:
            okx_symbol = normalize_okx_symbol(symbol)
            ticker = await call(exchange.fetch_ticker, okx_symbol, label=f"CLS-TICKER-{symbol}")
            if ticker:
                ep = ticker.get("last") or ticker.get("close")
                exit_price = float(ep) if ep is not None else None
        except Exception:
            exit_price = None
        if exit_price is None:
            exit_price = fallback_exit
        fee_val = None
        slippage_val = None
    # Finally log trade closure using computed metrics
    try:
        if exit_price is not None:
            log_trade_close(symbol, float(exit_price), fee=fee_val, slippage_pct=slippage_val)
        else:
            log_trade_close(symbol, fallback_exit, fee=fee_val, slippage_pct=slippage_val)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Ana Döngü
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Open-Position Protection: Ensure TP/SL Exists (ReduceOnly)
# ──────────────────────────────────────────────────────────────────────────────
async def ensure_tp_sl_for_open_positions(exchange) -> None:
    """Ensure each open position has reduceOnly TP and SL on the exchange."""
    # If an existing exit order deviates beyond this relative threshold, refresh it.
    UPDATE_THRESHOLD = 0.005  # 0.5%

    try:
        # OKX returns different position sets depending on instType/type.
        # Prefer SWAP positions for this bot.
        positions = await call(
            exchange.fetch_positions,
            {"type": "swap", "instType": "SWAP"},
            label="POS-PROTECT",
        )
    except Exception:
        positions = []

    if not positions:
        return

    for pos in positions:
        try:
            raw_sym = pos.get("symbol") or (pos.get("info") or {}).get("instId")
            if not raw_sym:
                continue
            sym_clean = str(raw_sym).split(":")[0]
            # Determine size
            sz = pos.get("contracts")
            if sz is None:
                sz = pos.get("size")
            if sz is None:
                sz = pos.get("positionAmt")
            try:
                sz_f = float(sz)
            except Exception:
                sz_f = 0.0
            if abs(sz_f) <= 0:
                continue

            base_side = "long" if sz_f > 0 else "short"
            qty = abs(sz_f)

            # Entry price (fallback to current last)
            entry = pos.get("entryPrice") or pos.get("avgPrice") or (pos.get("info") or {}).get("avgPx")
            try:
                entry_f = float(entry) if entry is not None else None
            except Exception:
                entry_f = None

            okx_symbol = normalize_okx_symbol(sym_clean)
            last_price = None
            try:
                t = await call(exchange.fetch_ticker, okx_symbol, label=f"PROTECT-TICK-{sym_clean}")
                if isinstance(t, dict):
                    lp = t.get("last") or t.get("close")
                    if lp is not None:
                        last_price = float(lp)
            except Exception:
                last_price = None

            if entry_f is None:
                if last_price is None:
                    continue
                entry_f = float(last_price)

            # Compute target exit prices
            sl_pct = float(STOP_LOSS_PERCENT)
            tp_pct = float(TP_MIN_PCT)
            if base_side == "long":
                target_sl = entry_f * (1.0 - sl_pct)
                target_tp = entry_f * (1.0 + tp_pct)
                tp_side = "sell"
            else:
                target_sl = entry_f * (1.0 + sl_pct)
                target_tp = entry_f * (1.0 - tp_pct)
                tp_side = "buy"

            # Inspect open orders to see if TP/SL exists
            try:
                open_orders = await call(exchange.fetch_open_orders, okx_symbol, label=f"PROTECT-ORD-{sym_clean}") or []
            except Exception:
                open_orders = []

            tp_order = None
            sl_order = None
            for o in open_orders:
                try:
                    info = o.get("info") or {}
                    o_type = o.get("type")
                    o_side = o.get("side")
                    reduce_only = bool(o.get("reduceOnly") or info.get("reduceOnly"))

                    # TP: limit, opposite side, reduceOnly
                    if o_type == "limit" and o_side == tp_side and reduce_only:
                        tp_order = o

                    # SL: stopLossPrice/slTriggerPx present and reduceOnly
                    sl_px = info.get("stopLossPrice") or info.get("slTriggerPx") or info.get("stopPx")
                    if sl_px and reduce_only:
                        sl_order = o
                except Exception:
                    continue

            def _needs_update(existing_price: float | None, target_price: float) -> bool:
                if existing_price is None:
                    return True
                try:
                    existing_price = float(existing_price)
                    target_price = float(target_price)
                    if target_price <= 0:
                        return True
                    return abs(existing_price - target_price) / target_price > UPDATE_THRESHOLD
                except Exception:
                    return True

            # Cancel outdated TP/SL before recreate
            cancel_ids = []
            if tp_order is not None:
                ex_tp_price = tp_order.get("price")
                if _needs_update(ex_tp_price, target_tp):
                    cancel_ids.append(tp_order.get("id"))
                    tp_order = None
            if sl_order is not None:
                ex_sl_price = (sl_order.get("info") or {}).get("stopLossPrice") or (sl_order.get("info") or {}).get("slTriggerPx")
                if _needs_update(ex_sl_price, target_sl):
                    cancel_ids.append(sl_order.get("id"))
                    sl_order = None

            for oid in cancel_ids:
                if not oid:
                    continue
                try:
                    await call(exchange.cancel_order, oid, okx_symbol, label=f"PROTECT-CANCEL-{sym_clean}")
                except Exception:
                    pass

            # If either TP or SL is missing, create them via safe_submit_exit_plan
            if tp_order is None or sl_order is None:
                tp_levels = [{"price": float(target_tp), "size": float(qty)}]
                await safe_submit_exit_plan(
                    exchange,
                    sym_clean,
                    base_side,
                    tp_levels=tp_levels,
                    sl_price=float(target_sl),
                    dry_run=bool(WRAPPER_DRY_RUN),
                )

        except Exception:
            continue

async def trading_loop_async(exchange):
    """Ana asenkron ticaret döngüsü."""
    global CURRENT_BALANCE
    try:
        corr_df = calculate_correlation(exchange, ACTIVE_SYMBOLS)
        log.info(f"Korelasyon matrisi hazır: {corr_df.shape}")
    except Exception as e:
        log.warning(f"Korelasyon hesaplanamadı: {e}")
        corr_df = None

    balance = BALANCE_START
    killer = KILLER
    # Launch order workers and pending order monitors as background tasks
    worker_task = create_bg_task(orders_worker(exchange))
    monitor_task = create_bg_task(monitor_pending_orders(exchange))
    hybrid_weights = _load_hybrid_weights()

    while True:
        if is_killed():
            log.error("KILL flag tespit edildi. Güvenli kapanış.")
            # Cancel all background tasks and close the exchange to ensure a clean shutdown
            try:
                await _cancel_bg_tasks()
            except Exception:
                pass
            try:
                if exchange is not None:
                    await exchange.close()
            except Exception:
                pass
            break

        # Gerçek bakiye güncelle (artık sarmalanmış async fonksiyon)
        try:
            CURRENT_BALANCE = await _get_real_balance(exchange)
            balance = CURRENT_BALANCE
        except Exception:
            balance = CURRENT_BALANCE

        pause_loop_sleep(10)
        log.info(f"=== Döngü başlangıcı @ {datetime.now():%Y-%m-%d %H:%M:%S} | Balance: ${balance:.2f} ===")

        # Her tur başında OKX’ten gerçek açık pozisyonları çek (artık async)
        real_open_trades = await load_open_trades_from_okx(exchange)

        # Ensure existing open positions always have TP/SL protection on the exchange
        try:
            await ensure_tp_sl_for_open_positions(exchange)
        except Exception:
            pass

        # Update runtime metrics (unrealized PnL, exposure, expectancy) and
        # evaluate alarms.  Any exceptions inside metrics logic are
        # swallowed to avoid disrupting trading.  By measuring these
        # values at the beginning of each loop we ensure the dashboard
        # remains current and alarms fire promptly when thresholds are
        # breached.
        try:
            await update_from_exchange(exchange)
            check_alerts()
        except Exception:
            pass
        # Bu turda açmayı planladığın yeni işlemleri de eklemek için kopya
        planned_open_trades: Dict[str, str] = dict(real_open_trades)

        # Önce çok kademeli kill‑switch'i kontrol et (varsa)
        kill_action: str = "normal"
        if KILLER_MULTI is not None:
            try:
                kill_action = KILLER_MULTI.check(balance)
            except Exception:
                kill_action = "normal"
        # 'stop' aksiyonu → botu tamamen durdur
        if kill_action == "stop":
            try:
                from state_manager import request_kill  # type: ignore
                request_kill()
            except Exception:
                pass
            log.error("Multi-level kill-switch STOP tetiklendi. Bot durduruluyor.")
            # Cancel background tasks and close exchange on stop
            try:
                await _cancel_bg_tasks()
            except Exception:
                pass
            try:
                if exchange is not None:
                    await exchange.close()
            except Exception:
                pass
            break
        # 'halt' aksiyonu → yeni giriş yok, yalnızca mevcut pozisyonlar yönetilir
        kill_halt = kill_action == "halt"
        # 'reduce' aksiyonu → risk yarıya düşürülür (wallet_allocation çarpanı)
        risk_reduction_factor = 0.5 if kill_action == "reduce" else 1.0
        # Kullanıcı risk faktörü ile ölçekle
        try:
            risk_reduction_factor *= float(RISK_FACTOR_USER)
        except Exception:
            pass

        # Canary model risk reduction.  If a new BiLSTM model is in
        # canary mode (i.e. recently published with the 'canary' flag),
        # apply an additional multiplier to reduce exposure.  The
        # `_get_canary_multiplier` function reads the model update
        # manifest and computes whether the canary period is active.  A
        # multiplier < 1.0 reduces the total wallet allocation for
        # entries and prevents over‑exposure while validating the new model.
        try:
            can_mult = _get_canary_multiplier()
            if can_mult < 1.0:
                risk_reduction_factor *= can_mult
                log.info(
                    f"[CANARY] New BiLSTM model in canary mode: risk multiplier {can_mult:.2f}. "
                    f"Effective risk reduction factor now {risk_reduction_factor:.2f}"
                )
        except Exception:
            pass

        # Günlük kill switch (eski sistem) kontrolü. Eğer kill aktifse, bekle.
        if killer.check_switch(balance):
            log.warning("Kill-switch aktif. 900 sn bekleniyor...")
            await asyncio.sleep(900)
            continue

        # Circuit breaker: If recent performance is poor, halt new trades.  The
        # circuit breaker works alongside the multi‑level kill switch.  If
        # kill_action already demands 'halt' or 'stop', we respect that.
        try:
            if kill_action not in ("halt", "stop") and circuit_should_halt(balance):
                log.warning("Circuit breaker aktif: yeni girişler durduruluyor.")
                kill_halt = True
        except Exception:
            pass

        # Eğer bot duraklatıldıysa, bu turu atla ve nedenini bildir
        if BOT_PAUSED:
            # Use the recorded pause reason if available
            reason = PAUSE_REASON or "unknown reason"
            log.info(f"Bot paused: {reason}. Sleeping 30 seconds...")
            await asyncio.sleep(30)
            continue
        # Karaliste kontrolü: karalistede olan sembolleri atla
        blacklist = _load_blacklist()
        symbols_to_analyze = [s for s in ACTIVE_SYMBOLS if s.upper() not in blacklist]
        # Optionally limit the number of symbols to analyse per loop to reduce
        # concurrency and CPU usage.  If MAX_SYMBOLS_TO_ANALYZE is 0, analyse all symbols.
        try:
            if MAX_SYMBOLS_TO_ANALYZE > 0:
                symbols_to_analyze = symbols_to_analyze[:MAX_SYMBOLS_TO_ANALYZE]
        except Exception:
            pass
        
        # Clear per-loop price cache to avoid stale prices
        try:
            LAST_PRICES.clear()
        except Exception:
            LAST_PRICES = {}

        # Force a markets reload once per loop to reduce stale instrument
        # metadata (price limits, tick size) and to improve symbol resolution.
        # OKX demo/live sometimes updates limits; keeping this fresh helps TP/SL.
        try:
            await call(exchange.load_markets, True, label="LOAD_MARKETS")
        except Exception:
            pass

        # Fetch OKX server time once per loop to demonstrate freshness and
        # detect clock drift (helps users interpret UTC vs local timestamps).
        try:
            server_ms = await call(exchange.fetch_time, label="FETCH_TIME")
            server_dt = datetime.fromtimestamp(float(server_ms) / 1000.0, tz=timezone.utc)
            if _LOCAL_TZ is not None:
                local_dt = server_dt.astimezone(_LOCAL_TZ)
                drift_ms = int((datetime.now(timezone.utc) - server_dt).total_seconds() * 1000)
                log.info(f"[OKX_TIME] server={server_dt.isoformat()} | local={local_dt.isoformat()} | drift_ms={drift_ms}")
            else:
                log.info(f"[OKX_TIME] server={server_dt.isoformat()}")
        except Exception:
            pass

        # ─────────────────────────────────────────────────────────────────
        # BATCH TICKER: Rate limit çözümü
        # Tek API çağrısıyla tüm fiyatları çek, cache'e kaydet
        # ─────────────────────────────────────────────────────────────────
        all_tickers = None
        try:
            all_tickers = await call(exchange.fetch_tickers, label="BATCH_TICKERS")
            if all_tickers:
                for sym_key, ticker_data in all_tickers.items():
                    try:
                        # OKX format: "BTC/USDT:USDT" -> "BTC/USDT"
                        clean_sym = sym_key.split(":")[0] if ":" in sym_key else sym_key
                        last_price = ticker_data.get("last")
                        if last_price and float(last_price) > 0:
                            LAST_PRICES[clean_sym] = float(last_price)
                    except Exception:
                        pass
                log.info(f"[BATCH_TICKER] {len(LAST_PRICES)} sembol fiyatı cache'lendi")
        except Exception as batch_err:
            log.warning(f"[BATCH_TICKER] Batch ticker başarısız, tekil çekime devam: {batch_err}")
        
        # Analyze every symbol and then adapt the raw analysis payload into the schema
        # expected by controller_async.decide_batch (price/ta_pack/senti/etc.).
        tf_main = os.getenv("TF_MAIN", "1h")
        lookback = int(os.getenv("LOOKBACK", "60"))

        async def _limited_analyze(sym: str):
            # Limit per-loop analysis concurrency to avoid hammering OKX endpoints
            # (ticker/ohlcv) and to reduce CPU bursts.
            async with _ANALYZE_SEM:
                return await _analyze_one(exchange, sym, tf_main=tf_main, lookback=lookback, tickers=all_tickers)

        anal_tasks = [_limited_analyze(s) for s in symbols_to_analyze]
        anal_res = await asyncio.gather(*anal_tasks, return_exceptions=True)

        def _safe_last(df, col, default=None):
            try:
                if df is None or df.empty or col not in df.columns:
                    return default
                v = df[col].iloc[-1]
                # Convert numpy scalars
                if hasattr(v, "item"):
                    v = v.item()
                return v
            except Exception:
                return default

        def _build_ta_pack_from_multidata(multi_data: dict, tf_main_local: str) -> dict:
            # multi_data is a dict like {"1h": DataFrame, "4h": DataFrame, ...}
            # [FIX] DataFrame için 'or' kullanılamaz - None kontrolü yapıyoruz
            def _get_df(d, key):
                """DataFrame için güvenli get - None kontrolü yapar."""
                v = d.get(key)
                if v is None:
                    return None
                return v
            
            df_main = _get_df(multi_data, tf_main_local)
            if df_main is None:
                df_main = _get_df(multi_data, tf_main_local.lower())
            
            df_h1 = _get_df(multi_data, "1h")
            if df_h1 is None:
                df_h1 = df_main
                
            df_h4 = _get_df(multi_data, "4h")
            if df_h4 is None:
                df_h4 = df_h1
            # DEBUG: multi_data içeriğini logla
            if DEBUG_TA:
                if multi_data:
                    log.info(f"[DEBUG_TA] multi_data keys: {list(multi_data.keys())[:5]}")
                    if df_main is not None and hasattr(df_main, 'columns'):
                        log.info(f"[DEBUG_TA] df_main columns: {list(df_main.columns)[:10]}")
                    elif df_main is not None:
                        log.info(
                            f"[DEBUG_TA] df_main type: {type(df_main)}, keys: {list(df_main.keys())[:10] if isinstance(df_main, dict) else 'N/A'}"
                        )
                    else:
                        log.warning(f"[DEBUG_TA] df_main is None for tf={tf_main_local}")
                else:
                    log.warning(f"[DEBUG_TA] multi_data is empty or None")

            # The analyzer module may emit indicator columns in multiple naming
            # conventions (lowercase vs uppercase, and with timeframe suffixes
            # like "RSI_1h", "EMA_20_1h", "MACD_Hist_4h", etc.).  Build a
            # resilient mapping so controller_async receives a non-empty ta_pack.
            tf = str(tf_main_local).lower()


            def _pick(df, *candidates, default=None):
                for c in candidates:
                    v = _safe_last(df, c, default=None)
                    if v is not None:
                        return v
                return default


            def _sanitize_num(v):
                """Convert NaN/inf-like values to None to keep downstream logic stable."""
                try:
                    if v is None:
                        return None
                    fv = float(v)
                    if math.isnan(fv) or math.isinf(fv):
                        return None
                    return fv
                except Exception:
                    return v

            # Common column candidates
            rsi = _pick(df_main, "rsi", f"RSI_{tf}", f"rsi_{tf}")
            adx = _pick(df_main, "adx", f"ADX_{tf}", f"adx_{tf}")
            # ATR ratio / NATR (Normalized ATR) naming varies across libs
            atr_ratio = _pick(df_main, "atr_ratio", f"ATR_Ratio_{tf}", f"atr_ratio_{tf}", f"NATR_{tf}")
            vol_z = _pick(df_main, "vol_z", f"VOL_Z_{tf}", f"vol_z_{tf}")
            trend_strength = _pick(df_main, "trend_strength", f"TrendStrength_{tf}", f"TREND_STRENGTH_{tf}")

            # EMA columns: prefer explicit fast/slow if present; else fall back to EMA_20/EMA_50.
            ema_fast = _pick(df_main, "ema_fast", f"EMA_FAST_{tf}", f"EMA_20_{tf}", f"EMA20_{tf}")
            ema_slow = _pick(df_main, "ema_slow", f"EMA_SLOW_{tf}", f"EMA_50_{tf}", f"EMA50_{tf}")

            macd_h1 = _sanitize_num(_pick(df_h1, "macd_hist", f"MACD_Hist_1h", f"MACD_HIST_1h", f"macd_hist_1h"))
            macd_h4 = _sanitize_num(_pick(df_h4, "macd_hist", f"MACD_Hist_4h", f"MACD_HIST_4h", f"macd_hist_4h"))

            # base_decision'ı EMA ve MACD'den hesapla
            base_decision = "neutral"
            
            # Önce EMA'dan dene
            if ema_fast is not None and ema_slow is not None:
                try:
                    if float(ema_fast) > float(ema_slow):
                        base_decision = "long"
                    elif float(ema_fast) < float(ema_slow):
                        base_decision = "short"
                except Exception:
                    pass
            
            # EMA yoksa MACD histogram'dan dene
            if base_decision == "neutral":
                # 1h ve 4h MACD'nin ikisi de aynı yönde ise
                if macd_h1 is not None and macd_h4 is not None:
                    try:
                        if float(macd_h1) > 0 and float(macd_h4) > 0:
                            base_decision = "long"
                        elif float(macd_h1) < 0 and float(macd_h4) < 0:
                            base_decision = "short"
                    except Exception:
                        pass
                # Sadece 1h MACD varsa
                elif macd_h1 is not None:
                    try:
                        if float(macd_h1) > 0:
                            base_decision = "long"
                        elif float(macd_h1) < 0:
                            base_decision = "short"
                    except Exception:
                        pass
            
            # RSI extreme değerlerinde yön belirle (son çare)
            if base_decision == "neutral" and rsi is not None:
                try:
                    rsi_val = float(rsi)
                    if rsi_val < 30:
                        base_decision = "long"  # Oversold = potansiyel long
                    elif rsi_val > 70:
                        base_decision = "short"  # Overbought = potansiyel short
                except Exception:
                    pass
            
            # DEBUG: Hesaplanan değerleri logla (sadece ilk 3 coin için)
            if not hasattr(_build_ta_pack_from_multidata, '_debug_count'):
                _build_ta_pack_from_multidata._debug_count = 0
            if _build_ta_pack_from_multidata._debug_count < 3:
                log.info(f"[DEBUG_TA] ema_fast={ema_fast}, ema_slow={ema_slow}, macd_h1={macd_h1}, macd_h4={macd_h4}, rsi={rsi} -> base_decision={base_decision}")
                _build_ta_pack_from_multidata._debug_count += 1
            
            # Small recent windows (kept short to control prompt size)
            recent_closes = None
            recent_volumes = None
            lookback_bars = None
            try:
                if df_main is not None:
                    lookback_bars = int(len(df_main))
                    if hasattr(df_main, "__getitem__"):
                        if "close" in df_main.columns:
                            recent_closes = [float(x) for x in df_main["close"].tail(10).tolist()]
                        if "volume" in df_main.columns:
                            recent_volumes = [float(x) for x in df_main["volume"].tail(10).tolist()]
            except Exception:
                pass

            return {
                "rsi": rsi,
                "adx": adx,
                "atr_ratio": atr_ratio,
                "vol_z": vol_z,
                "trend_strength": trend_strength,
                "ema": {
                    "fast": ema_fast,
                    "slow": ema_slow,
                },
                "trend": {
                    "h1_macd_hist": macd_h1,
                    "h4_macd_hist": macd_h4,
                },
                "recent_closes": recent_closes,
                "recent_volumes": recent_volumes,
                "lookback_bars": lookback_bars,
                "base_decision": base_decision,  # YENİ: base_decision eklendi
            }

        batch_items = []
        for r in anal_res:
            if not isinstance(r, dict):
                continue
            sym = r.get("symbol")
            price = r.get("current_price")
            multi_data = r.get("analysis") or {}
            ta_pack = _build_ta_pack_from_multidata(multi_data, tf_main)
            batch_items.append(
                {
                    "symbol": sym,
                    # controller_async historically expects `price`.
                    "price": price,
                    "current_price": price,
                    "ta_pack": ta_pack,
                    "senti": {},
                    "anomaly": 0,
                    "imbalance": r.get("imbalance"),
                    "skip_reason": r.get("skip_reason"),
                    "error": r.get("error"),
                }
            )
        # Notify the circuit breaker about price anomalies.  The anomaly
        # flag is set by `_analyze_one` when prices jump beyond ATR
        # bounds.  Each flagged item increments an internal counter in
        # the circuit breaker; exceeding the configured threshold will
        # set a cooldown during which new trades are suppressed.
        try:
            for item in batch_items:
                if item.get("anomaly"):
                    record_anomaly()
        except Exception:
            pass
        if not batch_items:
            log.info("Analiz yok veya tüm semboller karalistede, uyku...")
            await asyncio.sleep(30)
            continue

        # If circuit breaker is active (kill_halt=True), skip generating new imbalance metrics
        # and mark all items to have zero master confidence later.  Otherwise, precompute
        # order book imbalance for each symbol asynchronously.  The imbalance is stored
        # in the item dict so controller_async can adjust confidence accordingly.
        if not kill_halt:
            try:
                # Launch imbalance fetch tasks
                imb_tasks = [calculate_imbalance(exchange, item["symbol"]) for item in batch_items]
                imbalances = await asyncio.gather(*imb_tasks, return_exceptions=True)
                for idx, item in enumerate(batch_items):
                    imb_res = imbalances[idx]
                    if isinstance(imb_res, Exception):
                        continue
                    item["imbalance"] = imb_res
            except Exception:
                # If imbalance computation fails, proceed without setting imbalance
                pass
        else:
            # When halted by circuit breaker, mark items to skip trades
            for item in batch_items:
                item["imbalance"] = None

        # GPT/DeepSeek kararları (baz master_confidence içerir)
        decisions = await decide_batch(batch_items)
        log.info(f"[AI] Master score computed for {len(decisions)} symbols (from {len(batch_items)} analyzed items)")
        # Persist master scores for dashboard/debug
        try:
            import json
            from pathlib import Path
            md = Path('metrics')
            md.mkdir(parents=True, exist_ok=True)
            payload = {
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'n_symbols': len(decisions),
                'decisions': decisions,
            }
            (md / 'master_scores_latest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            log.warning(f"[AI] Could not write metrics/master_scores_latest.json: {e}")

        # Meta strateji entegrasyonu: her sembol için master confidence'i
        # meta stratejiye göre hafifçe ölçekle.  Trend stratejisinde
        # master %5 artırılır; mean_reversion'da %5 azaltılır.  Değerler
        # 0–1 aralığında sınırlandırılır.  Bu ayarlama controller
        # katmanını değiştirmeden çalışır.
        try:
            for item in batch_items:
                sym = item.get("symbol")
                meta = item.get("meta_strategy")
                dec = decisions.get(sym)
                if not dec:
                    continue
                if meta == "trend":
                    # Trend stratejisinde güveni hafif artırın
                    mc = float(dec.get("master_confidence", 0.0))
                    mc *= 1.05
                    dec["master_confidence"] = min(1.0, max(0.0, mc))
                elif meta == "mean_reversion":
                    # Mean‑reversion stratejisinde güveni biraz azaltın
                    mc = float(dec.get("master_confidence", 0.0))
                    mc *= 0.95
                    dec["master_confidence"] = min(1.0, max(0.0, mc))
                elif meta == "breakout":
                    # Breakout durumunda daha agresif artırma
                    mc = float(dec.get("master_confidence", 0.0))
                    mc *= 1.10
                    dec["master_confidence"] = min(1.0, max(0.0, mc))
        except Exception:
            pass

        # Bu loop için sayaçlar (dashboard için)
        entries_sent = 0
        enter_signals = 0

        # Bu loop'ta LLM teknik-only moda alındıysa hibrit LLM ağırlıklarını 0'a çek
        try:
            tech_only_mode = is_llm_tech_only_mode()
        except Exception:
            tech_only_mode = False

        if tech_only_mode:
            loop_hybrid_weights = dict(hybrid_weights)
            if loop_hybrid_weights.get("chatgpt", 0.0) != 0.0 or loop_hybrid_weights.get("deepseek", 0.0) != 0.0:
                loop_hybrid_weights["chatgpt"] = 0.0
                loop_hybrid_weights["deepseek"] = 0.0
                log.info(
                    "[AI_SAFE] Bu turda LLM sinyalleri devre dışı. "
                    "Hibrit confidence füzyonunda ChatGPT/DeepSeek ağırlıkları 0'a çekildi."
                )
        else:
            loop_hybrid_weights = hybrid_weights

        # Her sembol için BiLSTM/RL/LLM tahminlerini okuyup güven skoruna entegre et
        for item in batch_items:
            sym = item["symbol"]
            dec = decisions.get(sym)
            if not dec:
                continue

            # Baz güven skoru
            base_conf = float(dec.get("master_confidence", 0.0))

            # AI skorları
            ai_scores = _get_latest_ai_scores(sym)
            bilstm_c = ai_scores.get("bilstm")
            rl_c = ai_scores.get("ppo_rl")
            chatgpt_c = ai_scores.get("chatgpt")
            deepseek_c = ai_scores.get("deepseek")

            # Füzyon
            # Hibrit ağırlıkları dinamik olarak ayarla (model performansına göre)
            dynamic_weights = _get_dynamic_hybrid_weights(loop_hybrid_weights)
            fused_conf = _fuse_confidence(
                base_conf,
                bilstm_c,
                rl_c,
                chatgpt_c,
                deepseek_c,
                dynamic_weights
            )
            dec["master_confidence"] = fused_conf  # karara yaz
            # Risk penalty based on real‑time metrics (options, liquidation, whales)
            # Compute a penalty factor in [0, 1] to reduce confidence for this symbol.
            try:
                penalty = _compute_risk_penalty(sym)
                # Apply penalty to fused confidence.  If penalty is 0.2, retain 80% of the score.
                fused_conf_adj = fused_conf * (1.0 - penalty)
                dec["master_confidence"] = fused_conf_adj
                dec["risk_penalty"] = penalty
            except Exception:
                # In case of any error computing risk, keep original confidence.
                pass

            # --- Decision transparency ---
            # Attach detailed AI component scores, technical signal pack
            # and the final weight allocation to the decision dict.  These
            # fields are persisted alongside the trade when it is opened
            # (via trade_logger.log_trade_open).  They provide context on
            # why the bot entered a position: the contributions of each
            # model (BiLSTM, RL, ChatGPT/DeepSeek), the underlying
            # technical indicators at the time of decision, and the
            # dynamically computed hybrid weights.
            try:
                dec["ai_components"] = {
                    "bilstm": bilstm_c,
                    "rl": rl_c,
                    "chatgpt": chatgpt_c,
                    "deepseek": deepseek_c,
                }
            except Exception:
                pass
            try:
                # Provide the raw technical indicators used for this symbol
                dec["tech_signals_detail"] = item.get("ta_pack")
            except Exception:
                pass
            try:
                dec["final_weights"] = dynamic_weights
            except Exception:
                pass

            action = dec.get("action")

            # Sadece 'enter' sinyalleri için sayaç tut
            if action == "enter":
                enter_signals += 1

                # Eşik altına düşerse 'enter' olsa bile atla
                if fused_conf < MIN_CONFIDENCE_FOR_TRADE:
                    log.info(f"{sym}: enter sinyali füzyon sonrası eşiğin altında ({fused_conf:.2f} < {MIN_CONFIDENCE_FOR_TRADE}), atlandı.")
                    continue

                # Çok kademeli kill-switch 'halt' modundaysa yeni giriş yapma
                if kill_halt:
                    log.info(f"{sym}: kill-switch HALT modunda, yeni giriş atlandı.")
                    continue

                # Performans analizi sonucu sembol cooldown altındaysa atla
                try:
                    if _is_symbol_in_cooldown(sym):
                        log.info(f"{sym}: performans cooldown aktif, yeni giriş atlandı.")
                        continue
                except Exception:
                    pass

                # Korelasyon filtresi ve açık pozisyon kontrolü
                if sym in planned_open_trades:
                    log.info(f"{sym}: zaten açık/planned pozisyon var, yeni giriş atlandı.")
                    continue

                # Önce symbol kilitlenmiş mi kontrol et
                lock = _lock_for(sym)
                if lock.locked():
                    log.info(f"{sym}: symbol-locked, atlanıyor.")
                    continue

                # Portföy limit kontrolü
                if ENFORCE_MAX_OPEN_POSITIONS and MAX_OPEN_POSITIONS > 0 and len(planned_open_trades) >= MAX_OPEN_POSITIONS:
                    log.info(f"{sym}: maks. açık pozisyon sayısı ({MAX_OPEN_POSITIONS}) aşıldı, yeni giriş atlandı.")
                    continue

                # Yeni: Sektör bazlı portföy dengeleme
                # Belirli bir sektörde çok sayıda açık pozisyonu önlemek için,
                # sector_map.json ile eşleştirilmiş kategoriye göre sayım yapılır.
                if CATEGORY_LIMITS and SECTOR_MAP:
                    try:
                        # Kategori haritasından sembolün kategorisini al
                        cat = _get_symbol_category(sym)
                        if cat:
                            # Şu an planlanan açık pozisyonlardaki kategori sayısını hesapla
                            cat_count = 0
                            for open_sym in planned_open_trades.keys():
                                open_cat = _get_symbol_category(open_sym)
                                if open_cat == cat:
                                    cat_count += 1
                            # Bu kategori için tanımlı bir limit varsa kontrol et
                            limit = CATEGORY_LIMITS.get(cat)
                            if limit is not None and cat_count >= limit:
                                log.info(
                                    f"{sym}: kategori '{cat}' için açık pozisyon sınırı ({limit}) dolu, yeni giriş atlandı."
                                )
                                continue
                    except Exception:
                        pass

                # Cooldown kontrolü: Aynı sembolde belirli süre içinde yeni trade açma
                if _should_skip_due_to_cooldown(sym):
                    log.info(f"{sym}: trade cooldown ({TRADE_COOLDOWN_MIN} dk) devam ediyor, yeni giriş atlandı.")
                    continue

                skip_trade = False
                try:
                    if corr_df is not None:
                        for open_sym, open_side in planned_open_trades.items():
                            if sym == open_sym:
                                continue

                            corr_value = None
                            try:
                                corr_value = corr_df.loc[sym, open_sym]
                            except Exception:
                                pass
                            if corr_value is None:
                                continue

                            if corr_value > 0.8 and open_side == dec.get("base_decision"):
                                log.info(
                                    f"{sym}: {open_sym} ile korelasyon {corr_value:.2f} ve aynı yönde. Pozisyon açılmıyor."
                                )
                                skip_trade = True
                                break
                except Exception as e:
                    log.warning(f"Korelasyon kontrol hatası: {e}")
                if skip_trade:
                    continue

                planned_open_trades[sym] = dec.get("base_decision")

                # Emir kuyruğuna at
                await order_queue.put((sym, dec))
                entries_sent += 1

        # Basit console dashboard satırı
        try:
            open_trades_count = len(real_open_trades)
        except Exception:
            open_trades_count = 0

        try:
            daily_pnl = get_daily_realized_pnl()
        except Exception:
            daily_pnl = 0.0

        skips = max(0, enter_signals - entries_sent)
        kill_status = "ON" if is_killed() else "OFF"

        log.info(
            f"[DASH] Open trades: {open_trades_count} | Daily PnL: {daily_pnl:.2f} | "
            f"KillSwitch: {kill_status} | Last loop: {entries_sent} entries / {skips} skips"
        )

        # Persist runtime status for external dashboards.  Use a
        # try/except to ensure metrics writing does not impact
        # trading logic.  The update records the current open trades,
        # PnL, kill switch state and the counts of entries and skips.
        try:
            _update_runtime_status(
                open_trades=open_trades_count,
                daily_pnl=float(daily_pnl) if daily_pnl is not None else 0.0,
                kill_switch=kill_status,
                last_entries=entries_sent,
                last_skips=skips,
            )
        except Exception:
            pass

        log.info("Döngü tamamlandı. 900 sn bekleme...")
        await asyncio.sleep(900)


# ──────────────────────────────────────────────────────────────────────────────
# Zamanlayıcı Entegrasyonu: Veri Çekimi → Dataset → RL → BiLSTM (03:00 & 15:00)
# ──────────────────────────────────────────────────────────────────────────────
def run_generate_and_train():
    """
    Sıralı tetik zinciri:
      1) generate_ohlc_bulk.py    → metrics/ohlc_history.json  (kök klasör)
      2) ml.build_dataset         → data/training/merged_training_data.csv
      3) ml.rl_train              → RL modeli eğitimi / güncelleme
      4) ml.bilstm_train          → BiLSTM modeli eğitimi / güncelleme

    Ardından risk modülleri (kök klasör) + controller_async reload
    """
    try:
        # HANGİ PYTHON? → Dışarıda py -3.12 -m main_bot_async ile açtığın interpreter
        # Import sys here to ensure it exists when run inside the scheduler thread.
        import sys  # noqa: F401
        py = sys.executable

        # 1) generate_ohlc_bulk.py (kök klasör)
        log.info("⏱ 1/4 generate_ohlc_bulk.py çalışıyor...")
        subprocess.run([py, "generate_ohlc_bulk.py"], check=True)

        # 2) ml.build_dataset (ml klasörü içindeki build_dataset.py)
        log.info("⏱ 2/4 ml.build_dataset çalışıyor...")
        subprocess.run([py, "-m", "ml.build_dataset"], check=True)

        # 3) ml.rl_train (ml klasörü içindeki rl_train.py)
        log.info("⏱ 3/4 ml.rl_train çalışıyor...")
        subprocess.run([py, "-m", "ml.rl_train"], check=True)

        # 4) ml.bilstm_train (ml klasörü içindeki bilstm_train.py)
        log.info("⏱ 4/5 ml.bilstm_train çalışıyor...")
        subprocess.run([py, "-m", "ml.bilstm_train"], check=True)

        # 5) risk calibration (calibrate_confidence.py)
        try:
            log.info("⏱ 5/5 calibrate_confidence.py çalışıyor...")
            subprocess.run([py, "calibrate_confidence.py"], check=True)
        except Exception as e:
            log.warning(f"risk calibration hatası: {e}")

        # 6) optional hyperparameter search (hyperparameter_search.py)
        try:
            if str(os.getenv("ENABLE_HYPERPARAM_SEARCH", "0")).lower() not in ("0", "false", "no", ""):  # truthy
                log.info("⏱ hyperparameter_search.py çalışıyor...")
                subprocess.run([py, "hyperparameter_search.py"], check=True)
        except Exception as e:
            log.warning(f"hyperparameter_search hatası: {e}")

        log.info("✅ Zamanlanmış veri→eğitim zinciri tamamlandı.")

        # --- Ek risk modülleri (kök klasör scriptleri) ---
        try:
            log.info("⏱ risk_dataset_builder.py çalışıyor...")
            subprocess.run([py, "risk_dataset_builder.py"], check=True)
        except subprocess.CalledProcessError as e:
            log.error(f"risk_dataset_builder hatası: {e}")

        try:
            log.info("⏱ risk_calibrator.py çalışıyor...")
            subprocess.run([py, "risk_calibrator.py"], check=True)
        except subprocess.CalledProcessError as e:
            log.error(f"risk_calibrator hatası: {e}")

        try:
            log.info("⏱ risk_schedule_builder.py çalışıyor...")
            subprocess.run([py, "risk_schedule_builder.py"], check=True)
        except subprocess.CalledProcessError as e:
            log.error(f"risk_schedule_builder hatası: {e}")

        # --- Yeni risk parametreleri için controller_async reload ---
        try:
            import controller_async  # tür: ignore
            importlib.reload(controller_async)
            log.info("🔄 controller_async yeniden yüklendi; yeni risk parametreleri aktif.")
        except Exception as e:
            log.warning(f"controller_async reload başarısız: {e}")

    except subprocess.CalledProcessError as e:
        log.error(f"❌ Zamanlanmış görev hatası: {e}")


def _schedule_thread():
    # Türkiye (Europe/Istanbul) yerel saate göre 03:00 ve 15:00
    schedule.every().day.at("03:00").do(run_generate_and_train)
    schedule.every().day.at("15:00").do(run_generate_and_train)
    log.info("⏲ Zamanlayıcı başlatıldı (03:00 & 15:00).")
    while True:
        try:
            schedule.run_pending()
            time_module.sleep(30)
        except Exception as e:
            log.error(f"Scheduler döngü hatası: {e}")
            time_module.sleep(10)


# Arka planda zamanlayıcıyı başlat (daemon thread)
threading.Thread(target=_schedule_thread, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# Giriş Noktası
# ──────────────────────────────────────────────────────────────────────────────
async def main():
    # Başlangıçta borsa bağlantısını kur
    exchange = initialize_exchange()
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Microservice katmanı
    #
    # Varsayılan olarak microservice modunu **açık** bırakıyoruz.  Böylece
    # bot çalışırken DataCollector, SignalGenerator ve OrderExecutor
    # sınıfları başlatılır ve dahili kuyruklar aracılığıyla veri/sinyal akışı
    # gerçekleştirilir.  Microservice runner, `microservices.py` içindeki
    # iskelet sınıfları veya fallback implementasyonlarını otomatik olarak
    # adapte edebildiği için hata üretmez.  Microservice katmanını devre
    # dışı bırakmak isterseniz ``USE_MICROSERVICES=0`` veya ``false`` olarak
    # tanımlayabilirsiniz.  Aksi halde (varsayılan) microservice katmanı
    # etkin olacaktır.
    micro_mgr = None
    try:
        # Ortam değişkenini oku ve microservice'in etkin olup olmadığını belirle
        use_ms_env = os.environ.get("USE_MICROSERVICES", "1").strip().lower()
        if use_ms_env not in ("0", "false", "no"):
            from microservice_runner import run_microservices  # type: ignore
            micro_mgr = run_microservices()
            try:
                log.info("Microservice runner started (enabled)")
            except Exception:
                pass
        else:
            micro_mgr = None
            try:
                log.info("Microservice runner disabled (USE_MICROSERVICES set to false)")
            except Exception:
                pass
    except Exception:
        micro_mgr = None
    # WebSocket akışını başlat.  Akış mevcut değilse sessizce yut.
    async def _websocket_stream() -> None:
        """Gerçek zamanlı fiyat akışını dinler ve fiyat geçmişini günceller.

        Bu görev websockets kütüphanesi mevcutsa Binance ticker akışını
        kullanır; aksi durumda hiçbir şey yapmadan döner.  Her bir güncellemede
        ``LAST_PRICES`` ve ``PRICE_HISTORY`` güncellenir.  Hata durumunda
        kısa bir bekleme sonrasında yeniden bağlanmayı dener.
        """
        try:
            from websocket_data_provider import stream_prices  # type: ignore
        except Exception:
            return
        symbols = [s.replace("/", "").lower() for s in ACTIVE_SYMBOLS]
        while True:
            try:
                async for sym, price in stream_prices(symbols):
                    try:
                        # WebSocket sembolü küçük harflerle gelir, ACTIVE_SYMBOLS büyük harfli olabilir
                        # Örneğin btcusdt → BTC/USDT formatına çevir
                        if not sym:
                            continue
                        # split by nothing; we have symbol like btcusdt -> uppercase with slash
                        sym_norm = sym.upper()
                        # Enstrümanları slash'lı formata dönüştür: btcusdt → BTC/USDT
                        if len(sym_norm) > 4:
                            base = sym_norm[:-4]
                            quote = sym_norm[-4:]
                            sym_norm = f"{base}/{quote}"
                        # Güncel fiyat ve tarihçe güncelle
                        LAST_PRICES[sym_norm] = float(price)
                        _update_price_history(sym_norm, float(price))
                    except Exception:
                        continue
            except Exception:
                # Yeniden bağlanmadan önce bekle
                await asyncio.sleep(5)
                continue
    # WebSocket akışını background task olarak başlat
    try:
        # Launch via create_bg_task so it is tracked and properly cancelled
        create_bg_task(_websocket_stream())
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # Opsiyon ve likidasyon WebSocket görevleri ile balina izleme
    #
    # Deribit ve likidasyon WebSocket'leri ile kendi balina takipçisini
    # arka planda çalıştırmak için aşağıdaki görevler tanımlanır.  Hangi
    # görevlerin etkin olacağını ortam değişkenleri kontrol eder.

    async def _options_monitor() -> None:
        """Fetch options metrics via REST periodically and update metrics.

        This monitor runs when WebSocket streaming is disabled (i.e.,
        ``ENABLE_DERIBIT_WS`` is false or not set).  It queries the
        Deribit REST API via ``fetch_options_metrics`` for each base
        symbol and updates the global ``OPTIONS_LIVE_METRICS`` dict.
        An anomaly is recorded if the implied volatility exceeds a
        configurable threshold.  The polling interval and symbol list
        can be configured via environment variables.
        """
        # If WebSocket streaming is enabled, skip this REST monitor
        ws_flag = os.environ.get("ENABLE_DERIBIT_WS", "0").strip().lower()
        if ws_flag not in ("0", "false", "no", ""):
            return
        try:
            from ops_data_provider import fetch_options_metrics  # type: ignore
        except Exception:
            return
        # Determine which base symbols to query.  Use DERIBIT_REST_SYMBOLS
        # environment variable if provided, else derive from ACTIVE_SYMBOLS.
        bases: list[str] = []
        env_syms = os.environ.get("DERIBIT_REST_SYMBOLS")
        if env_syms:
            bases = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
        if not bases:
            # derive from active symbols (unique bases)
            for s in ACTIVE_SYMBOLS:
                try:
                    base = s.split("/")[0].upper()
                    bases.append(base)
                except Exception:
                    continue
            bases = list(set(bases))
        # Only keep supported bases to avoid fetching metrics for illiquid markets
        try:
            bases = [b for b in bases if str(b).upper() in OPTIONS_SUPPORTED_BASES]
        except Exception:
            # In case of any error, fallback to BTC/ETH
            bases = [b for b in ["BTC", "ETH"] if b in bases]
        if not bases:
            # Nothing to monitor if all bases are unsupported
            return
        # Poll interval and volatility threshold
        try:
            poll_sec = int(float(os.environ.get("OPTIONS_POLL_INTERVAL", "600")))
        except Exception:
            poll_sec = 600
        try:
            vol_thresh = float(os.environ.get("OPTIONS_IV_THRESHOLD", "1.0"))
        except Exception:
            vol_thresh = 1.0
        while True:
            for base in bases:
                try:
                    metrics = fetch_options_metrics(base)
                except Exception:
                    continue
                if isinstance(metrics, dict):
                    OPTIONS_LIVE_METRICS[base] = metrics
                    # Check implied volatility threshold
                    iv = metrics.get("implied_volatility")
                    try:
                        iv_f = float(iv) if iv is not None else None
                        if iv_f is not None and iv_f >= vol_thresh:
                            try:
                                record_anomaly()
                            except Exception:
                                pass
                    except Exception:
                        pass
            await asyncio.sleep(poll_sec)

    async def _liquidation_monitor() -> None:
        """Monitor liquidation data via WebSocket or REST polling.

        If ``ENABLE_LIQUIDATION_WS`` is false, this function falls back to
        calling ``fetch_liquidation_heatmap`` periodically to build a
        cumulative liquidation intensity for each base symbol.  If the
        WebSocket flag is true, it subscribes to a public liquidation
        stream (Binance or Bybit) via ``start_liquidation_ws``.  An
        anomaly is recorded when notional values exceed the configured
        threshold.
        """
        enable_ws = os.environ.get("ENABLE_LIQUIDATION_WS", "0").strip().lower()
        if enable_ws in ("0", "false", "no", ""):
            # REST polling
            try:
                from liquidation_data_provider import fetch_liquidation_heatmap  # type: ignore
            except Exception:
                return
            # Determine base symbols to poll
            bases: list[str] = []
            env_syms = os.environ.get("LIQUIDATION_REST_SYMBOLS")
            if env_syms:
                bases = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
            if not bases:
                for sym in ACTIVE_SYMBOLS:
                    try:
                        bases.append(sym.split("/")[0].upper())
                    except Exception:
                        continue
                bases = list(set(bases))
            try:
                poll_sec = int(float(os.environ.get("LIQUIDATION_POLL_INTERVAL", "600")))
            except Exception:
                poll_sec = 600
            try:
                threshold = float(os.environ.get("LIQUIDATION_ANOMALY_THRESHOLD", "10000000"))
            except Exception:
                threshold = 10_000_000.0
            while True:
                for base in bases:
                    try:
                        heatmap = fetch_liquidation_heatmap(base)
                    except Exception:
                        continue
                    total = 0.0
                    if isinstance(heatmap, dict):
                        for v in heatmap.values():
                            try:
                                total += float(v or 0.0)
                            except Exception:
                                continue
                    LIQUIDATION_INTENSITY[base] = total
                    if total >= threshold:
                        try:
                            record_anomaly()
                        except Exception:
                            pass
                await asyncio.sleep(poll_sec)

    async def _options_ws_monitor() -> None:
        """Subscribe to Deribit WebSocket option metrics and update state.

        When ``ENABLE_DERIBIT_WS`` is enabled, this monitor connects to
        Deribit via ``ops_data_provider.start_deribit_ws`` and listens for
        option book updates.  On each notification it fetches the latest
        option metrics via ``fetch_options_metrics`` for each base asset
        and updates ``OPTIONS_LIVE_METRICS``.  If the implied volatility
        exceeds a configured threshold, an anomaly is recorded.  The
        WebSocket connection will automatically reconnect after errors or
        timeouts.  If WebSockets are disabled, this coroutine returns
        immediately.
        """
        ws_flag = os.environ.get("ENABLE_DERIBIT_WS", "0").strip().lower()
        if ws_flag in ("0", "false", "no", ""):
            return
        try:
            from ops_data_provider import start_deribit_ws, fetch_options_metrics  # type: ignore
        except Exception:
            return
        # Derive the set of base assets from DERIBIT_WS_SYMBOLS or ACTIVE_SYMBOLS
        bases: list[str] = []
        env_syms = os.environ.get("DERIBIT_WS_SYMBOLS")
        if env_syms:
            bases = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
        if not bases:
            for s in ACTIVE_SYMBOLS:
                try:
                    base = s.split("/")[0].upper()
                    bases.append(base)
                except Exception:
                    continue
            bases = list(set(bases))
        # Filter to supported bases only
        try:
            bases = [b for b in bases if str(b).upper() in OPTIONS_SUPPORTED_BASES]
        except Exception:
            bases = [b for b in ["BTC", "ETH"] if b in bases]
        if not bases:
            return
        # Read volatility threshold
        try:
            vol_thresh = float(os.environ.get("OPTIONS_IV_THRESHOLD", "1.0"))
        except Exception:
            vol_thresh = 1.0
        # Delay between reconnection attempts
        try:
            reconnect_delay = int(float(os.environ.get("DERIBIT_WS_RECONNECT", "5")))
        except Exception:
            reconnect_delay = 5
        # Callback invoked on each WebSocket event.  It ignores the message
        # contents and instead polls the REST API for each base to get
        # consolidated metrics.  This ensures consistent data structure and
        # allows anomaly detection using the same logic as the REST monitor.
        async def _on_ws_event(event: Any) -> None:
            for base in bases:
                try:
                    metrics = fetch_options_metrics(base)  # type: ignore
                except Exception:
                    continue
                if isinstance(metrics, dict):
                    OPTIONS_LIVE_METRICS[base] = metrics
                    iv = metrics.get("implied_volatility")
                    try:
                        iv_f = float(iv) if iv is not None else None
                        if iv_f is not None and iv_f >= vol_thresh:
                            try:
                                record_anomaly()
                            except Exception:
                                pass
                    except Exception:
                        pass
        # Persistent loop to maintain the WebSocket connection
        while True:
            try:
                await start_deribit_ws(bases, _on_ws_event)  # type: ignore[arg-type]
            except Exception:
                # Sleep before attempting to reconnect
                await asyncio.sleep(reconnect_delay)
        else:
            # WebSocket mode
            try:
                from liquidation_data_provider import start_liquidation_ws  # type: ignore
            except Exception:
                return
            exch = os.environ.get("LIQUIDATION_WS_EXCHANGE", "binance").strip().lower()
            syms_env = os.environ.get("LIQUIDATION_WS_SYMBOLS")
            symbols = None
            if syms_env:
                symbols = [s.strip().upper() for s in syms_env.split(",") if s.strip()]
            async def _cb(event: dict) -> None:
                try:
                    sym = event.get("symbol")
                    notional = float(event.get("notional") or 0.0)
                    if sym:
                        LIQUIDATION_INTENSITY[sym] = LIQUIDATION_INTENSITY.get(sym, 0.0) + notional
                        thr = float(os.environ.get("LIQUIDATION_ANOMALY_THRESHOLD", "10000000"))
                        if notional >= thr:
                            try:
                                record_anomaly()
                            except Exception:
                                pass
                except Exception:
                    return
            try:
                await start_liquidation_ws(exchange=exch, symbols=symbols, update_callback=_cb)
            except Exception:
                return

    async def _whale_monitor() -> None:
        """Periodically poll large transfers via Etherscan and update alerts."""
        enable = os.environ.get("ENABLE_WHALE_MONITOR", "0").strip().lower()
        if enable in ("0", "false", "no", ""):
            return
        try:
            from whale_alert_provider import fetch_whale_alerts  # type: ignore
        except Exception:
            return
        # Poll interval and minimum USD threshold
        try:
            poll_sec = int(float(os.environ.get("WHALE_POLL_INTERVAL", "300")))
        except Exception:
            poll_sec = 300
        try:
            min_usd = float(os.environ.get("WHALE_MIN_USD", "1000000"))
        except Exception:
            min_usd = 1_000_000.0
        try:
            lookback_hours = int(float(os.environ.get("WHALE_LOOKBACK_HOURS", "1")))
        except Exception:
            lookback_hours = 1
        while True:
            try:
                # fetch_whale_alerts aggregates events across chains using
                # the watchlist and configured minimum USD threshold.
                events = fetch_whale_alerts(min_value_usd=min_usd, lookback_hours=lookback_hours)  # type: ignore
                if events:
                    # Store under 'all' key; use lowercase to avoid collisions
                    WHALE_ALERTS["all"] = events
                    try:
                        record_anomaly()
                    except Exception:
                        pass
            except Exception:
                pass
            await asyncio.sleep(poll_sec)

    # Start optional background tasks
    try:
        # If WebSocket streaming for options is enabled, start the WS monitor.
        try:
            if os.environ.get("ENABLE_DERIBIT_WS", "").strip().lower() not in ("0", "false", "no", ""):
                create_bg_task(_options_ws_monitor())
        except Exception:
            pass
        # Always start the REST options monitor as a fallback.  It will exit
        # immediately when WebSocket streaming is enabled.
        try:
            create_bg_task(_options_monitor())
        except Exception:
            pass
    except Exception:
        pass
    try:
        if os.environ.get("ENABLE_LIQUIDATION_WS", "").strip().lower() not in ("0", "false", "no", ""):
            create_bg_task(_liquidation_monitor())
    except Exception:
        pass
    try:
        if os.environ.get("ENABLE_WHALE_MONITOR", "").strip().lower() not in ("0", "false", "no", ""):
            create_bg_task(_whale_monitor())
    except Exception:
        pass

    # Eğer microservice yöneticisi etkinse, köprü görevini başlat
    try:
        if micro_mgr:
            create_bg_task(_microservice_bridge())
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # Microservice köprüsü
    #
    # Eğer USE_MICROSERVICES ortam değişkeni doğrultusunda microservice
    # yöneticisi etkinleştirilmişse (micro_mgr != None), aşağıdaki köprü
    # fonksiyonu DataCollector ve SignalGenerator tarafından üretilen
    # sinyalleri ana botun sipariş kuyruğuna aktarır.  Bu görev, mevcut
    # fiyatları microservice veri toplayıcıya sağlamak ve üretilen işlemleri
    # botun kullandığı karar formatına dönüştürmekten sorumludur.  Böylece
    # microservice tabanlı sinyal üretimi tam otomatik olarak sisteme
    # entegre olur.  Eğer micro_mgr kullanılmıyorsa fonksiyon hemen döner.
    async def _microservice_bridge() -> None:
        # micro_mgr None ise köprüyü çalıştırma
        if not micro_mgr:
            return
        import asyncio as _asyncio  # gölge import: cancel hataları için
        from risk_manager import calculate_tiered_leverage_and_allocation  # type: ignore
        while True:
            try:
                # Microservice DataCollector'a güncel sembol istekleri gönder
                try:
                    for s in ACTIVE_SYMBOLS:
                        # microservice sembol formatı slash içermeyen büyük harf (ör. BTCUSDT)
                        sym_req = s.replace("/", "").upper()
                        try:
                            micro_mgr.add_price_request(sym_req)
                        except Exception:
                            pass
                except Exception:
                    pass
                # micro_mgr.q_orders listesindeki sinyalleri boşalt
                orders_to_process = []
                try:
                    q = getattr(micro_mgr, "q_orders", None)
                    while q:
                        try:
                            order = q.pop(0)
                        except Exception:
                            break
                        if order:
                            orders_to_process.append(order)
                        else:
                            break
                except Exception:
                    pass
                # Her sinyali bot karar formatına dönüştür
                for ms_order in orders_to_process:
                    try:
                        sym_raw = ms_order.get("symbol")
                        action = ms_order.get("action")
                        confidence = ms_order.get("confidence")
                        if not sym_raw or not action:
                            continue
                        sym_up = str(sym_raw).upper()
                        if "/" in sym_up:
                            sym_norm = sym_up
                        else:
                            if len(sym_up) > 4:
                                base = sym_up[:-4]
                                quote = sym_up[-4:]
                                sym_norm = f"{base}/{quote}"
                            elif len(sym_up) > 3:
                                base = sym_up[:-3]
                                quote = sym_up[-3:]
                                sym_norm = f"{base}/{quote}"
                            else:
                                sym_norm = sym_up
                        price_val = LAST_PRICES.get(sym_norm)
                        if price_val is None:
                            continue
                        act_lower = str(action).lower()
                        if act_lower == "buy":
                            base_dec = "long"
                        elif act_lower == "sell":
                            base_dec = "short"
                        else:
                            continue
                        try:
                            conf_val = float(confidence)
                        except Exception:
                            conf_val = 0.5
                        atr_val = float(price_val) * 0.008
                        try:
                            tier = calculate_tiered_leverage_and_allocation(conf_val)
                            lev_val = tier.get("leverage") or 1
                        except Exception:
                            lev_val = 1
                        decision = {
                            "price": float(price_val),
                            "master_confidence": conf_val,
                            "base_decision": base_dec,
                            "lev": int(lev_val),
                            "atr": float(atr_val),
                            "meta_strategy": "microservice"
                        }
                        try:
                            await order_queue.put((sym_norm, decision))
                        except Exception:
                            pass
                    except Exception:
                        continue
            except _asyncio.CancelledError:
                return
            except Exception:
                pass
            await _asyncio.sleep(1.0)


    # ---------------------------------------------------------------------
    # Pair trading sinyallerini izle ve bildirim gönder
    #
    # Bu görev, fiyat tarihçesindeki oran sapmalarını periyodik olarak
    # kontrol eder.  Eğer pair_trading modülü bir spread fırsatı tespit
    # ederse, kullanıcıya bir bildirim gönderilir.  Bu, daha karmaşık bir
    # entegrasyon için temel oluşturur; sinyallerin otomatik emir
    # oluşturması karar katmanına bağlanabilir.
    async def _pair_trade_monitor() -> None:
        try:
            from pair_trading import find_spread_opportunities  # type: ignore
        except Exception:
            return
        # Dinamik olarak notification göndermek için gecikmeli yükleme
        try:
            from notification import send_notification  # type: ignore
        except Exception:
            def send_notification(msg: str, subject: str | None = None) -> bool:  # type: ignore
                return False
        while True:
            try:
                # Yeterli fiyat verisi biriktiğinde spread fırsatlarını ara
                signals = find_spread_opportunities(PRICE_HISTORY, threshold=2.0, lookback=50)
                if signals:
                    for sig in signals:
                        try:
                            pair = sig.get("pair")
                            long_sym = sig.get("long")
                            short_sym = sig.get("short")
                            zscore = sig.get("zscore")
                            ts = sig.get("timestamp")
                            # Before acting on this signal, enforce a cooldown per pair. If the same
                            # pair was signalled recently, skip to avoid rapid open/close cycles.
                            try:
                                if pair and isinstance(pair, (list, tuple)) and len(pair) == 2:
                                    # Normalise to a sorted tuple for keying; ensure strings for consistency
                                    p0, p1 = str(pair[0]), str(pair[1])
                                    key = tuple(sorted([p0, p1]))
                                    now_ts = time_module.time()
                                    last_ts = _PAIR_TRADE_LAST_TIME.get(key)
                                    # If within cooldown window, skip this signal
                                    if last_ts is not None and now_ts - last_ts < PAIR_TRADE_COOLDOWN_SECONDS:
                                        log.info(f"[PAIR_COOLDOWN] Skipping signal for {pair} due to cooldown.")
                                        continue
                                    # Record the timestamp of this signal
                                    _PAIR_TRADE_LAST_TIME[key] = now_ts
                            except Exception:
                                pass
                            # Send a notification describing the opportunity
                            try:
                                msg = (
                                    f"📊 Pair trading fırsatı: {pair} z-score={zscore:.2f}. "
                                    f"Long {long_sym}, Short {short_sym} at {ts}."
                                )
                                send_notification(msg)
                                log.info(msg)
                            except Exception:
                                # Notification errors should not block trading logic
                                pass
                            # -----------------------------------------------------------------
                            # Automatically convert pair trading signal into trade decisions
                            # -----------------------------------------------------------------
                            def _create_pair_decision(sym: str, side: str) -> dict:
                                """Construct a minimal decision object for pair trading.

                                Args:
                                    sym: symbol to trade (e.g. "BTC/USDT").
                                    side: direction, either "long" or "short".
                                Returns:
                                    A dict with required fields for orders_worker or empty if price missing.
                                """
                                price_seq = PRICE_HISTORY.get(sym)
                                price = None
                                if price_seq:
                                    try:
                                        price = float(price_seq[-1])
                                    except Exception:
                                        price = None
                                if price is None:
                                    return {}
                                ta_pack = {"base_decision": side}
                                senti = _sentiment_snapshot_for_decision()
                                # Include z-score and pair information so the orders_worker can make
                                # informed risk allocations (e.g. scale position size by z-score).
                                return {
                                    "symbol": sym,
                                    "tf": "pair",
                                    "price": price,
                                    "ta_pack": ta_pack,
                                    "senti": senti,
                                    "anomaly": False,
                                    "atr": None,
                                    "meta_strategy": "pair_trading",
                                    "base_decision": side,
                                    # Assign a moderate confidence; risk management will adjust leverage.
                                    "master_confidence": 0.6,
                                    "lev": 1,
                                    "zscore": zscore,
                                    "pair": pair,
                                }

                            # Build and enqueue decisions for both legs of the pair trade
                            for sym, side in ((long_sym, "long"), (short_sym, "short")):
                                decision = _create_pair_decision(sym, side)
                                if decision:
                                    try:
                                        await order_queue.put((sym, decision))
                                        log.info(
                                            f"Pair trade enqueued: {sym} {side} @ {decision['price']:.6f} (zscore={zscore:.2f})"
                                        )
                                    except Exception as ex:
                                        log.warning(f"Pair trade enqueue error for {sym}: {ex}")
                        except Exception:
                            continue
                # 60 saniye bekle
                await asyncio.sleep(60)
            except Exception:
                await asyncio.sleep(60)
                continue
    # Pair trading monitor'ü background task olarak başlat
    try:
        create_bg_task(_pair_trade_monitor())
    except Exception:
        pass
    # Ana ticaret döngüsünü çalıştır
    await trading_loop_async(exchange)
    # Döngü sonlandığında tüm arka plan görevlerini iptal et ve borsayı kapat
    try:
        await _cancel_bg_tasks()
    except Exception:
        pass
    try:
        if exchange is not None:
            await exchange.close()
    except Exception:
        pass


if __name__ == "__main__":
    # Fail fast on missing critical credentials instead of crashing later.
    # You can bypass this for offline development by setting STRICT_ENV=0.
    try:
        from settings import require_env, env_bool
        if env_bool("STRICT_ENV", True):
            require_env(["OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE"], context="OKX")
    except Exception as e:
        # Re-raise to make the problem visible
        raise

    # When running this script directly, launch background services for
    # sentiment scheduling and stop order monitoring.  These services
    # run in separate daemon threads so they do not block the main event loop.
    try:
        import sentiment_scheduler  # provides run_scheduler()
        import stop_order_watchdog  # provides run_watchdog()
        # Start scheduler and stop order watchdog threads
        threading.Thread(target=sentiment_scheduler.run_scheduler, daemon=True).start()
        threading.Thread(target=stop_order_watchdog.run_watchdog, daemon=True).start()
        log = get_logger("main_bot_async")
        log.info("🔁 Background services started: sentiment scheduler & stop order watchdog")
    except Exception as _e:
        # Background services may fail silently if dependencies missing
        try:
            print(f"[WARN] Background services not started: {_e}")
        except Exception:
            pass
    # Kullanıcı kontrolü (Telegram) ve watchdog'u başlat
    try:
        from user_control import TelegramController  # type: ignore
        ctrl = TelegramController(_handle_user_command)
        ctrl.start()
        log.info("Telegram user control started")
    except Exception as _e:
        try:
            log.info(f"[INFO] Telegram user control not started: {_e}")
        except Exception:
            pass
    try:
        from watchdog import Watchdog  # type: ignore
        # İzlenecek kritik environment anahtarları
        env_keys = [
            "OKX_API_KEY",
            "OKX_API_SECRET",
            "OKX_API_PASSPHRASE",
            "OPENAI_API_KEY",
        ]
        # Adjust watchdog thresholds: use a slightly higher CPU threshold to reduce false positives.
        wd = Watchdog(memory_threshold=0.9, cpu_threshold=0.98, interval=60.0, callback=_watchdog_callback, env_keys=env_keys, ping_url="https://www.google.com")
        wd.start()
        log.info("Watchdog started")
    except Exception as _e:
        try:
            log.info(f"[INFO] Watchdog not started: {_e}")
        except Exception:
            pass
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        try:
            log.warning("KeyboardInterrupt: güvenli çıkış.")
        except Exception:
            pass