# -*- coding: utf-8 -*-
"""
regime_detector.py
==================

Piyasa Rejim Tespit Modulu (BULL / BEAR / SIDEWAYS)

Bu modul, 4 saatlik grafik verisinden piyasa rejimini tespit eder:
- EMA200 ve MACD histogram ortalamasini kullanir
- Sonucu data/market_regime.json dosyasina yazar
- start_regime_scheduler() ile haftalik (Pazar 03:00 UTC) otomatik calisir

Rejim Tanimlari:
- BULL: Fiyat EMA200 uzerinde ve MACD pozitif
- BEAR: Fiyat EMA200 altinda ve MACD negatif
- SIDEWAYS: Karisik sinyaller

Kullanim:
    from regime_detector import detect_market_regime, start_regime_scheduler
    
    # Tek seferlik tespit
    result = detect_market_regime("BTC/USDT")
    
    # Zamanlanmis tespit
    start_regime_scheduler()

CHANGELOG:
- v1.0: Initial version
- v1.1: Fixed encoding issues (Turkish characters)
- v1.2: Added error handling and fallback
- v1.3: Added async version support
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Exchange imports
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None  # type: ignore

# Data processing imports
try:
    import pandas as pd
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    pd = None  # type: ignore
    ta = None  # type: ignore

# Scheduler import
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    BackgroundScheduler = None  # type: ignore

# Logger import
try:
    from logger import get_logger
    log = get_logger("regime_detector")
except ImportError:
    import logging
    log = logging.getLogger("regime_detector")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s"))
        log.addHandler(handler)

# Settings import
try:
    from settings import (
        OKX_API_KEY,
        OKX_API_SECRET,
        OKX_API_PASSPHRASE,
        OKX_USE_TESTNET,
    )
except ImportError:
    OKX_API_KEY = os.getenv("OKX_API_KEY", "")
    OKX_API_SECRET = os.getenv("OKX_API_SECRET", "")
    OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "")
    OKX_USE_TESTNET = os.getenv("OKX_USE_TESTNET", "true").lower() in ("1", "true", "yes")


# Path configuration
REGIME_PATH = os.path.join("data", "market_regime.json")


def initialize_exchange() -> Optional[Any]:
    """
    OKX baglantisini kurar.
    
    Returns:
        ccxt.okx instance or None if unavailable
    """
    if not CCXT_AVAILABLE:
        log.error("ccxt kutuphanesi kurulu degil")
        return None
    
    try:
        exchange = ccxt.okx({
            "apiKey": OKX_API_KEY,
            "secret": OKX_API_SECRET,
            "password": OKX_API_PASSPHRASE,
            "enableRateLimit": True,
        })
        exchange.set_sandbox_mode(bool(OKX_USE_TESTNET))
        exchange.load_markets()
        return exchange
    except Exception as e:
        log.error(f"Exchange baglanti hatasi: {e}")
        return None


def calculate_regime_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate regime indicators from OHLCV dataframe.
    
    Args:
        df: DataFrame with 'close' column
        
    Returns:
        Dict with ema_diff_mean and macd_hist_mean
    """
    if not TALIB_AVAILABLE:
        # Simple fallback without talib
        closes = df["close"].values
        ema200 = df["close"].ewm(span=200).mean()
        
        # Simple MACD calculation
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_hist = macd - signal
        
        tail = min(100, len(df))
        ema_diff_mean = (df["close"].iloc[-tail:] - ema200.iloc[-tail:]).mean()
        macd_mean = macd_hist.iloc[-tail:].mean()
        
        return {
            "ema_diff_mean": float(ema_diff_mean),
            "macd_hist_mean": float(macd_mean)
        }
    
    # Use talib for more accurate calculations
    df["EMA200"] = ta.EMA(df["close"], timeperiod=200)
    macd, macd_signal, macd_hist = ta.MACD(df["close"], 12, 26, 9)
    df["MACD_Hist"] = macd_hist
    
    # Calculate means over last 100 bars
    tail = min(100, len(df))
    if tail < 50:
        raise ValueError("Yeterli 4h veri yok (>=50 bar gerekli)")
    
    ema_diff_mean = (df["close"].iloc[-tail:] - df["EMA200"].iloc[-tail:]).mean()
    macd_mean = df["MACD_Hist"].iloc[-tail:].mean()
    
    return {
        "ema_diff_mean": float(ema_diff_mean),
        "macd_hist_mean": float(macd_mean)
    }


def classify_regime(ema_diff_mean: float, macd_hist_mean: float) -> str:
    """
    Classify market regime based on indicators.
    
    Args:
        ema_diff_mean: Average difference from EMA200
        macd_hist_mean: Average MACD histogram value
        
    Returns:
        "BULL", "BEAR", or "SIDEWAYS"
    """
    if ema_diff_mean > 0 and macd_hist_mean > 0:
        return "BULL"
    elif ema_diff_mean < 0 and macd_hist_mean < 0:
        return "BEAR"
    else:
        return "SIDEWAYS"


def detect_market_regime(symbol: str = "BTC/USDT", limit: int = 1500) -> Dict[str, Any]:
    """
    4H grafikten EMA200 + MACD histogram analizine gore rejim tespiti.
    
    Args:
        symbol: Trading pair symbol
        limit: Number of 4h candles to fetch
        
    Returns:
        Dict with regime info including:
        - REGIME: "BULL", "BEAR", or "SIDEWAYS"
        - ema_diff_mean: Average price difference from EMA200
        - macd_hist_mean: Average MACD histogram
        - symbol: The analyzed symbol
        - timeframe: "4h"
        - sample: Number of bars used for final calculation
        - last_update: Timestamp
        - env: "Demo" or "Live"
    """
    result: Dict[str, Any] = {
        "REGIME": "UNKNOWN",
        "symbol": symbol,
        "timeframe": "4h",
        "last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "env": "Demo" if OKX_USE_TESTNET else "Live"
    }
    
    try:
        # Initialize exchange
        ex = initialize_exchange()
        if ex is None:
            result["error"] = "Exchange baglantisi kurulamadi"
            _save_result(result)
            return result
        
        # Fetch OHLCV data
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="4h", limit=limit)
        if not ohlcv:
            raise RuntimeError("OKX OHLCV bos dondu")
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["close"] = df["close"].astype(float)
        
        # Calculate indicators
        indicators = calculate_regime_indicators(df)
        
        # Classify regime
        regime = classify_regime(
            indicators["ema_diff_mean"],
            indicators["macd_hist_mean"]
        )
        
        # Build result
        result.update({
            "REGIME": regime,
            "ema_diff_mean": round(indicators["ema_diff_mean"], 6),
            "macd_hist_mean": round(indicators["macd_hist_mean"], 6),
            "sample": min(100, len(df)),
            "total_bars": len(df)
        })
        
        log.info(
            f"Rejim tespit edildi: {regime} | "
            f"ema_diff={result['ema_diff_mean']:.6f} | "
            f"macd_hist={result['macd_hist_mean']:.6f} | "
            f"sample={result['sample']}"
        )
        
    except Exception as e:
        log.error(f"Rejim tespiti hatasi: {e}")
        result["error"] = str(e)
    
    # Save result
    _save_result(result)
    
    return result


def _save_result(result: Dict[str, Any]) -> None:
    """Save regime result to file."""
    try:
        os.makedirs(os.path.dirname(REGIME_PATH), exist_ok=True)
        with open(REGIME_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    except Exception as e:
        log.error(f"Rejim kaydetme hatasi: {e}")


def load_current_regime() -> Dict[str, Any]:
    """
    Load the current regime from file.
    
    Returns:
        Regime dict or default UNKNOWN regime
    """
    default = {
        "REGIME": "UNKNOWN",
        "last_update": None
    }
    
    try:
        if os.path.exists(REGIME_PATH):
            with open(REGIME_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    
    return default


def get_regime_multiplier(regime: Optional[str] = None) -> float:
    """
    Get risk multiplier based on current regime.
    
    Args:
        regime: Regime string, or None to load from file
        
    Returns:
        Risk multiplier (0.5 to 1.2)
    """
    if regime is None:
        data = load_current_regime()
        regime = data.get("REGIME", "UNKNOWN")
    
    multipliers = {
        "BULL": 1.1,      # Slightly increase risk in bull market
        "BEAR": 0.7,      # Reduce risk in bear market
        "SIDEWAYS": 0.9,  # Slightly reduce risk in ranging market
        "UNKNOWN": 0.8    # Conservative when unknown
    }
    
    return multipliers.get(regime, 0.8)


def start_regime_scheduler() -> Optional[Any]:
    """
    Haftalik (Pazar 03:00 UTC) rejim tespiti zamanlayicisi.
    
    Returns:
        Scheduler instance or None if unavailable
    """
    if not SCHEDULER_AVAILABLE:
        log.error("APScheduler kurulu degil. Zamanlanmis rejim tespiti kullanilamiyor.")
        return None
    
    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            detect_market_regime,
            "cron",
            day_of_week="sun",
            hour=3,
            minute=0
        )
        scheduler.start()
        log.info("Haftalik Rejim Tespiti zamanlayicisi baslatildi (Pazar 03:00 UTC)")
        return scheduler
    except Exception as e:
        log.error(f"Scheduler baslatilamadi: {e}")
        return None


if __name__ == "__main__":
    print("Rejim tespiti baslatiliyor...")
    result = detect_market_regime()
    print(f"Sonuc: {json.dumps(result, indent=2, ensure_ascii=False)}")
