# -*- coding: utf-8 -*-
"""
enhancements.py
===============
AutoTraderBot için kapsamlı iyileştirmeler ve ek özellikler.

İçerik:
1. AI Agreement Bonus Calculator
2. Multi-Timeframe Confirmation System
3. Trailing Stop Manager
4. Smart TP Levels
5. Correlation Checker
6. Performance Tracker
7. Volume Spike Detector
8. Order Book Imbalance Analyzer
9. Funding Rate Strategy
10. Time-Based Exit Rules
11. Daily Summary Generator
12. Dynamic Threshold Calculator
13. Entry Optimizer
14. Auto-Parameter Tuning (YENİ)
15. Position Hedging Strategy (YENİ)
16. Drawdown Recovery Mode (YENİ)
17. Liquidity Analyzer (YENİ)
18. Slippage Tracker (YENİ)

Bu modül mevcut sisteme entegre edilmek üzere tasarlanmıştır.
Versiyon: 3.0
Tarih: 2026-01-01
"""

import time
import json
import asyncio
import math
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import random
import copy


# ==============================================================================
# 1. AI AGREEMENT BONUS CALCULATOR
# ==============================================================================

def calculate_ai_agreement_bonus(
    chatgpt_result: Dict[str, Any],
    deepseek_result: Dict[str, Any],
    bilstm_prob: float
) -> Dict[str, Any]:
    """
    Tüm AI modeller aynı yönü gösterdiğinde bonus ver.
    
    Returns:
        {"bonus": float, "reasons": List[str], "agreement_level": str}
    """
    # Direction extraction
    chatgpt_dir = str(chatgpt_result.get("direction", "")).lower()
    deepseek_dir = str(deepseek_result.get("direction", "")).lower()
    
    # BiLSTM direction (threshold bazlı)
    if bilstm_prob > 0.58:
        bilstm_dir = "long"
    elif bilstm_prob < 0.42:
        bilstm_dir = "short"
    else:
        bilstm_dir = "neutral"
    
    # Action extraction
    chatgpt_act = str(chatgpt_result.get("action", "")).lower()
    deepseek_act = str(deepseek_result.get("action", "")).lower()
    
    bonus = 0.0
    reasons = []
    agreement_level = "none"
    
    # Direction agreement checks
    directions = [chatgpt_dir, deepseek_dir, bilstm_dir]
    valid_directions = [d for d in directions if d in ("long", "short")]
    
    if len(valid_directions) >= 2:
        # Check if all valid directions agree
        if len(set(valid_directions)) == 1:
            if len(valid_directions) == 3:
                # Full agreement (3/3)
                bonus += 0.08
                reasons.append("full_direction_agreement_3_3")
                agreement_level = "full"
            else:
                # Partial agreement (2/3)
                bonus += 0.05
                reasons.append("partial_direction_agreement_2_3")
                agreement_level = "partial"
    
    # Action agreement (both LLMs say "enter")
    if chatgpt_act == "enter" and deepseek_act == "enter":
        bonus += 0.05
        reasons.append("both_llm_enter")
        if agreement_level == "full":
            agreement_level = "strong"
    elif chatgpt_act == "enter" or deepseek_act == "enter":
        bonus += 0.02
        reasons.append("one_llm_enter")
    
    # Confidence agreement (if both high confidence)
    chatgpt_conf = float(chatgpt_result.get("confidence", 0.5))
    deepseek_conf = float(deepseek_result.get("confidence", 0.5))
    
    if chatgpt_conf >= 0.65 and deepseek_conf >= 0.65:
        bonus += 0.03
        reasons.append("high_confidence_agreement")
    
    return {
        "bonus": round(min(0.15, bonus), 3),  # Max %15 bonus
        "reasons": reasons,
        "agreement_level": agreement_level,
        "details": {
            "chatgpt_dir": chatgpt_dir,
            "deepseek_dir": deepseek_dir,
            "bilstm_dir": bilstm_dir,
            "chatgpt_act": chatgpt_act,
            "deepseek_act": deepseek_act
        }
    }


# ==============================================================================
# 2. MULTI-TIMEFRAME CONFIRMATION SYSTEM
# ==============================================================================

@dataclass
class MTFSignal:
    """Multi-timeframe sinyal sonucu."""
    timeframe: str
    aligned: bool
    strength: float  # 0-1
    reason: str


def check_mtf_alignment(
    ta_pack: Dict[str, Any],
    direction: str,
    include_timeframes: List[str] = None
) -> Dict[str, Any]:
    """
    Multi-timeframe alignment kontrolü.
    
    Args:
        ta_pack: Teknik analiz paketi (her timeframe için ayrı veriler içermeli)
        direction: "long" veya "short"
        include_timeframes: Kontrol edilecek timeframe'ler
    
    Returns:
        {
            "score": float,
            "confirmed_count": int,
            "total_count": int,
            "approved": bool,
            "signals": Dict[str, MTFSignal]
        }
    """
    if include_timeframes is None:
        include_timeframes = ["5m", "15m", "1h", "4h"]
    
    # Timeframe ağırlıkları
    weights = {
        "5m": 0.10,
        "15m": 0.15,
        "1h": 0.25,
        "4h": 0.30,
        "1d": 0.20
    }
    
    signals = {}
    total_weight = 0
    weighted_score = 0
    confirmed_count = 0
    
    for tf in include_timeframes:
        tf_data = ta_pack.get(tf, {})
        if not tf_data:
            tf_data = ta_pack.get(f"tf_{tf}", {})
        
        if not tf_data:
            continue
        
        # Her timeframe için alignment kontrolü
        signal = _check_single_tf_alignment(tf_data, direction, tf)
        signals[tf] = signal
        
        weight = weights.get(tf, 0.15)
        total_weight += weight
        weighted_score += signal.strength * weight
        
        if signal.aligned:
            confirmed_count += 1
    
    # Normalize score
    if total_weight > 0:
        final_score = weighted_score / total_weight
    else:
        final_score = 0.5
    
    # En az 3 timeframe'den 2'si onaylamalı
    min_confirmations = max(2, len(include_timeframes) // 2)
    approved = confirmed_count >= min_confirmations and final_score >= 0.5
    
    return {
        "score": round(final_score, 3),
        "confirmed_count": confirmed_count,
        "total_count": len(signals),
        "approved": approved,
        "min_required": min_confirmations,
        "signals": {tf: vars(sig) for tf, sig in signals.items()}
    }


def _check_single_tf_alignment(
    tf_data: Dict[str, Any],
    direction: str,
    timeframe: str
) -> MTFSignal:
    """Tek bir timeframe için alignment kontrolü."""
    
    aligned = False
    strength = 0.5
    reasons = []
    
    # MACD histogram
    macd_hist = tf_data.get("macd_hist")
    if macd_hist is not None:
        if direction == "long" and macd_hist > 0:
            strength += 0.15
            reasons.append("macd_positive")
            aligned = True
        elif direction == "short" and macd_hist < 0:
            strength += 0.15
            reasons.append("macd_negative")
            aligned = True
        else:
            strength -= 0.10
    
    # EMA alignment
    ema_fast = tf_data.get("ema_fast") or tf_data.get("ema", {}).get("fast")
    ema_slow = tf_data.get("ema_slow") or tf_data.get("ema", {}).get("slow")
    
    if ema_fast is not None and ema_slow is not None:
        if direction == "long" and ema_fast > ema_slow:
            strength += 0.15
            reasons.append("ema_bullish")
            aligned = True
        elif direction == "short" and ema_fast < ema_slow:
            strength += 0.15
            reasons.append("ema_bearish")
            aligned = True
        else:
            strength -= 0.10
    
    # RSI
    rsi = tf_data.get("rsi")
    if rsi is not None:
        if direction == "long":
            if 30 < rsi < 70:
                strength += 0.10
                reasons.append("rsi_neutral_ok")
            elif rsi < 30:
                strength += 0.15
                reasons.append("rsi_oversold")
            else:
                strength -= 0.10
                reasons.append("rsi_overbought_caution")
        else:  # short
            if 30 < rsi < 70:
                strength += 0.10
                reasons.append("rsi_neutral_ok")
            elif rsi > 70:
                strength += 0.15
                reasons.append("rsi_overbought")
            else:
                strength -= 0.10
                reasons.append("rsi_oversold_caution")
    
    # Clamp strength
    strength = max(0.0, min(1.0, strength))
    
    return MTFSignal(
        timeframe=timeframe,
        aligned=aligned and strength >= 0.5,
        strength=strength,
        reason="; ".join(reasons) if reasons else "no_data"
    )


# ==============================================================================
# 3. TRAILING STOP MANAGER
# ==============================================================================

@dataclass
class TrailingState:
    """Trailing stop durumu."""
    symbol: str
    side: str
    entry_price: float
    activated: bool = False
    highest_profit: float = 0.0
    trailing_stop: Optional[float] = None
    breakeven_set: bool = False
    last_update: float = field(default_factory=time.time)


class TrailingStopManager:
    """
    Aktif trailing stop yöneticisi.
    
    Özellikler:
    - %1.5 profit'te aktivasyon
    - Breakeven koruma
    - ATR bazlı trail distance
    - Profit locking
    """
    
    ACTIVATION_PROFIT_PCT = 0.015  # %1.5
    BREAKEVEN_PROFIT_PCT = 0.01   # %1.0
    MIN_TRAIL_DISTANCE = 0.008    # %0.8
    
    def __init__(self):
        self.positions: Dict[str, TrailingState] = {}
    
    def register_position(
        self,
        symbol: str,
        side: str,
        entry_price: float
    ) -> None:
        """Yeni pozisyon kaydet."""
        self.positions[symbol] = TrailingState(
            symbol=symbol,
            side=side,
            entry_price=entry_price
        )
    
    def update(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Trailing stop'u güncelle.
        
        Returns:
            {"action": "hold"|"close", "trailing_stop": float, "reason": str}
        """
        state = self.positions.get(symbol)
        if not state:
            return {"action": "hold", "reason": "no_position_registered"}
        
        entry = state.entry_price
        side = state.side
        
        # Profit hesapla
        if side == "long":
            profit_pct = (current_price - entry) / entry
        else:
            profit_pct = (entry - current_price) / entry
        
        # Breakeven aktivasyonu
        if profit_pct >= self.BREAKEVEN_PROFIT_PCT and not state.breakeven_set:
            state.breakeven_set = True
            state.trailing_stop = entry
        
        # Trailing aktivasyonu
        if profit_pct >= self.ACTIVATION_PROFIT_PCT and not state.activated:
            state.activated = True
            state.highest_profit = profit_pct
            # Initial trailing stop at entry (breakeven)
            if state.trailing_stop is None:
                state.trailing_stop = entry
        
        # Trailing stop güncelleme
        if state.activated:
            if profit_pct > state.highest_profit:
                state.highest_profit = profit_pct
                
                # Trail distance hesapla
                if atr and entry > 0:
                    atr_based = 0.5 * atr / entry
                    trail_dist = max(self.MIN_TRAIL_DISTANCE, atr_based)
                else:
                    trail_dist = self.MIN_TRAIL_DISTANCE
                
                # Yeni stop hesapla
                if side == "long":
                    new_stop = current_price * (1 - trail_dist)
                    if state.trailing_stop is None or new_stop > state.trailing_stop:
                        state.trailing_stop = new_stop
                else:
                    new_stop = current_price * (1 + trail_dist)
                    if state.trailing_stop is None or new_stop < state.trailing_stop:
                        state.trailing_stop = new_stop
        
        state.last_update = time.time()
        
        # Stop kontrolü
        if state.trailing_stop is not None:
            if side == "long" and current_price <= state.trailing_stop:
                return {
                    "action": "close",
                    "trailing_stop": state.trailing_stop,
                    "reason": "trailing_stop_hit",
                    "profit_pct": profit_pct
                }
            elif side == "short" and current_price >= state.trailing_stop:
                return {
                    "action": "close",
                    "trailing_stop": state.trailing_stop,
                    "reason": "trailing_stop_hit",
                    "profit_pct": profit_pct
                }
        
        return {
            "action": "hold",
            "trailing_stop": state.trailing_stop,
            "activated": state.activated,
            "breakeven_set": state.breakeven_set,
            "profit_pct": profit_pct,
            "highest_profit": state.highest_profit
        }
    
    def remove_position(self, symbol: str) -> None:
        """Pozisyonu kaldır."""
        self.positions.pop(symbol, None)
    
    def get_all_states(self) -> Dict[str, Dict]:
        """Tüm trailing state'leri döndür."""
        return {sym: vars(state) for sym, state in self.positions.items()}


# ==============================================================================
# 4. SMART TP LEVELS
# ==============================================================================

def get_smart_tp_levels(
    entry_price: float,
    atr: float,
    side: str,
    confidence: float
) -> List[Dict[str, Any]]:
    """
    Confidence'a göre dinamik TP seviyeleri.
    
    Args:
        entry_price: Giriş fiyatı
        atr: Average True Range
        side: "long" veya "short"
        confidence: Master confidence (0-1)
    
    Returns:
        List of {"price": float, "size_pct": float, "level": int}
    """
    # Confidence'a göre profil seç
    if confidence >= 0.80:
        # Yüksek güven: 3 kademe, uzak hedefler
        profile = {
            "multipliers": [1.5, 3.0, 5.0],
            "sizes": [0.30, 0.40, 0.30],
            "name": "aggressive"
        }
    elif confidence >= 0.70:
        # Orta-yüksek güven: 3 kademe
        profile = {
            "multipliers": [1.2, 2.5, 4.0],
            "sizes": [0.35, 0.40, 0.25],
            "name": "balanced"
        }
    elif confidence >= 0.60:
        # Orta güven: 2 kademe
        profile = {
            "multipliers": [1.0, 2.0],
            "sizes": [0.50, 0.50],
            "name": "moderate"
        }
    else:
        # Düşük güven: Tek kademe, yakın hedef
        profile = {
            "multipliers": [1.0],
            "sizes": [1.0],
            "name": "conservative"
        }
    
    levels = []
    for i, (mult, size_pct) in enumerate(zip(profile["multipliers"], profile["sizes"])):
        if side == "long":
            tp_price = entry_price + (atr * mult)
        else:
            tp_price = entry_price - (atr * mult)
        
        levels.append({
            "price": round(tp_price, 8),
            "size_pct": size_pct,
            "level": i + 1,
            "multiplier": mult,
            "distance_pct": round(abs(tp_price - entry_price) / entry_price * 100, 2)
        })
    
    return levels


# ==============================================================================
# 5. CORRELATION CHECKER
# ==============================================================================

CORRELATED_GROUPS = {
    "BTC_ECOSYSTEM": ["BTC/USDT", "WBTC/USDT"],
    "ETH_ECOSYSTEM": ["ETH/USDT", "STETH/USDT", "WETH/USDT", "CBETH/USDT"],
    "LAYER1_ALT": ["SOL/USDT", "AVAX/USDT", "DOT/USDT", "ATOM/USDT", "NEAR/USDT"],
    "LAYER2": ["ARB/USDT", "OP/USDT", "MATIC/USDT", "IMX/USDT", "STRK/USDT"],
    "MEME": ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "FLOKI/USDT", "BONK/USDT", "WIF/USDT"],
    "AI_TOKENS": ["FET/USDT", "AGIX/USDT", "OCEAN/USDT", "RNDR/USDT", "TAO/USDT"],
    "DEFI_BLUE": ["UNI/USDT", "AAVE/USDT", "MKR/USDT", "SNX/USDT", "CRV/USDT", "LDO/USDT"],
    "EXCHANGE": ["BNB/USDT", "OKB/USDT", "CRO/USDT", "FTT/USDT"],
    "GAMING": ["AXS/USDT", "SAND/USDT", "MANA/USDT", "GALA/USDT", "IMX/USDT"],
}


def check_correlation_limit(
    symbol: str,
    open_positions: List[str],
    max_per_group: int = 2
) -> Dict[str, Any]:
    """
    Aynı gruptan maximum pozisyon kontrolü.
    
    Returns:
        {"allowed": bool, "group": str|None, "current_count": int, "reason": str}
    """
    # Find symbol's group
    symbol_upper = symbol.upper()
    symbol_group = None
    
    for group_name, symbols in CORRELATED_GROUPS.items():
        if symbol_upper in [s.upper() for s in symbols]:
            symbol_group = group_name
            break
    
    if not symbol_group:
        return {
            "allowed": True,
            "group": None,
            "current_count": 0,
            "reason": "no_correlation_group"
        }
    
    # Count open positions in same group
    group_symbols = [s.upper() for s in CORRELATED_GROUPS[symbol_group]]
    current_count = sum(
        1 for pos in open_positions
        if pos.upper() in group_symbols
    )
    
    allowed = current_count < max_per_group
    
    return {
        "allowed": allowed,
        "group": symbol_group,
        "current_count": current_count,
        "max_allowed": max_per_group,
        "reason": "limit_reached" if not allowed else "ok"
    }


# ==============================================================================
# 6. PERFORMANCE TRACKER
# ==============================================================================

@dataclass
class TradeRecord:
    """Tek bir trade kaydı."""
    symbol: str
    side: str
    pnl_pct: float
    entry_time: float
    exit_time: float
    confidence: float = 0.0
    exit_reason: str = "unknown"


class PerformanceTracker:
    """
    Trade performansını takip et ve adaptif ayarlamalar öner.
    """
    
    def __init__(self, window: int = 50, save_path: str = "metrics/performance_history.json"):
        self.window = window
        self.save_path = Path(save_path)
        self.trades: deque = deque(maxlen=window)
        self._load()
    
    def _load(self) -> None:
        """Geçmiş kayıtları yükle."""
        try:
            if self.save_path.exists():
                data = json.loads(self.save_path.read_text())
                for t in data.get("trades", [])[-self.window:]:
                    self.trades.append(TradeRecord(**t))
        except Exception:
            pass
    
    def _save(self) -> None:
        """Kayıtları dosyaya yaz."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"trades": [vars(t) for t in self.trades]}
            self.save_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
    
    def add_trade(
        self,
        symbol: str,
        side: str,
        pnl_pct: float,
        entry_time: float,
        exit_time: float,
        confidence: float = 0.0,
        exit_reason: str = "unknown"
    ) -> None:
        """Trade kaydı ekle."""
        self.trades.append(TradeRecord(
            symbol=symbol,
            side=side,
            pnl_pct=pnl_pct,
            entry_time=entry_time,
            exit_time=exit_time,
            confidence=confidence,
            exit_reason=exit_reason
        ))
        self._save()
    
    def get_stats(self) -> Dict[str, Any]:
        """Performans istatistikleri."""
        if len(self.trades) < 5:
            return {"sufficient_data": False, "trade_count": len(self.trades)}
        
        trades = list(self.trades)
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        
        win_rate = len(wins) / len(trades)
        avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
        
        # Long vs Short
        longs = [t for t in trades if t.side == "long"]
        shorts = [t for t in trades if t.side == "short"]
        
        long_wr = len([t for t in longs if t.pnl_pct > 0]) / len(longs) if longs else 0.5
        short_wr = len([t for t in shorts if t.pnl_pct > 0]) / len(shorts) if shorts else 0.5
        
        # Profit factor
        gross_profit = sum(t.pnl_pct for t in wins)
        gross_loss = abs(sum(t.pnl_pct for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            "sufficient_data": True,
            "trade_count": len(trades),
            "win_rate": round(win_rate, 3),
            "avg_win_pct": round(avg_win * 100, 2),
            "avg_loss_pct": round(avg_loss * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(expectancy * 100, 3),
            "long_win_rate": round(long_wr, 3),
            "short_win_rate": round(short_wr, 3),
            "long_count": len(longs),
            "short_count": len(shorts)
        }
    
    def get_adjustments(self) -> Dict[str, Any]:
        """Performansa göre ayarlama önerileri."""
        stats = self.get_stats()
        
        if not stats.get("sufficient_data"):
            return {
                "threshold_adj": 0,
                "leverage_mult": 1.0,
                "long_bias": 0,
                "short_bias": 0,
                "reason": "insufficient_data"
            }
        
        adjustments = {
            "threshold_adj": 0.0,
            "leverage_mult": 1.0,
            "long_bias": 0.0,
            "short_bias": 0.0,
            "reasons": []
        }
        
        win_rate = stats["win_rate"]
        profit_factor = stats["profit_factor"]
        
        # Win rate bazlı threshold ayarı
        if win_rate < 0.45:
            adjustments["threshold_adj"] = 0.05
            adjustments["leverage_mult"] = 0.8
            adjustments["reasons"].append("low_win_rate_conservative")
        elif win_rate > 0.60 and profit_factor > 1.5:
            adjustments["threshold_adj"] = -0.03
            adjustments["leverage_mult"] = 1.1
            adjustments["reasons"].append("high_win_rate_aggressive")
        
        # Long/Short bias
        long_wr = stats["long_win_rate"]
        short_wr = stats["short_win_rate"]
        
        if long_wr > short_wr + 0.15 and stats["long_count"] >= 10:
            adjustments["long_bias"] = 0.05
            adjustments["short_bias"] = -0.05
            adjustments["reasons"].append("long_outperforming")
        elif short_wr > long_wr + 0.15 and stats["short_count"] >= 10:
            adjustments["short_bias"] = 0.05
            adjustments["long_bias"] = -0.05
            adjustments["reasons"].append("short_outperforming")
        
        return adjustments


# ==============================================================================
# 7. VOLUME SPIKE DETECTOR
# ==============================================================================

class VolumeSpikeDetector:
    """
    Anormal hacim artışlarını tespit et.
    Yüksek hacim = güçlü hareket sinyali.
    """
    
    def __init__(self, lookback: int = 20, spike_threshold: float = 2.5):
        self.lookback = lookback
        self.spike_threshold = spike_threshold
        self.volume_history: Dict[str, deque] = {}
    
    def update(self, symbol: str, volume: float) -> Dict[str, Any]:
        """
        Yeni hacim verisi ekle ve spike kontrolü yap.
        """
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=self.lookback)
        
        history = self.volume_history[symbol]
        
        result = {
            "is_spike": False,
            "volume": volume,
            "avg_volume": 0,
            "spike_ratio": 0,
            "signal": "normal"
        }
        
        if len(history) >= 5:  # Minimum 5 veri noktası
            avg_vol = sum(history) / len(history)
            result["avg_volume"] = avg_vol
            
            if avg_vol > 0:
                ratio = volume / avg_vol
                result["spike_ratio"] = round(ratio, 2)
                
                if ratio >= self.spike_threshold:
                    result["is_spike"] = True
                    result["signal"] = "high_volume_spike"
                elif ratio >= 1.5:
                    result["signal"] = "elevated_volume"
                elif ratio <= 0.5:
                    result["signal"] = "low_volume"
        
        history.append(volume)
        return result
    
    def get_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """Hacim profili istatistikleri."""
        history = self.volume_history.get(symbol, deque())
        
        if len(history) < 3:
            return {"sufficient_data": False}
        
        volumes = list(history)
        return {
            "sufficient_data": True,
            "avg": sum(volumes) / len(volumes),
            "max": max(volumes),
            "min": min(volumes),
            "last": volumes[-1] if volumes else 0,
            "trend": "increasing" if len(volumes) >= 3 and volumes[-1] > volumes[-3] else "decreasing"
        }


# ==============================================================================
# 8. ORDER BOOK IMBALANCE ANALYZER
# ==============================================================================

def analyze_orderbook_imbalance(
    bids: List[List[float]],
    asks: List[List[float]],
    depth_levels: int = 10
) -> Dict[str, Any]:
    """
    Orderbook dengesizliğini analiz et.
    
    Args:
        bids: [[price, size], ...] - Alış emirleri
        asks: [[price, size], ...] - Satış emirleri
        depth_levels: Kaç seviye analiz edilecek
    
    Returns:
        {
            "imbalance": float (-1 to +1),
            "bid_volume": float,
            "ask_volume": float,
            "signal": str,
            "strength": float
        }
    """
    if not bids or not asks:
        return {"imbalance": 0, "signal": "no_data", "strength": 0}
    
    # İlk N seviyedeki toplam hacim
    bid_volume = sum(b[1] for b in bids[:depth_levels])
    ask_volume = sum(a[1] for a in asks[:depth_levels])
    
    total = bid_volume + ask_volume
    if total == 0:
        return {"imbalance": 0, "signal": "no_volume", "strength": 0}
    
    # Imbalance: +1 = tam alış baskısı, -1 = tam satış baskısı
    imbalance = (bid_volume - ask_volume) / total
    
    # Signal belirleme
    if imbalance > 0.3:
        signal = "strong_buy_pressure"
        strength = min(1.0, imbalance / 0.5)
    elif imbalance > 0.1:
        signal = "moderate_buy_pressure"
        strength = imbalance / 0.3
    elif imbalance < -0.3:
        signal = "strong_sell_pressure"
        strength = min(1.0, abs(imbalance) / 0.5)
    elif imbalance < -0.1:
        signal = "moderate_sell_pressure"
        strength = abs(imbalance) / 0.3
    else:
        signal = "balanced"
        strength = 0
    
    return {
        "imbalance": round(imbalance, 4),
        "bid_volume": round(bid_volume, 2),
        "ask_volume": round(ask_volume, 2),
        "signal": signal,
        "strength": round(strength, 3),
        "recommendation": "long" if imbalance > 0.2 else "short" if imbalance < -0.2 else "neutral"
    }


# ==============================================================================
# 9. FUNDING RATE STRATEGY
# ==============================================================================

class FundingRateAnalyzer:
    """
    Funding rate bazlı contrarian strateji.
    Extreme funding = crowded trade = ters yön fırsatı.
    """
    
    EXTREME_POSITIVE = 0.0005   # %0.05 per 8h
    EXTREME_NEGATIVE = -0.0005
    VERY_EXTREME = 0.001       # %0.1 per 8h
    
    def __init__(self):
        self.funding_history: Dict[str, deque] = {}
    
    def analyze(self, symbol: str, funding_rate: float) -> Dict[str, Any]:
        """
        Funding rate analizi.
        
        Returns:
            {
                "rate": float,
                "rate_8h_pct": float,
                "rate_daily_pct": float,
                "signal": str,
                "confidence_adj": float,
                "contrarian_direction": str
            }
        """
        # History tracking
        if symbol not in self.funding_history:
            self.funding_history[symbol] = deque(maxlen=24)  # 8 gün (3 funding/gün)
        self.funding_history[symbol].append(funding_rate)
        
        rate_8h_pct = funding_rate * 100
        rate_daily_pct = rate_8h_pct * 3
        
        result = {
            "rate": funding_rate,
            "rate_8h_pct": round(rate_8h_pct, 4),
            "rate_daily_pct": round(rate_daily_pct, 4),
            "signal": "neutral",
            "confidence_adj": 0,
            "contrarian_direction": "none"
        }
        
        # Extreme positive: Long crowded → Short bias
        if funding_rate >= self.VERY_EXTREME:
            result["signal"] = "extreme_long_crowded"
            result["confidence_adj"] = 0.05
            result["contrarian_direction"] = "short"
        elif funding_rate >= self.EXTREME_POSITIVE:
            result["signal"] = "long_crowded"
            result["confidence_adj"] = 0.03
            result["contrarian_direction"] = "short"
        
        # Extreme negative: Short crowded → Long bias
        elif funding_rate <= -self.VERY_EXTREME:
            result["signal"] = "extreme_short_crowded"
            result["confidence_adj"] = 0.05
            result["contrarian_direction"] = "long"
        elif funding_rate <= self.EXTREME_NEGATIVE:
            result["signal"] = "short_crowded"
            result["confidence_adj"] = 0.03
            result["contrarian_direction"] = "long"
        
        # Avg funding (trend)
        history = list(self.funding_history[symbol])
        if len(history) >= 3:
            avg_funding = sum(history) / len(history)
            result["avg_funding"] = round(avg_funding, 6)
            result["funding_trend"] = "increasing" if funding_rate > avg_funding else "decreasing"
        
        return result


# ==============================================================================
# 10. TIME-BASED EXIT RULES
# ==============================================================================

def check_time_based_exit(
    entry_timestamp: float,
    current_pnl_pct: float,
    side: str,
    regime: str = "SIDEWAYS"
) -> Dict[str, Any]:
    """
    Zaman bazlı çıkış kuralları.
    
    Kurallar:
    - 24 saat sonra kar varsa al
    - 48 saat sonra breakeven'daysa çık
    - 72 saat sonra ne olursa olsun çık
    - Sideways rejimde daha erken çık
    """
    hours_open = (time.time() - entry_timestamp) / 3600
    
    # Regime bazlı max hold time
    max_hold_hours = {
        "BULL": 96,
        "BEAR": 72,
        "SIDEWAYS": 48
    }.get(regime.upper(), 72)
    
    rules = [
        # Kısa vadeli kar al
        {"min_hours": 12, "min_pnl": 0.02, "action": "early_profit", "priority": 1},
        
        # Orta vadeli kar al
        {"min_hours": 24, "min_pnl": 0.01, "action": "take_profit", "priority": 2},
        
        # Uzun vadeli breakeven çıkış
        {"min_hours": 48, "min_pnl": -0.005, "action": "breakeven_exit", "priority": 3},
        
        # Maksimum hold süresi
        {"min_hours": max_hold_hours, "min_pnl": -1.0, "action": "max_time_exit", "priority": 4},
        
        # Sideways'de erken çıkış
        {"min_hours": 24, "min_pnl": 0.005, "action": "sideways_exit", "priority": 2, "regime": "SIDEWAYS"},
    ]
    
    result = {
        "should_exit": False,
        "reason": None,
        "hours_open": round(hours_open, 1),
        "max_hold_hours": max_hold_hours,
        "pnl_pct": round(current_pnl_pct * 100, 2)
    }
    
    for rule in rules:
        # Regime kontrolü
        if "regime" in rule and rule["regime"] != regime.upper():
            continue
        
        if hours_open >= rule["min_hours"] and current_pnl_pct >= rule["min_pnl"]:
            result["should_exit"] = True
            result["reason"] = rule["action"]
            result["rule_matched"] = rule
            break
    
    return result


# ==============================================================================
# 11. DAILY SUMMARY GENERATOR
# ==============================================================================

class DailySummaryGenerator:
    """
    Günlük performans özeti oluştur.
    """
    
    def __init__(self, save_path: str = "metrics/daily_summaries"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.today_trades: List[Dict] = []
        self.current_date: str = ""
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Trade kaydı ekle."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Gün değiştiyse sıfırla
        if today != self.current_date:
            self._save_summary()
            self.today_trades = []
            self.current_date = today
        
        self.today_trades.append({
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "pnl_pct": trade.get("pnl_pct", 0),
            "pnl_usd": trade.get("pnl_usd", 0),
            "entry_time": trade.get("entry_time"),
            "exit_time": trade.get("exit_time"),
            "confidence": trade.get("confidence", 0),
            "exit_reason": trade.get("exit_reason", "unknown")
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Günlük özet oluştur."""
        if not self.today_trades:
            return {"trade_count": 0, "date": self.current_date}
        
        trades = self.today_trades
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        
        total_pnl_pct = sum(t["pnl_pct"] for t in trades)
        total_pnl_usd = sum(t["pnl_usd"] for t in trades)
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            reason = t.get("exit_reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Symbol performance
        symbol_perf = {}
        for t in trades:
            sym = t["symbol"]
            if sym not in symbol_perf:
                symbol_perf[sym] = {"count": 0, "pnl": 0}
            symbol_perf[sym]["count"] += 1
            symbol_perf[sym]["pnl"] += t["pnl_pct"]
        
        return {
            "date": self.current_date,
            "trade_count": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "total_pnl_pct": round(total_pnl_pct * 100, 2),
            "total_pnl_usd": round(total_pnl_usd, 2),
            "avg_pnl_pct": round(total_pnl_pct / len(trades) * 100, 2) if trades else 0,
            "best_trade": max(trades, key=lambda x: x["pnl_pct"]) if trades else None,
            "worst_trade": min(trades, key=lambda x: x["pnl_pct"]) if trades else None,
            "exit_reasons": exit_reasons,
            "symbol_performance": symbol_perf,
            "long_count": len([t for t in trades if t["side"] == "long"]),
            "short_count": len([t for t in trades if t["side"] == "short"])
        }
    
    def _save_summary(self) -> None:
        """Özeti dosyaya kaydet."""
        if not self.today_trades or not self.current_date:
            return
        
        try:
            summary = self.get_summary()
            filepath = self.save_path / f"{self.current_date}.json"
            filepath.write_text(json.dumps(summary, indent=2, default=str))
        except Exception:
            pass


# ==============================================================================
# 12. DYNAMIC THRESHOLD CALCULATOR
# ==============================================================================

def calculate_dynamic_threshold(
    base_threshold: float = 0.70,
    regime: str = "SIDEWAYS",
    volatility: str = "MEDIUM",
    recent_win_rate: float = 0.50,
    time_of_day_hour: int = 12
) -> Dict[str, Any]:
    """
    Dinamik threshold hesapla.
    
    Faktörler:
    - Market regime
    - Volatilite seviyesi
    - Son performans
    - Günün saati
    
    Returns:
        {"threshold": float, "adjustments": Dict, "reason": str}
    """
    threshold = base_threshold
    adjustments = {}
    reasons = []
    
    # 1. Regime adjustment
    regime_adj = {
        "BULL": -0.03,    # Boğa'da biraz daha agresif
        "BEAR": +0.03,    # Ayı'da daha temkinli
        "SIDEWAYS": +0.02  # Sideways'de temkinli
    }
    if regime.upper() in regime_adj:
        adj = regime_adj[regime.upper()]
        threshold += adj
        adjustments["regime"] = adj
        reasons.append(f"regime_{regime.lower()}")
    
    # 2. Volatility adjustment
    vol_adj = {
        "LOW": -0.02,     # Düşük vol = daha fazla fırsat
        "MEDIUM": 0,
        "HIGH": +0.05     # Yüksek vol = daha temkinli
    }
    if volatility.upper() in vol_adj:
        adj = vol_adj[volatility.upper()]
        threshold += adj
        adjustments["volatility"] = adj
        reasons.append(f"vol_{volatility.lower()}")
    
    # 3. Performance adjustment
    if recent_win_rate < 0.40:
        adj = +0.05  # Kötü performans = daha seçici
        threshold += adj
        adjustments["performance"] = adj
        reasons.append("poor_performance")
    elif recent_win_rate > 0.60:
        adj = -0.03  # İyi performans = biraz gevşet
        threshold += adj
        adjustments["performance"] = adj
        reasons.append("good_performance")
    
    # 4. Time of day adjustment (UTC)
    if 0 <= time_of_day_hour < 8:
        adj = +0.02  # Asya - daha az likidite
        adjustments["session"] = adj
        reasons.append("asia_session")
    elif 16 <= time_of_day_hour < 24:
        adj = 0  # Amerika - yüksek volatilite ama likidite iyi
        adjustments["session"] = adj
        reasons.append("us_session")
    else:
        adjustments["session"] = 0
        reasons.append("europe_session")
    
    threshold += adjustments.get("session", 0)
    
    # Clamp
    threshold = max(0.55, min(0.80, threshold))
    
    return {
        "threshold": round(threshold, 3),
        "base": base_threshold,
        "adjustments": adjustments,
        "total_adjustment": round(threshold - base_threshold, 3),
        "reasons": reasons
    }


# ==============================================================================
# 13. ENTRY OPTIMIZER (Smart Limit Orders)
# ==============================================================================

def calculate_optimal_entry(
    best_bid: float,
    best_ask: float,
    side: str,
    urgency: str = "normal"
) -> Dict[str, Any]:
    """
    Optimal giriş fiyatı hesapla.
    
    Args:
        best_bid: En iyi alış fiyatı
        best_ask: En iyi satış fiyatı
        side: "long" veya "short"
        urgency: "low", "normal", "high"
    
    Returns:
        {"order_type": str, "price": float, "timeout_sec": int}
    """
    spread = (best_ask - best_bid) / best_bid
    mid_price = (best_bid + best_ask) / 2
    
    result = {
        "spread_pct": round(spread * 100, 4),
        "mid_price": mid_price,
        "best_bid": best_bid,
        "best_ask": best_ask
    }
    
    # Çok dar spread = market order OK
    if spread < 0.0003:  # < %0.03
        result["order_type"] = "market"
        result["price"] = None
        result["timeout_sec"] = 0
        result["reason"] = "tight_spread"
        return result
    
    # Urgency bazlı strateji
    if urgency == "high":
        # Acil: Spread'in içinde agresif limit
        if side == "long":
            price = best_ask * 0.9999  # Ask'ın hemen altı
        else:
            price = best_bid * 1.0001  # Bid'in hemen üstü
        timeout = 10
    elif urgency == "low":
        # Sabırlı: Daha iyi fiyat için bekle
        if side == "long":
            price = best_bid * 1.0002  # Bid'in biraz üstü
        else:
            price = best_ask * 0.9998  # Ask'ın biraz altı
        timeout = 60
    else:
        # Normal: Orta yol
        if side == "long":
            price = mid_price * 0.9999
        else:
            price = mid_price * 1.0001
        timeout = 30
    
    result["order_type"] = "limit"
    result["price"] = round(price, 8)
    result["timeout_sec"] = timeout
    result["reason"] = f"{urgency}_urgency"
    
    return result


# ==============================================================================
# 14. AUTO-PARAMETER TUNING
# ==============================================================================

@dataclass
class ParameterConfig:
    """Optimize edilecek parametre tanımı."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step: float
    category: str  # "threshold", "risk", "technical"


class AutoParameterTuner:
    """
    Otomatik parametre optimizasyonu.
    
    Özellikler:
    - Performans bazlı parametre ayarlama
    - Bayesian-like optimization (simplified)
    - Safe bounds enforcement
    - Gradual adjustment (ani değişiklik yok)
    """
    
    def __init__(self, save_path: str = "config/tuned_parameters.json"):
        self.save_path = Path(save_path)
        self.parameters: Dict[str, ParameterConfig] = {}
        self.performance_history: deque = deque(maxlen=100)
        self.tuning_history: List[Dict] = []
        self._load()
        self._init_default_parameters()
    
    def _init_default_parameters(self) -> None:
        """Varsayılan parametreleri tanımla."""
        defaults = [
            # Threshold parametreleri
            ParameterConfig("min_confidence_threshold", 0.70, 0.55, 0.80, 0.02, "threshold"),
            ParameterConfig("ai_agreement_bonus_max", 0.15, 0.05, 0.20, 0.02, "threshold"),
            
            # Risk parametreleri
            ParameterConfig("max_leverage", 20, 5, 30, 2, "risk"),
            ParameterConfig("max_wallet_pct", 0.12, 0.05, 0.20, 0.02, "risk"),
            ParameterConfig("stop_loss_atr_mult", 2.5, 1.5, 4.0, 0.25, "risk"),
            ParameterConfig("take_profit_atr_mult", 3.0, 2.0, 6.0, 0.5, "risk"),
            
            # Technical parametreleri
            ParameterConfig("adx_threshold_low", 15, 10, 20, 2, "technical"),
            ParameterConfig("adx_threshold_mid", 25, 20, 30, 2, "technical"),
            ParameterConfig("rsi_oversold", 30, 20, 35, 2, "technical"),
            ParameterConfig("rsi_overbought", 70, 65, 80, 2, "technical"),
            
            # Trailing stop parametreleri
            ParameterConfig("trailing_activation_pct", 0.015, 0.01, 0.03, 0.005, "trailing"),
            ParameterConfig("trailing_distance_pct", 0.008, 0.005, 0.015, 0.002, "trailing"),
        ]
        
        for param in defaults:
            if param.name not in self.parameters:
                self.parameters[param.name] = param
    
    def _load(self) -> None:
        """Kaydedilmiş parametreleri yükle."""
        try:
            if self.save_path.exists():
                data = json.loads(self.save_path.read_text())
                for name, values in data.get("parameters", {}).items():
                    self.parameters[name] = ParameterConfig(**values)
                self.tuning_history = data.get("history", [])
        except Exception:
            pass
    
    def _save(self) -> None:
        """Parametreleri kaydet."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "parameters": {name: vars(p) for name, p in self.parameters.items()},
                "history": self.tuning_history[-50:],  # Son 50 kayıt
                "last_updated": datetime.now().isoformat()
            }
            self.save_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
    
    def record_trade_result(self, pnl_pct: float, params_used: Dict[str, float]) -> None:
        """Trade sonucunu kaydet."""
        self.performance_history.append({
            "pnl_pct": pnl_pct,
            "params": params_used,
            "timestamp": time.time()
        })
    
    def get_parameter(self, name: str) -> float:
        """Parametre değerini al."""
        if name in self.parameters:
            return self.parameters[name].current_value
        return None
    
    def set_parameter(self, name: str, value: float) -> bool:
        """Parametre değerini güvenli şekilde ayarla."""
        if name not in self.parameters:
            return False
        
        param = self.parameters[name]
        # Bounds check
        value = max(param.min_value, min(param.max_value, value))
        # Step alignment
        steps = round((value - param.min_value) / param.step)
        value = param.min_value + (steps * param.step)
        
        param.current_value = value
        self._save()
        return True
    
    def optimize(self, min_trades: int = 20) -> Dict[str, Any]:
        """
        Performans verilerine göre parametreleri optimize et.
        
        Args:
            min_trades: Minimum trade sayısı optimizasyon için
        
        Returns:
            {"optimized": bool, "changes": Dict, "reason": str}
        """
        if len(self.performance_history) < min_trades:
            return {
                "optimized": False,
                "reason": f"insufficient_data ({len(self.performance_history)}/{min_trades})"
            }
        
        trades = list(self.performance_history)
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        
        win_rate = len(wins) / len(trades)
        avg_pnl = sum(t["pnl_pct"] for t in trades) / len(trades)
        
        changes = {}
        reasons = []
        
        # Threshold optimization
        if win_rate < 0.45:
            # Win rate düşük - threshold'u artır
            current = self.get_parameter("min_confidence_threshold")
            if current:
                new_val = min(current + 0.02, 0.75)
                if self.set_parameter("min_confidence_threshold", new_val):
                    changes["min_confidence_threshold"] = {"old": current, "new": new_val}
                    reasons.append("low_win_rate_increase_threshold")
        elif win_rate > 0.60 and avg_pnl > 0.005:
            # Win rate yüksek ve karlı - threshold'u düşür
            current = self.get_parameter("min_confidence_threshold")
            if current:
                new_val = max(current - 0.02, 0.60)
                if self.set_parameter("min_confidence_threshold", new_val):
                    changes["min_confidence_threshold"] = {"old": current, "new": new_val}
                    reasons.append("high_win_rate_decrease_threshold")
        
        # Risk optimization
        avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        if avg_loss < -0.03:  # Ortalama kayıp > %3
            # Stop loss çok uzak - ATR multiplier'ı düşür
            current = self.get_parameter("stop_loss_atr_mult")
            if current:
                new_val = max(current - 0.25, 1.5)
                if self.set_parameter("stop_loss_atr_mult", new_val):
                    changes["stop_loss_atr_mult"] = {"old": current, "new": new_val}
                    reasons.append("high_avg_loss_tighten_sl")
        
        # Trailing stop optimization
        # Erken çıkış çok fazla mı?
        early_exits = [t for t in trades if t["pnl_pct"] > 0 and t["pnl_pct"] < 0.01]
        if len(early_exits) > len(wins) * 0.5:
            # Çok erken çıkış - trailing'i gevşet
            current = self.get_parameter("trailing_activation_pct")
            if current:
                new_val = min(current + 0.005, 0.025)
                if self.set_parameter("trailing_activation_pct", new_val):
                    changes["trailing_activation_pct"] = {"old": current, "new": new_val}
                    reasons.append("too_early_exits_relax_trailing")
        
        # Record tuning
        if changes:
            self.tuning_history.append({
                "timestamp": datetime.now().isoformat(),
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "changes": changes,
                "reasons": reasons
            })
            self._save()
        
        return {
            "optimized": bool(changes),
            "changes": changes,
            "reasons": reasons,
            "stats": {
                "trade_count": len(trades),
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl * 100, 2)
            }
        }
    
    def get_all_parameters(self) -> Dict[str, Dict]:
        """Tüm parametreleri döndür."""
        return {
            name: {
                "value": p.current_value,
                "min": p.min_value,
                "max": p.max_value,
                "step": p.step,
                "category": p.category
            }
            for name, p in self.parameters.items()
        }
    
    def reset_to_defaults(self) -> None:
        """Parametreleri varsayılana sıfırla."""
        self.parameters.clear()
        self._init_default_parameters()
        self._save()


# ==============================================================================
# 15. POSITION HEDGING STRATEGY
# ==============================================================================

@dataclass
class HedgePosition:
    """Hedge pozisyonu bilgisi."""
    symbol: str
    main_side: str
    main_size: float
    hedge_symbol: str
    hedge_side: str
    hedge_size: float
    hedge_ratio: float
    created_at: float = field(default_factory=time.time)


class PositionHedger:
    """
    Pozisyon hedging stratejisi.
    
    Stratejiler:
    1. BTC Hedge: Alt coinlerde BTC ile hedge
    2. Pair Hedge: Korele coinlerde ters pozisyon
    3. Delta Neutral: Aynı varlıkta ters pozisyon (düşük kaldıraç)
    """
    
    # Hedge pair tanımları
    HEDGE_PAIRS = {
        # Alt coin -> BTC hedge
        "ETH/USDT": {"hedge_with": "BTC/USDT", "ratio": 0.5, "type": "btc_hedge"},
        "SOL/USDT": {"hedge_with": "BTC/USDT", "ratio": 0.4, "type": "btc_hedge"},
        "AVAX/USDT": {"hedge_with": "BTC/USDT", "ratio": 0.4, "type": "btc_hedge"},
        
        # Korele pair hedge
        "ARB/USDT": {"hedge_with": "OP/USDT", "ratio": 0.6, "type": "pair_hedge"},
        "OP/USDT": {"hedge_with": "ARB/USDT", "ratio": 0.6, "type": "pair_hedge"},
        "DOGE/USDT": {"hedge_with": "SHIB/USDT", "ratio": 0.5, "type": "pair_hedge"},
        "SHIB/USDT": {"hedge_with": "DOGE/USDT", "ratio": 0.5, "type": "pair_hedge"},
    }
    
    def __init__(self, max_hedge_pct: float = 0.5):
        self.max_hedge_pct = max_hedge_pct  # Max pozisyon boyutunun %'si
        self.active_hedges: Dict[str, HedgePosition] = {}
    
    def should_hedge(
        self,
        symbol: str,
        side: str,
        unrealized_pnl_pct: float,
        volatility: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """
        Hedge gerekli mi kontrol et.
        
        Koşullar:
        - Zarar %1'i geçti
        - Volatilite yüksek
        - Hedge pair mevcut
        """
        result = {
            "should_hedge": False,
            "hedge_symbol": None,
            "hedge_side": None,
            "hedge_ratio": 0,
            "reason": None
        }
        
        # Hedge pair kontrolü
        if symbol not in self.HEDGE_PAIRS:
            result["reason"] = "no_hedge_pair_available"
            return result
        
        pair_info = self.HEDGE_PAIRS[symbol]
        
        # Koşul 1: Zarar eşiği
        loss_threshold = -0.01 if volatility == "HIGH" else -0.015
        if unrealized_pnl_pct > loss_threshold:
            result["reason"] = "pnl_above_threshold"
            return result
        
        # Koşul 2: Zaten hedge var mı?
        if symbol in self.active_hedges:
            result["reason"] = "already_hedged"
            return result
        
        # Hedge öner
        result["should_hedge"] = True
        result["hedge_symbol"] = pair_info["hedge_with"]
        result["hedge_side"] = "short" if side == "long" else "long"
        result["hedge_ratio"] = pair_info["ratio"] * self.max_hedge_pct
        result["hedge_type"] = pair_info["type"]
        result["reason"] = "loss_threshold_exceeded"
        
        return result
    
    def create_hedge(
        self,
        main_symbol: str,
        main_side: str,
        main_size: float,
        hedge_symbol: str,
        hedge_side: str,
        hedge_ratio: float
    ) -> HedgePosition:
        """Hedge pozisyonu oluştur."""
        hedge_size = main_size * hedge_ratio
        
        hedge = HedgePosition(
            symbol=main_symbol,
            main_side=main_side,
            main_size=main_size,
            hedge_symbol=hedge_symbol,
            hedge_side=hedge_side,
            hedge_size=hedge_size,
            hedge_ratio=hedge_ratio
        )
        
        self.active_hedges[main_symbol] = hedge
        return hedge
    
    def should_close_hedge(
        self,
        symbol: str,
        main_pnl_pct: float,
        hedge_pnl_pct: float
    ) -> Dict[str, Any]:
        """
        Hedge kapatılmalı mı kontrol et.
        
        Koşullar:
        - Ana pozisyon kara geçti
        - Toplam PnL pozitif
        - Hedge süresi doldu
        """
        if symbol not in self.active_hedges:
            return {"should_close": False, "reason": "no_active_hedge"}
        
        hedge = self.active_hedges[symbol]
        combined_pnl = main_pnl_pct + (hedge_pnl_pct * hedge.hedge_ratio)
        
        result = {
            "should_close": False,
            "combined_pnl": combined_pnl,
            "reason": None
        }
        
        # Ana pozisyon kara geçti
        if main_pnl_pct > 0.005:
            result["should_close"] = True
            result["reason"] = "main_position_profitable"
            return result
        
        # Toplam PnL pozitif
        if combined_pnl > 0.002:
            result["should_close"] = True
            result["reason"] = "combined_pnl_positive"
            return result
        
        # Hedge süresi doldu (24 saat)
        hours_open = (time.time() - hedge.created_at) / 3600
        if hours_open > 24:
            result["should_close"] = True
            result["reason"] = "hedge_expired"
            return result
        
        return result
    
    def close_hedge(self, symbol: str) -> Optional[HedgePosition]:
        """Hedge'i kapat ve döndür."""
        return self.active_hedges.pop(symbol, None)
    
    def get_active_hedges(self) -> Dict[str, Dict]:
        """Aktif hedge'leri döndür."""
        return {sym: vars(h) for sym, h in self.active_hedges.items()}
    
    def calculate_net_exposure(
        self,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Net exposure hesapla.
        
        Returns:
            {"long_exposure": float, "short_exposure": float, "net": float, "hedged_pct": float}
        """
        long_exp = 0.0
        short_exp = 0.0
        
        for pos in positions:
            size_usd = pos.get("size_usd", 0)
            side = pos.get("side", "")
            
            if side == "long":
                long_exp += size_usd
            else:
                short_exp += size_usd
        
        total = long_exp + short_exp
        net = long_exp - short_exp
        hedged_pct = min(long_exp, short_exp) / max(total, 1) * 2
        
        return {
            "long_exposure": round(long_exp, 2),
            "short_exposure": round(short_exp, 2),
            "net_exposure": round(net, 2),
            "total_exposure": round(total, 2),
            "hedged_pct": round(hedged_pct * 100, 1),
            "direction_bias": "long" if net > 0 else "short" if net < 0 else "neutral"
        }


# ==============================================================================
# 16. DRAWDOWN RECOVERY MODE
# ==============================================================================

class DrawdownRecoveryManager:
    """
    Drawdown durumunda koruyucu mod.
    
    Özellikler:
    - Drawdown seviyesine göre kademeli risk azaltma
    - Recovery trade'leri için özel kurallar
    - Otomatik mod geçişleri
    """
    
    # Drawdown seviyeleri ve aksiyonlar
    LEVELS = {
        "NORMAL": {"min_dd": 0, "max_dd": 0.03, "leverage_mult": 1.0, "trade_mult": 1.0},
        "CAUTION": {"min_dd": 0.03, "max_dd": 0.05, "leverage_mult": 0.7, "trade_mult": 0.8},
        "WARNING": {"min_dd": 0.05, "max_dd": 0.08, "leverage_mult": 0.5, "trade_mult": 0.5},
        "CRITICAL": {"min_dd": 0.08, "max_dd": 0.10, "leverage_mult": 0.3, "trade_mult": 0.3},
        "LOCKDOWN": {"min_dd": 0.10, "max_dd": 1.0, "leverage_mult": 0.0, "trade_mult": 0.0},
    }
    
    def __init__(self):
        self.peak_balance: float = 0
        self.current_balance: float = 0
        self.current_level: str = "NORMAL"
        self.level_history: deque = deque(maxlen=100)
        self.recovery_trades: int = 0
    
    def update_balance(self, balance: float) -> Dict[str, Any]:
        """
        Bakiye güncelle ve mod kontrol et.
        """
        self.current_balance = balance
        
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Drawdown hesapla
        drawdown = (self.peak_balance - balance) / self.peak_balance if self.peak_balance > 0 else 0
        
        # Seviye belirle
        old_level = self.current_level
        for level_name, config in self.LEVELS.items():
            if config["min_dd"] <= drawdown < config["max_dd"]:
                self.current_level = level_name
                break
        
        # Seviye değişti mi?
        level_changed = old_level != self.current_level
        if level_changed:
            self.level_history.append({
                "timestamp": time.time(),
                "from": old_level,
                "to": self.current_level,
                "drawdown": drawdown
            })
        
        return {
            "drawdown_pct": round(drawdown * 100, 2),
            "current_level": self.current_level,
            "level_changed": level_changed,
            "peak_balance": self.peak_balance,
            "current_balance": balance,
            "config": self.LEVELS[self.current_level]
        }
    
    def get_risk_adjustments(self) -> Dict[str, float]:
        """Mevcut seviyeye göre risk ayarlamaları."""
        config = self.LEVELS[self.current_level]
        return {
            "leverage_multiplier": config["leverage_mult"],
            "trade_size_multiplier": config["trade_mult"],
            "max_positions": int(5 * config["trade_mult"]),
            "allow_new_trades": config["trade_mult"] > 0
        }
    
    def is_recovery_trade(self, confidence: float) -> bool:
        """
        Recovery trade mi kontrol et.
        Recovery trade = düşük drawdown modunda yüksek confidence trade.
        """
        if self.current_level in ["CAUTION", "WARNING"]:
            return confidence >= 0.75
        return False
    
    def record_recovery_trade(self, pnl_pct: float) -> None:
        """Recovery trade sonucunu kaydet."""
        if pnl_pct > 0:
            self.recovery_trades += 1
            # Başarılı recovery - peak'i güncelle
            if self.recovery_trades >= 3 and self.current_level != "NORMAL":
                # 3 başarılı recovery = seviye düşür
                self.recovery_trades = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Mevcut durumu döndür."""
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        
        return {
            "level": self.current_level,
            "drawdown_pct": round(drawdown * 100, 2),
            "peak_balance": self.peak_balance,
            "current_balance": self.current_balance,
            "recovery_trades": self.recovery_trades,
            "adjustments": self.get_risk_adjustments(),
            "recent_level_changes": list(self.level_history)[-5:]
        }


# ==============================================================================
# 17. LIQUIDITY ANALYZER
# ==============================================================================

class LiquidityAnalyzer:
    """
    Likidite analizi.
    
    Düşük likidite = yüksek slippage riski = küçük pozisyon.
    """
    
    def __init__(self):
        self.liquidity_cache: Dict[str, Dict] = {}
    
    def analyze(
        self,
        symbol: str,
        orderbook_depth: List[List[float]],
        avg_daily_volume: float,
        position_size_usd: float
    ) -> Dict[str, Any]:
        """
        Likidite analizi yap.
        
        Args:
            symbol: Trading pair
            orderbook_depth: [[price, size], ...] 
            avg_daily_volume: Ortalama günlük hacim (USD)
            position_size_usd: Planlanan pozisyon büyüklüğü (USD)
        
        Returns:
            {"liquidity_score": float, "recommended_size_mult": float, "warnings": List}
        """
        warnings = []
        
        # 1. Orderbook depth analizi
        if orderbook_depth:
            total_depth_usd = sum(level[0] * level[1] for level in orderbook_depth[:20])
        else:
            total_depth_usd = 0
        
        # 2. Volume bazlı kontrol
        volume_ratio = position_size_usd / avg_daily_volume if avg_daily_volume > 0 else 1
        
        # Pozisyon günlük hacmin %1'inden fazlaysa uyarı
        if volume_ratio > 0.01:
            warnings.append("position_exceeds_1pct_daily_volume")
        
        # 3. Depth bazlı kontrol
        depth_ratio = position_size_usd / total_depth_usd if total_depth_usd > 0 else 1
        
        # Pozisyon depth'in %5'inden fazlaysa uyarı
        if depth_ratio > 0.05:
            warnings.append("position_exceeds_5pct_orderbook_depth")
        
        # 4. Likidite skoru hesapla (0-1)
        # Düşük volume_ratio ve depth_ratio = yüksek skor
        volume_score = max(0, 1 - (volume_ratio * 50))  # %2'de sıfır
        depth_score = max(0, 1 - (depth_ratio * 10))     # %10'da sıfır
        
        liquidity_score = (volume_score * 0.6 + depth_score * 0.4)
        
        # 5. Önerilen pozisyon çarpanı
        if liquidity_score >= 0.8:
            size_mult = 1.0
        elif liquidity_score >= 0.6:
            size_mult = 0.8
        elif liquidity_score >= 0.4:
            size_mult = 0.5
        elif liquidity_score >= 0.2:
            size_mult = 0.3
        else:
            size_mult = 0.1
            warnings.append("very_low_liquidity")
        
        result = {
            "liquidity_score": round(liquidity_score, 3),
            "recommended_size_mult": size_mult,
            "total_depth_usd": round(total_depth_usd, 2),
            "avg_daily_volume": round(avg_daily_volume, 2),
            "volume_ratio": round(volume_ratio * 100, 4),
            "depth_ratio": round(depth_ratio * 100, 4),
            "warnings": warnings
        }
        
        # Cache'e kaydet
        self.liquidity_cache[symbol] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
    
    def get_cached(self, symbol: str, max_age_sec: int = 300) -> Optional[Dict]:
        """Cache'den likidite verisi al."""
        cached = self.liquidity_cache.get(symbol)
        if cached and (time.time() - cached["timestamp"]) < max_age_sec:
            return cached["data"]
        return None


# ==============================================================================
# 18. SLIPPAGE TRACKER
# ==============================================================================

class SlippageTracker:
    """
    Slippage takibi ve analizi.
    """
    
    def __init__(self, save_path: str = "metrics/slippage_history.json"):
        self.save_path = Path(save_path)
        self.history: deque = deque(maxlen=500)
        self._load()
    
    def _load(self) -> None:
        try:
            if self.save_path.exists():
                data = json.loads(self.save_path.read_text())
                for record in data.get("history", []):
                    self.history.append(record)
        except Exception:
            pass
    
    def _save(self) -> None:
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"history": list(self.history)}
            self.save_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
    
    def record(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
        size_usd: float
    ) -> Dict[str, Any]:
        """
        Slippage kaydı ekle.
        """
        if expected_price <= 0:
            return {"error": "invalid_expected_price"}
        
        # Slippage hesapla
        if side == "long":
            slippage_pct = (actual_price - expected_price) / expected_price
        else:
            slippage_pct = (expected_price - actual_price) / expected_price
        
        slippage_usd = size_usd * abs(slippage_pct)
        
        record = {
            "symbol": symbol,
            "side": side,
            "expected_price": expected_price,
            "actual_price": actual_price,
            "slippage_pct": round(slippage_pct * 100, 4),
            "slippage_usd": round(slippage_usd, 4),
            "size_usd": size_usd,
            "timestamp": time.time(),
            "favorable": slippage_pct < 0  # Negatif slippage = iyi
        }
        
        self.history.append(record)
        self._save()
        
        return record
    
    def get_stats(self, symbol: str = None, last_n: int = 100) -> Dict[str, Any]:
        """
        Slippage istatistikleri.
        """
        records = list(self.history)
        
        if symbol:
            records = [r for r in records if r.get("symbol") == symbol]
        
        records = records[-last_n:]
        
        if not records:
            return {"sufficient_data": False}
        
        slippages = [r["slippage_pct"] for r in records]
        
        return {
            "sufficient_data": True,
            "count": len(records),
            "avg_slippage_pct": round(statistics.mean(slippages), 4),
            "median_slippage_pct": round(statistics.median(slippages), 4),
            "max_slippage_pct": round(max(slippages), 4),
            "min_slippage_pct": round(min(slippages), 4),
            "total_slippage_usd": round(sum(r["slippage_usd"] for r in records), 2),
            "favorable_pct": round(sum(1 for r in records if r["favorable"]) / len(records) * 100, 1)
        }
    
    def get_symbol_ranking(self) -> List[Dict]:
        """Sembolleri slippage'a göre sırala."""
        symbol_stats = {}
        
        for record in self.history:
            sym = record.get("symbol")
            if sym not in symbol_stats:
                symbol_stats[sym] = {"total": 0, "count": 0}
            symbol_stats[sym]["total"] += record["slippage_pct"]
            symbol_stats[sym]["count"] += 1
        
        ranking = []
        for sym, stats in symbol_stats.items():
            avg = stats["total"] / stats["count"] if stats["count"] > 0 else 0
            ranking.append({
                "symbol": sym,
                "avg_slippage_pct": round(avg, 4),
                "trade_count": stats["count"]
            })
        
        # En düşük slippage önce
        ranking.sort(key=lambda x: x["avg_slippage_pct"])
        return ranking


# ==============================================================================
# ENTEGRASYON YARDIMCILARI
# ==============================================================================

def apply_enhancements_to_master(
    master_raw: float,
    direction: str,
    ai_results: Dict[str, Any],
    ta_pack: Dict[str, Any],
    bilstm_prob: float,
    performance_tracker: Optional[PerformanceTracker] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Tüm iyileştirmeleri master score'a uygula.
    
    Returns:
        (adjusted_master, details)
    """
    details = {
        "original_master": master_raw,
        "adjustments": []
    }
    
    adjusted = master_raw
    
    # 1. AI Agreement Bonus
    chatgpt_res = ai_results.get("chatgpt", {})
    deepseek_res = ai_results.get("deepseek", {})
    
    agreement = calculate_ai_agreement_bonus(chatgpt_res, deepseek_res, bilstm_prob)
    if agreement["bonus"] > 0:
        adjusted += agreement["bonus"]
        details["ai_agreement"] = agreement
        details["adjustments"].append(f"ai_agreement: +{agreement['bonus']:.3f}")
    
    # 2. MTF Confirmation
    mtf = check_mtf_alignment(ta_pack, direction)
    if mtf["approved"]:
        mtf_bonus = (mtf["score"] - 0.5) * 0.10  # Max ±5%
        adjusted += mtf_bonus
        details["mtf"] = mtf
        details["adjustments"].append(f"mtf: {mtf_bonus:+.3f}")
    elif not mtf["approved"] and mtf["score"] < 0.4:
        # MTF rejection penalty
        adjusted *= 0.95
        details["mtf"] = mtf
        details["adjustments"].append("mtf_rejection: ×0.95")
    
    # 3. Performance-based adjustments
    if performance_tracker:
        perf_adj = performance_tracker.get_adjustments()
        
        # Threshold adjustment (applied externally, just note it)
        details["performance_adj"] = perf_adj
        
        # Direction bias
        if direction == "long":
            adjusted += perf_adj.get("long_bias", 0)
        else:
            adjusted += perf_adj.get("short_bias", 0)
        
        if perf_adj.get("long_bias", 0) != 0 or perf_adj.get("short_bias", 0) != 0:
            details["adjustments"].append(f"direction_bias: {direction}")
    
    # Clamp
    adjusted = max(0.0, min(1.0, adjusted))
    details["final_master"] = adjusted
    details["total_adjustment"] = adjusted - master_raw
    
    return adjusted, details


# ==============================================================================
# SINGLETON INSTANCES
# ==============================================================================

# Global instances (import edildiğinde kullanılabilir)
_trailing_manager: Optional[TrailingStopManager] = None
_performance_tracker: Optional[PerformanceTracker] = None
_volume_detector: Optional[VolumeSpikeDetector] = None
_funding_analyzer: Optional[FundingRateAnalyzer] = None
_daily_summary: Optional[DailySummaryGenerator] = None
_auto_tuner: Optional[AutoParameterTuner] = None
_hedger: Optional[PositionHedger] = None
_drawdown_manager: Optional[DrawdownRecoveryManager] = None
_liquidity_analyzer: Optional[LiquidityAnalyzer] = None
_slippage_tracker: Optional[SlippageTracker] = None


def get_trailing_manager() -> TrailingStopManager:
    global _trailing_manager
    if _trailing_manager is None:
        _trailing_manager = TrailingStopManager()
    return _trailing_manager


def get_performance_tracker() -> PerformanceTracker:
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker


def get_volume_detector() -> VolumeSpikeDetector:
    global _volume_detector
    if _volume_detector is None:
        _volume_detector = VolumeSpikeDetector()
    return _volume_detector


def get_funding_analyzer() -> FundingRateAnalyzer:
    global _funding_analyzer
    if _funding_analyzer is None:
        _funding_analyzer = FundingRateAnalyzer()
    return _funding_analyzer


def get_daily_summary() -> DailySummaryGenerator:
    global _daily_summary
    if _daily_summary is None:
        _daily_summary = DailySummaryGenerator()
    return _daily_summary


def get_auto_tuner() -> AutoParameterTuner:
    global _auto_tuner
    if _auto_tuner is None:
        _auto_tuner = AutoParameterTuner()
    return _auto_tuner


def get_hedger() -> PositionHedger:
    global _hedger
    if _hedger is None:
        _hedger = PositionHedger()
    return _hedger


def get_drawdown_manager() -> DrawdownRecoveryManager:
    global _drawdown_manager
    if _drawdown_manager is None:
        _drawdown_manager = DrawdownRecoveryManager()
    return _drawdown_manager


def get_liquidity_analyzer() -> LiquidityAnalyzer:
    global _liquidity_analyzer
    if _liquidity_analyzer is None:
        _liquidity_analyzer = LiquidityAnalyzer()
    return _liquidity_analyzer


def get_slippage_tracker() -> SlippageTracker:
    global _slippage_tracker
    if _slippage_tracker is None:
        _slippage_tracker = SlippageTracker()
    return _slippage_tracker


# ==============================================================================
# CONVENIENCE EXPORTS
# ==============================================================================

__all__ = [
    # Fonksiyonlar
    "calculate_ai_agreement_bonus",
    "check_mtf_alignment",
    "get_smart_tp_levels",
    "check_correlation_limit",
    "analyze_orderbook_imbalance",
    "check_time_based_exit",
    "calculate_dynamic_threshold",
    "calculate_optimal_entry",
    "apply_enhancements_to_master",
    
    # Sınıflar
    "TrailingStopManager",
    "PerformanceTracker",
    "VolumeSpikeDetector",
    "FundingRateAnalyzer",
    "DailySummaryGenerator",
    "AutoParameterTuner",
    "PositionHedger",
    "DrawdownRecoveryManager",
    "LiquidityAnalyzer",
    "SlippageTracker",
    
    # Singleton getters
    "get_trailing_manager",
    "get_performance_tracker",
    "get_volume_detector",
    "get_funding_analyzer",
    "get_daily_summary",
    "get_auto_tuner",
    "get_hedger",
    "get_drawdown_manager",
    "get_liquidity_analyzer",
    "get_slippage_tracker",
]
