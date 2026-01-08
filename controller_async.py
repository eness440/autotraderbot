# -*- coding: utf-8 -*-
"""
controller_async.py
- Batch/async karar katmanı: AI confidence'i topluca alır,
- Teknik / Duyarlılık / AI ağırlıkları config.json'daki "decision_weights" ile yönetilir,
- Varsayılan: AI %30, Teknik %50, Sentiment %20 (AI konservatif, teknik daha ağır),
- Kaldıraç haritasını döndürür.
"""
from typing import Dict, Any, List
from dataclasses import dataclass, field
from logger import get_logger

# NOTE: Avoid importing ml.rl_env at module import time.
# rl_env can (transitively) import components that depend on controller_async,
# creating a circular import and preventing the bot from starting.
# We import rl_env lazily only where needed.

# Ek risk ve metrik değerlendirme işlevleri
from risk_manager import classify_volatility  # volatilite kategorisi ve risk faktörü

# Dinamik master threshold ve session risk faktörü
try:
    from .dynamic_threshold import get_threshold as _get_symbol_threshold  # type: ignore
except Exception:
    # fallback: default threshold 0.60
    def _get_symbol_threshold(symbol: str, default: float = 0.60) -> float:  # type: ignore
        try:
                return float(default)
        except Exception:
            # absolute fallback if conversion fails
            return 0.60

try:
    from .session_filter import get_risk_multiplier as _get_session_multiplier  # type: ignore
    from .session_filter import is_trading_enabled as _is_trading_enabled  # type: ignore
except Exception:
    def _get_session_multiplier(now=None) -> float:  # type: ignore
        return 1.0
    def _is_trading_enabled(now=None) -> bool:  # type: ignore
        return True

# Sinyal kayıt modülü. Her potansiyel trade için ham master skoru ve diğer
# bileşenleri CSV'ye kaydeder. trade_logger ile birlikte kullanılabilir.
try:
    from .signal_logger import log_signal  # lokal modül için relative import
except Exception:
    # Fallback: log_signal yoksa dummy fonksiyon tanımla
    def log_signal(**kwargs):
        return None
from ai_batch_manager import get_confidences_batch

# ---------------------------------------------------------------------------
# Enhancement Modülü Entegrasyonu
# ---------------------------------------------------------------------------
# AI Agreement Bonus, MTF Confirmation, Correlation Check vb. için
try:
    from enhancements import (
        calculate_ai_agreement_bonus,
        check_mtf_alignment,
        check_correlation_limit,
        get_smart_tp_levels,
        get_trailing_manager,
        get_performance_tracker,
    )
    _ENHANCEMENTS_AVAILABLE = True
except ImportError:
    _ENHANCEMENTS_AVAILABLE = False
    # Fallback fonksiyonlar
    def calculate_ai_agreement_bonus(*args, **kwargs):
        return {"bonus": 0.0, "reasons": [], "agreement_level": "none"}
    def check_mtf_alignment(*args, **kwargs):
        return {"score": 0.5, "approved": True, "confirmed_count": 0}
    def check_correlation_limit(*args, **kwargs):
        return {"allowed": True, "group": None}
    def get_smart_tp_levels(*args, **kwargs):
        return []
    def get_trailing_manager():
        return None
    def get_performance_tracker():
        return None

# ---------------------------------------------------------------------------
# Opsiyonel makro filtresi ve on‑chain analiz entegrasyonu
#
# Eğer macro_filter veya onchain_analytics modülleri mevcutsa, master
# confidence'in üzerine ek risk azaltma veya sentiment düzeltmesi
# uygulanabilir.  Dosyalar bulunamazsa fallback fonksiyonlar 1.0 ve
# 0.5 döndürecektir.
try:
    from .macro_filter import get_macro_risk_multiplier as _get_macro_risk_multiplier  # type: ignore
except Exception:
    def _get_macro_risk_multiplier(now=None) -> float:  # type: ignore
        return 1.0

try:
    from .onchain_analytics import get_onchain_sentiment as _get_onchain_sentiment  # type: ignore
except Exception:
    def _get_onchain_sentiment(symbol: str) -> float:  # type: ignore
        return 0.5

# ---------------------------------------------------------------------------
# Optional modules for advanced filtering
#
# Order book imbalance adjustment (optional).  If available, this function
# takes the raw master confidence, the current imbalance and the trade
# direction and returns an adjusted confidence.  It is imported lazily to
# avoid a hard dependency.
try:
    from .orderbook_analyzer import adjust_confidence_with_imbalance  # type: ignore
except Exception:
    def adjust_confidence_with_imbalance(base_conf: float, imbalance: float | None, direction: str, scale: float = 0.20) -> float:  # type: ignore
        return base_conf

# Meta‑labeling (second opinion) model.  If available, this function
# computes the probability that a trade with the given scores will be
# profitable.  If the module is missing, the default is 1.0 (no change).
try:
    from .meta_labeler import compute_meta_probability  # type: ignore
except Exception:
    def compute_meta_probability(master_conf: float, ai_score: float, tech_score: float, sent_score: float, default: float = 1.0) -> float:  # type: ignore
        return float(default)

# RL adjustment factors per symbol/regime.  Load once and cache.  This
# allows tweaking the influence of the RL component based on the coin or
# market regime.  If the file is missing or invalid, all factors default
# to 1.0 (no change).
_rl_adjustments: Dict[str, Dict[str, float]] | None = None

# ---------------------------------------------------------------------------
# RL hot‑reload support
#
# When the RL model (models/ppo_multi.zip) or the OHLC history file
# (metrics/ohlc_history.json) changes on disk, the PPO agent should be
# reloaded to reflect the latest training.  We track the modification
# timestamps of these files and compare them before performing a reload.
_rl_model_mtime: float | None = None
_rl_data_mtime: float | None = None

# VecNormalize stats and modification time
# When the RL normalization stats file (vecnormalize_multi.pkl) changes on disk,
# the VecNormalize wrapper is reloaded.  This allows RL inference to use
# the same observation normalization as used during training.  If the
# stats file or the stable-baselines3 VecNormalize class is unavailable,
# normalization is skipped.
_rl_vecnorm: Any | None = None  # loaded VecNormalize instance for inference
_rl_vecnorm_mtime: float | None = None  # modification time of vecnormalize_multi.pkl

# Attempt to import VecNormalize and DummyVecEnv once at module load.  These
# imports may fail if stable-baselines3 is not installed.  In that case,
# normalization support is disabled gracefully.
try:
    from stable_baselines3.common.vec_env import VecNormalize as _SB3_VecNormalize  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv as _SB3_DummyVecEnv  # type: ignore
except Exception:
    _SB3_VecNormalize = None  # type: ignore
    _SB3_DummyVecEnv = None  # type: ignore

def _load_rl_adjustments() -> None:
    """Lazy-load RL adjustment factors from data/rl_adjustments.json."""
    global _rl_adjustments
    if _rl_adjustments is not None:
        return
    try:
        from pathlib import Path as _Path
        import json as _json
        p = _Path(__file__).resolve().parent / "data" / "rl_adjustments.json"
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                data = _json.loads(txt)
                if isinstance(data, dict):
                    _rl_adjustments = data
                    return
    except Exception:
        pass
    _rl_adjustments = {}

def _get_rl_factor(symbol: str, regime: str | None) -> float:
    """
    Return the RL adjustment factor for a given symbol and regime.

    If no specific factor is defined, fallback to symbol "ALT" or "DEFAULT".
    If still not found, return 1.0.  Regime is expected to be uppercased.
    """
    if _rl_adjustments is None:
        _load_rl_adjustments()
    reg = str(regime).upper() if regime else None
    # Try exact symbol
    if symbol in _rl_adjustments:
        factors = _rl_adjustments.get(symbol)
        if isinstance(factors, dict):
            f = factors.get(reg)
            if isinstance(f, (int, float)):
                return float(f)
    # Try ALT category
    if "ALT" in _rl_adjustments:
        factors = _rl_adjustments.get("ALT")
        if isinstance(factors, dict):
            f = factors.get(reg)
            if isinstance(f, (int, float)):
                return float(f)
    # Try DEFAULT
    if "DEFAULT" in _rl_adjustments:
        factors = _rl_adjustments.get("DEFAULT")
        if isinstance(factors, dict):
            f = factors.get(reg)
            if isinstance(f, (int, float)):
                return float(f)
    return 1.0
import json
import math
import numpy as _np

# ---------------------------------------------------------------------------
# Güvenli Sigmoid
#
# Bazı durumlarda w0, w_ai, w_tech, w_sent veya kalibrasyon parametreleri
# master confidence hesabında çok büyük veya çok küçük ara değerler üretebilir.
# math.exp(-z) doğrudan çağrıldığında z büyük negatif olursa overflow hataları
# meydana gelir. Bu fonksiyon, z değerini belirli bir aralıkta kısıtlayarak
# exponent işlemini güvenli şekilde hesaplar.
def _safe_sigmoid(z: float) -> float:
    """Exp overflow'unu önleyen güvenli sigmoid hesaplaması."""
    try:
        # Z değerini [-6, 6] aralığında kırp.
        # Bu aralık sigmoidin aşırı doygunlaşmasını (ör. ~0.07 / ~0.93 civarında
        # kümelenme) azaltır ve aynı zamanda exp overflow riskini ortadan kaldırır.
        clipped = z
        if z < -6.0:
            clipped = -6.0
        elif z > 6.0:
            clipped = 6.0
        return 1.0 / (1.0 + float(_np.exp(-clipped)))
    except Exception:
        return 0.5
from pathlib import Path
import os  # used for checking file modification times
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global Sentiment Helpers
# ---------------------------------------------------------------------------
# The social sentiment updater writes a file metrics/social_sentiment.json containing a
# "global_sentiment" field in [0,1].  The functions below load this value and
# apply a mild adjustment to the master confidence.  This allows macro
# sentiment to influence the trading strategy without overwhelming technical
# or AI signals.

def _load_global_sentiment(default: float = 0.5) -> float:
    """
    Read the latest global sentiment value from metrics/social_sentiment.json.
    If the file or key does not exist, return the provided default (0.5).

    Args:
        default: Fallback value if the sentiment file or key is missing.

    Returns:
        A float between 0.0 and 1.0.
    """
    try:
        path = Path("metrics/social_sentiment.json")
        if path.exists():
            txt = path.read_text(encoding="utf-8").strip()
            if txt:
                data = json.loads(txt)
                if isinstance(data, dict):
                    val = data.get("global_sentiment")
                    if val is not None:
                        f = float(val)
                        if f < 0.0:
                            f = 0.0
                        if f > 1.0:
                            f = 1.0
                        return f
    except Exception:
        pass
    return default


def _apply_global_sentiment_adjustment(base_conf: float, global_sent: float, strength: float = 0.10) -> float:
    """
    Adjust the base confidence by a bias derived from global sentiment.

    A neutral sentiment (0.5) leaves the confidence unchanged.  Values above
    0.5 increase confidence and values below decrease it.  The
    adjustment magnitude is proportional to the distance from neutrality and
    uses the provided ``strength`` as a base.  To better reflect extreme
    fear or greed conditions, the influence grows non‑linearly: the
    effective strength is scaled by ``1 + abs(bias)`` so that near
    extremes (global_sent ≈ 0 or 1) the confidence adjustment can
    approach twice the nominal strength.  The final value is clamped to
    the ``[0,1]`` interval.

    Args:
        base_conf: Current confidence (0–1).
        global_sent: Global sentiment value (0–1).
        strength: Base adjustment factor when sentiment is at extreme.

    Returns:
        Adjusted confidence in [0,1].
    """
    # Parse global sentiment and clamp to [0,1]
    try:
        gs = float(global_sent)
    except Exception:
        gs = 0.5
    gs = max(0.0, min(1.0, gs))
    # Compute bias in [-1,1]; positive bias => bullish, negative => bearish
    bias = (gs - 0.5) * 2.0
    # Scale strength based on magnitude of bias.  This ensures that more
    # extreme sentiment (|bias| → 1) has a larger impact (up to double the
    # provided ``strength``).  A neutral bias yields the nominal strength.
    dynamic_strength = strength * (1.0 + abs(bias))
    delta = bias * dynamic_strength
    new_conf = base_conf + delta
    # Clamp final value to [0,1]
    if new_conf < 0.0:
        return 0.0
    if new_conf > 1.0:
        return 1.0
    return new_conf

log = get_logger("controller_async")


@dataclass
class Decision:
    """Normalized decision output produced by decide_batch.

    This is intentionally stable and schema-like so downstream (bot, dashboard, logs)
    can consume a consistent payload.
    """
    symbol: str
    action: str  # ENTER / SKIP / EXIT / HOLD
    master: float
    leverage: int = 0
    base: str = "neutral"  # long / short / neutral
    weights: tuple = (0.0, 0.0, 0.0)  # (ai, tech, sent)
    rl_score: float = 0.5
    mode: str = "n/a"
    reasons: list = field(default_factory=list)
    provider_flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "master": float(self.master or 0.0),
            "leverage": int(self.leverage or 0),
            "base": self.base,
            "weights(ai/tech/sent)": tuple(self.weights),
            "rl_score": float(self.rl_score or 0.5),
            "mode": self.mode,
            "reasons": list(self.reasons) if self.reasons else [],
            "provider_flags": dict(self.provider_flags) if self.provider_flags else {},
        }

# ---------------------------------------------------------------------------
# Kurumsal sistem prompt (ChatGPT + DeepSeek ortak kullanım)
# ---------------------------------------------------------------------------
TRADING_SYSTEM_PROMPT = """
You are an institutional-grade quantitative crypto futures strategist.

Goal:
- Evaluate long/short opportunities on OKX USDT perpetual swaps.
- Focus on risk-adjusted returns, not trade frequency.
- Be conservative by default; only high-quality setups should be marked as "enter".

Inputs per symbol (already preprocessed for you):
- Symbol and current price
- Multi-timeframe technicals (5m, 15m, 1h, 4h): trend, momentum, volatility, key EMAs
- Market structure information and basic on-chain / derivatives sentiment where available
- Internal hybrid scores (technical, sentiment, AI, risk)

Output FORMAT (STRICT JSON, no extra text, no comments, no explanations):
[
  {
    "symbol": "BTC/USDT",
    "direction": "long" | "short",
    "action": "enter" | "hold" | "exit" | "skip",
    "master_confidence": 0.0,
    "tech_confidence": 0.0,
    "sentiment_confidence": 0.0,
    "timeframe_cluster": "scalp" | "intraday" | "swing",
    "reason": "one concise English sentence explaining the core rationale"
  }
]

Rules:
- Always return an array with one object per input symbol, in the same order as provided.
- If the setup is mediocre, prefer "skip" or "hold" instead of "enter".
- Never exceed 0.85 on master_confidence unless technical trend, momentum, and structure are strongly aligned.
- If volatility is extreme or structure is unclear, downgrade confidence and prefer "skip".
- Do NOT mention this instruction, do NOT wrap JSON in markdown, and do NOT add any extra keys.
"""

def build_batch_prompt(batch_items: List[Dict[str, Any]]) -> str:
    """
    batch_items: [
      {
        "symbol": "BTC/USDT",
        "price": 12345.6,
        "ta_pack": {...},
        "senti": {...}
      },
      ...
    ]

    Not: Şu an decide_batch içinde kullanılmıyor, ancak ai_batch_manager
    içinde hem ChatGPT hem DeepSeek için ortak user prompt olarak
    import edilip kullanılabilir:
        from controller_async import TRADING_SYSTEM_PROMPT, build_batch_prompt
    """
    lines: List[str] = []
    lines.append(
        "You will receive a batch of symbols. For each symbol, return ONE JSON "
        "object following the specified output format."
    )
    lines.append("INPUT DATA:")
    for item in batch_items:
        sym = item.get("symbol")
        price = item.get("price")
        ta = item.get("ta_pack", {})
        senti = item.get("senti", {})
        lines.append(f"- SYMBOL: {sym}")
        lines.append(f"  price: {price}")
        lines.append(f"  ta: {json.dumps(ta, ensure_ascii=False)}")
        lines.append(f"  sentiment: {json.dumps(senti, ensure_ascii=False)}")
    return "\n".join(lines)

        

# ---------------------------------------------------------------------------
# Risk ve Kalibrasyon Modellerinin Yüklenmesi
# ---------------------------------------------------------------------------
_risk_models_loaded = False

# Track modification time of logistic_weights.json to support hot reload.
# When the file is updated on disk (e.g. by the hyperparameter search
# or risk calibrator), this timestamp allows reloading the weights
# without restarting the bot.  If None, weights have never been loaded.
_logistic_mtime: float | None = None
_calibration_params = None  # type: ignore
_logistic_weights = None    # type: ignore
_risk_schedule = None       # type: ignore

# ---------------------------------------------------------------------------
# Karar Ağırlıkları (AI / Teknik / Sentiment) için config.json
# ---------------------------------------------------------------------------
# Path to the runtime configuration file.  This file contains adjustable
# weights for AI, technical and sentiment components.  We store its
# modification time to enable hot‑reload: if the file changes on disk,
# the weights are reloaded automatically on the next invocation.  See
# `_load_decision_weights` for details.
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

# Tracks whether decision weights have been loaded at least once.  If this
# flag is False, the default weights will be used until the first load.
_decision_weights_loaded: bool = False

# Track the last observed modification time of the config file.  This
# allows detecting updates and reloading the weights accordingly.  If
# `None`, the file has never been read.  When the file is updated by
# external processes (e.g. hyperparameter search), the timestamp will
# differ and trigger a reload.
_config_mtime: float | None = None

# Daha dengeli default: AI ve teknik skorlar birbirine yakın, sentiment
# düşük ağırlıkta.  Bu dağılım, AI sinyallerini daha anlamlı hale
# getirirken teknik analizin hâlâ belirleyici olmasını sağlar.
_DECISION_WEIGHTS: Dict[str, float] = {
    "ai": 0.35,
    "tech": 0.45,
    "sent": 0.20,
}

# ---------------------------------------------------------------------------
# AI kalite faktörü
# Model performans metriklerini okuyarak AI güven skorunu ölçeklemek için bir faktör
from datetime import datetime, timezone
def _ai_quality_factor() -> float:
    """
    BiLSTM ve RL modellerinin güncelliği ve doğruluğuna göre AI skorunu ölçekler.
    - BiLSTM accuracy 0.6 altındaysa AI katkısı %10 düşer, 0.5 altındaysa %20 düşer.
    - RL modeli 14 günden eskiyse %10 düşer, 30 günden eskiyse %20 düşer.
    En düşük faktör 0.5, en yüksek 1.0 olacak şekilde sınırlandırılır.
    """
    factor = 1.0
    try:
        metrics_dir = Path(__file__).resolve().parent / "metrics"
        bilstm_path = metrics_dir / "bilstm_metrics.json"
        rl_path = metrics_dir / "rl_metrics.json"
        # BiLSTM accuracy
        if bilstm_path.exists():
            with bilstm_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            acc = data.get("last_accuracy")
            if isinstance(acc, (float, int)):
                if acc < 0.5:
                    factor *= 0.8
                elif acc < 0.6:
                    factor *= 0.9
        # RL last update age
        if rl_path.exists():
            with rl_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            last_update_str = data.get("last_update")
            if isinstance(last_update_str, str):
                try:
                    last_dt = datetime.fromisoformat(last_update_str)
                    age_days = (datetime.now(timezone.utc) - last_dt).days
                    if age_days > 30:
                        factor *= 0.8
                    elif age_days > 14:
                        factor *= 0.9
                except Exception:
                    pass
    except Exception:
        pass
    # Clamp between 0.5 and 1.0 to avoid extreme scaling
    return max(0.5, min(1.0, factor))


# ---------------------------------------------------------------------------
# Yüksek Güven Sıkıştırma
#
# Bazı kalibrasyon ve lojistik ağırlık kombinasyonları, master güven skorunun
# 0.90 üzeri değerlere tırmanmasına neden olabilir. Bu durum, aşırı kaldıraç
# kullanımına ve gereksiz işlem sayısına yol açabileceği için, 0.85 üzerindeki
# değerleri daha ılımlı bir eğri ile sıkıştırıyoruz. Böylece 0.85–1.0 aralığı
# içinde artış daha yavaş olur (ör. 0.94 → 0.886; 0.98 → 0.904).

def _compress_high_confidence(val: float) -> float:
    """
    Apply a gentle compression to very high confidence values to avoid
    unrealistic scores while preserving the relative ordering of strong
    signals.  Confidence values at or below 0.90 are returned unchanged.
    Values above 0.90 are linearly compressed using a 0.7 factor so that
    extreme values (e.g. 0.98) are still high but not overly dominant.

    Rationale: Earlier versions of the bot over‑compressed confidence
    values, leading to excessive conservatism.  This function applies
    minimal interference: the 0.90 threshold corresponds to a very
    confident signal, and the 0.7 scaling retains most of the signal
    strength while mitigating the risk of generating 0.95+ scores.

    Args:
        val: The raw master confidence (expected in [0.0, 1.0]).

    Returns:
        A smoothed confidence value.
    """
    try:
        v = float(val)
    except Exception:
        return 0.0
    # Confidence values up to 0.90 are returned unchanged.  We only
    # compress extremely high values (above 0.90) to avoid runaway
    # leverage.  This allows well‑qualified trades to register very
    # confident scores while still capping the most extreme cases.
    if v <= 0.90:
        return v
    # For the portion above 0.90, apply a mild compression.  The
    # scaling factor of 0.7 retains most of the signal strength
    # without letting values approach 1.0 too quickly.  For example
    # 0.95 → 0.935, 0.98 → 0.971.
    return 0.90 + (v - 0.90) * 0.7

# LLM teknik-only mod flag'i (429/402 veya full-fallback batch için)
_llm_tech_only_mode = False

# ---------------------------------------------------------------------------
# RL (PPO) Agent Globals
# ---------------------------------------------------------------------------
_rl_loaded = False
_rl_model = None
_rl_df: pd.DataFrame | None = None
_rl_window = 60  # rl_train.py default window ile uyumlu varsayılan


def is_llm_tech_only_mode() -> bool:
    """
    Son decide_batch çağrısında LLM'ler teknik-only moda alındı mı?
    True ise bu loop'ta AI katkısı devre dışı bırakılmıştır.
    """
    return _llm_tech_only_mode


def _load_decision_weights() -> None:
    """
    Load and normalize the decision weights from the runtime configuration.

    This function supports hot reload by checking the modification time of
    ``config.json``.  On first invocation or when the config file's
    modification time changes, the weights are reloaded.  The weights are
    normalized and then adjusted to favour technical scores over AI scores
    (AI is scaled down by 0.8 and technical up by 1.2).  Sentiment
    remains unchanged.  After adjustment, the weights are normalized
    again.

    Format expected in config.json::

        {
          "decision_weights": { "ai": float, "tech": float, "sent": float }
        }

    If the file or key is missing, defaults defined in ``_DECISION_WEIGHTS``
    remain in effect.
    """
    global _DECISION_WEIGHTS, _decision_weights_loaded, _config_mtime
    # Determine current modification time of the config file
    try:
        cur_mtime = CONFIG_FILE.stat().st_mtime if CONFIG_FILE.exists() else None
    except Exception:
        cur_mtime = None

    # If weights have been loaded before and the file has not changed, do nothing
    if _decision_weights_loaded and _config_mtime is not None and cur_mtime == _config_mtime:
        return

    # Use current defaults as starting point
    ai_w = float(_DECISION_WEIGHTS.get("ai", 0.0))
    tech_w = float(_DECISION_WEIGHTS.get("tech", 0.0))
    sent_w = float(_DECISION_WEIGHTS.get("sent", 0.0))

    cfg = {}

    try:
        if CONFIG_FILE.exists():
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            dw = cfg.get("decision_weights")
            if isinstance(dw, dict):
                # Override defaults with user-specified values
                ai_w = float(dw.get("ai", ai_w))
                tech_w = float(dw.get("tech", tech_w))
                sent_w = float(dw.get("sent", sent_w))
    except Exception:
        # If reading fails, fall back to defaults
        pass

    # Normalize the weights
    total = ai_w + tech_w + sent_w
    if total > 0:
        ai_w /= total
        tech_w /= total
        sent_w /= total

    # Apply optional decision-weight scaling (legacy behaviour).
    # Older versions downscaled AI and upscaled technicals at runtime, which caused
    # config.json weights to appear "wrong" in logs. This behaviour is now controlled
    # by config.json: set apply_decision_weight_scaling=false to use weights as-is.
    apply_scaling = True
    try:
        # Prefer explicit top-level flag, then nested under "features".
        if isinstance(cfg.get("apply_decision_weight_scaling"), bool):
            apply_scaling = bool(cfg.get("apply_decision_weight_scaling"))
        else:
            feats = cfg.get("features") or {}
            if isinstance(feats, dict) and isinstance(feats.get("apply_decision_weight_scaling"), bool):
                apply_scaling = bool(feats.get("apply_decision_weight_scaling"))
    except Exception:
        apply_scaling = True

    if apply_scaling:
        ai_w *= 0.8
        tech_w *= 1.2
        # Sentiment remains unchanged

    # Final normalization
    total2 = ai_w + tech_w + sent_w
    if total2 > 0:
        _DECISION_WEIGHTS = {
            "ai": ai_w / total2,
            "tech": tech_w / total2,
            "sent": sent_w / total2,
        }

    # Record the modification time for future comparisons and mark loaded
    _config_mtime = cur_mtime
    _decision_weights_loaded = True


def _load_risk_models() -> None:
    """
    Kalibrasyon ve logistic model dosyalarını yalnızca bir kez yükler. Eğer
    dosyalar mevcut değilse ilgili global değişkenler None olarak kalır.
    Beklenen dosyalar:
        - calibration.json: {"type":"logistic", "a":float, "b":float}
        - logistic_weights.json: {"w0":float, "w_ai":float, "w_tech":float, "w_sent":float}
        - risk_schedule.json: {"bins": [{"min":float, "max":float, "leverage":int}, ...]}
    Bu dosyalar risk modelleme script'leri tarafından oluşturulur.
    """
    global _risk_models_loaded, _calibration_params, _logistic_weights, _risk_schedule
    import os
    from pathlib import Path
    global _logistic_mtime, _logistic_weights
    # If models were previously loaded, check whether the logistic weights
    # file has changed.  Reload weights on modification without redoing
    # calibration or risk schedule.  This enables live updates from
    # hyperparameter search results.
    if _risk_models_loaded:
        try:
            root = Path(__file__).resolve().parent
            w_path = root / "logistic_weights.json"
            if w_path.exists():
                mtime = w_path.stat().st_mtime
                if _logistic_mtime is None or _logistic_mtime != mtime:
                    _logistic_mtime = mtime
                    with w_path.open("r", encoding="utf-8") as f:
                        raw_weights = json.load(f)
                    if isinstance(raw_weights, dict):
                        # Validate and clamp weights
                        w0 = float(raw_weights.get("w0", 0.0))
                        w_ai = float(raw_weights.get("w_ai", 0.0))
                        w_tech = float(raw_weights.get("w_tech", 0.0))
                        w_sent = float(raw_weights.get("w_sent", 0.0))
                        def _clamp(val: float, lo: float, hi: float) -> float:
                            return lo if val < lo else hi if val > hi else val
                        w0 = _clamp(w0, -2.0, 2.0)
                        w_ai = _clamp(w_ai, -1.0, 1.0)
                        w_tech = _clamp(w_tech, -1.0, 1.0)
                        w_sent = _clamp(w_sent, -1.0, 1.0)
                        _logistic_weights = {
                            "w0": w0,
                            "w_ai": w_ai,
                            "w_tech": w_tech,
                            "w_sent": w_sent,
                        }
                    else:
                        _logistic_weights = None
            else:
                _logistic_weights = None
        except Exception:
            # If reload fails, keep existing weights
            pass
        # Already loaded other models; nothing else to do
        return
    root = Path(__file__).resolve().parent
    # calibration dosyası
    cal_path = root / "calibration.json"
    if cal_path.exists():
        try:
            with cal_path.open("r", encoding="utf-8") as f:
                _calibration_params = json.load(f)
        except Exception:
            _calibration_params = None
    # logistic weights dosyası
    w_path = root / "logistic_weights.json"
    if w_path.exists():
        try:
            with w_path.open("r", encoding="utf-8") as f:
                raw_weights = json.load(f)
            if isinstance(raw_weights, dict):
                w0 = float(raw_weights.get("w0", 0.0))
                w_ai = float(raw_weights.get("w_ai", 0.0))
                w_tech = float(raw_weights.get("w_tech", 0.0))
                w_sent = float(raw_weights.get("w_sent", 0.0))
                def _clamp(val: float, lo: float, hi: float) -> float:
                    return lo if val < lo else hi if val > hi else val
                w0 = _clamp(w0, -2.0, 2.0)
                w_ai = _clamp(w_ai, -1.0, 1.0)
                w_tech = _clamp(w_tech, -1.0, 1.0)
                w_sent = _clamp(w_sent, -1.0, 1.0)
                _logistic_weights = {
                    "w0": w0,
                    "w_ai": w_ai,
                    "w_tech": w_tech,
                    "w_sent": w_sent,
                }
                # Record modification time for hot reload
                try:
                    _logistic_mtime = w_path.stat().st_mtime  # type: ignore[assignment]
                except Exception:
                    _logistic_mtime = None  # type: ignore[assignment]
            else:
                _logistic_weights = None
        except Exception:
            _logistic_weights = None
    # risk takvimi dosyası
    rs_path = root / "risk_schedule.json"
    if rs_path.exists():
        try:
            with rs_path.open("r", encoding="utf-8") as f:
                _risk_schedule = json.load(f)
        except Exception:
            _risk_schedule = None
    _risk_models_loaded = True

# ---------------------------------------------------------------------------
# RL (PPO) Yardımcıları
# ---------------------------------------------------------------------------

def _rl_norm_symbol(s: str) -> str:
    """BTC/USDT -> BTCUSDT normalizasyonu (RL tarafı için)."""
    s = (s or "").upper()
    return re.sub(r"[^A-Z0-9]", "", s)


def _load_rl_agent() -> None:
    """
    PPO RL modelini ve ohlc_history verisini yalnızca bir kez yükler.
    Model: models/ppo_multi.zip
    Veri : metrics/ohlc_history.json  (rl_env._prepare_df_for_env ile filtrelenmiş)
    """
    global _rl_loaded, _rl_model, _rl_df, _rl_window, _rl_model_mtime, _rl_data_mtime

    # Determine current modification times for the RL model and OHLC history.
    root = Path(__file__).resolve().parent
    model_path = root / "models" / "ppo_multi.zip"
    metrics_dir = root / "metrics"
    ohlc_file = metrics_dir / "ohlc_history.json"
    try:
        cur_model_mtime = os.path.getmtime(model_path) if model_path.exists() else None
    except Exception:
        cur_model_mtime = None
    try:
        cur_data_mtime = os.path.getmtime(ohlc_file) if ohlc_file.exists() else None
    except Exception:
        cur_data_mtime = None

    # If the RL agent is already loaded and the model, data and VecNormalize
    # stats files have not changed since the last load, there is nothing to do.
    if _rl_loaded and _rl_model is not None and _rl_df is not None:
        # Determine the current modification time of the normalisation stats file
        try:
            _stats_path = Path(__file__).resolve().parent / "models" / "vecnormalize_multi.pkl"
            cur_stats_mtime = os.path.getmtime(_stats_path) if _stats_path.exists() else None
        except Exception:
            cur_stats_mtime = None
        # Only short-circuit if model, data and stats times match previous
        if (
            cur_model_mtime == _rl_model_mtime
            and cur_data_mtime == _rl_data_mtime
            and cur_stats_mtime == _rl_vecnorm_mtime
        ):
            return

    # Mark the agent as loaded to avoid recursive loads.  If reload fails,
    # these flags and timestamps will be updated accordingly.
    _rl_loaded = True

    try:
        from stable_baselines3 import PPO
    except Exception:
        log.warning("RL: stable_baselines3 import edilemedi, RL devre dışı.")
        return

    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    model_path = models_dir / "ppo_multi.zip"
    if not model_path.exists():
        log.info("RL: ppo_multi.zip bulunamadı, RL kullanılmayacak.")
        return

    try:
        _rl_model = PPO.load(str(model_path), device="cpu")
    except Exception as e:
        log.warning(f"RL: PPO model yüklenemedi: {e}")
        _rl_model = None
        return

    # Eğitimde window genelde 60 kullanıldı; burada sabitliyoruz.
    _rl_window = 60

    # Veri yükle
    metrics_dir = root / "metrics"
    ohlc_file = metrics_dir / "ohlc_history.json"
    if not ohlc_file.exists():
        log.warning("RL: ohlc_history.json bulunamadı; RL veri olmadan çalışmayacak.")
        return

    try:
        raw = json.loads(ohlc_file.read_text(encoding="utf-8"))
        rows = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
        df = pd.DataFrame(rows)
        if df.empty:
            log.warning("RL: ohlc_history.json boş.")
            return
        # rl_env içindeki aynı temizleme fonksiyonunu kullan
        try:
            from ml.rl_env import _prepare_df_for_env  # type: ignore
            _rl_df = _prepare_df_for_env(df)
        except Exception as e:
            log.warning(f"RL: rl_env._prepare_df_for_env import/çalıştırma hatası: {e}")
            _rl_df = None
    except Exception as e:
        log.warning(f"RL: JSON/veri hazırlama hatası: {e}")
        _rl_df = None

    # On successful load update the modification timestamps and load VecNormalize stats.
    # If either the model or dataset failed to load, leave the timestamps
    # unchanged to force a reload on next call.  Additionally, attempt to load
    # VecNormalize normalisation statistics (vecnormalize_multi.pkl) so that
    # inference uses the same observation normalisation as training.  The stats
    # file is only loaded if stable_baselines3 is available and the file exists.
    if _rl_model is not None and _rl_df is not None:
        _rl_model_mtime = cur_model_mtime
        _rl_data_mtime = cur_data_mtime

        # RL VecNormalize stats (if available)
        try:
            # Determine path and modification time for vecnormalize stats
            stats_path = root / "models" / "vecnormalize_multi.pkl"
            try:
                cur_stats_mtime: float | None = os.path.getmtime(stats_path) if stats_path.exists() else None
            except Exception:
                cur_stats_mtime = None

            # Only proceed if the stats file exists and VecNormalize can be imported
            if _SB3_VecNormalize is not None and stats_path.exists() and cur_stats_mtime is not None:
                # If stats changed or not loaded yet, reload
                reload_needed = (_rl_vecnorm is None) or (cur_stats_mtime != _rl_vecnorm_mtime)
                if reload_needed:
                    try:
                        # Attempt to load VecNormalize stats without a dummy environment first.
                        # Some versions of stable-baselines3 allow loading without an env;
                        # providing a dummy environment that returns ``None`` can trigger
                        # ``NoneType`` errors.  If the simple load fails, fall back to
                        # using DummyVecEnv only if necessary.
                        vecnorm = None
                        _vec_loaded = False
                        try:
                            vecnorm = _SB3_VecNormalize.load(str(stats_path))  # type: ignore
                            _vec_loaded = True
                        except Exception:
                            _vec_loaded = False
                        if not _vec_loaded and _SB3_DummyVecEnv is not None:
                            try:
                                # Use a minimal dummy environment with valid observation and action spaces.
                                # A ``None`` environment can cause errors in stable‑baselines3.  We instead
                                # define a simple gymnasium.Env that emits a constant observation so that
                                # VecNormalize.load can succeed without accessing an actual trading environment.
                                try:
                                    import gymnasium as gym  # type: ignore
                                    import numpy as np  # type: ignore
                                    from gymnasium import spaces  # type: ignore

                                    class _DummyEnv(gym.Env):  # type: ignore
                                        """Minimal env for VecNormalize.load.

                                        VecNormalize stores running statistics keyed to the *observation shape*.
                                        Our PPO model expects observations in the same shape used during training
                                        (MultiCoinTradeEnv emits (window, 1)). If we load stats with an incompatible
                                        dummy shape, SB3 may error or silently behave incorrectly.
                                        """

                                        def __init__(self):
                                            super().__init__()
                                            self.observation_space = spaces.Box(
                                                low=-np.inf,
                                                high=np.inf,
                                                shape=(_rl_window, 1),
                                                dtype=np.float32,
                                            )
                                            # MultiCoinTradeEnv uses 3 actions: 0=HOLD, 1=BUY, 2=SELL
                                            self.action_space = spaces.Discrete(3)

                                        def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
                                            return np.zeros((_rl_window, 1), dtype=np.float32), {}

                                        def step(self, action):  # type: ignore[override]
                                            return np.zeros((_rl_window, 1), dtype=np.float32), 0.0, True, False, {}
                                except Exception:
                                    _DummyEnv = None  # type: ignore
                                if _DummyEnv is not None:
                                    dummy_env = _SB3_DummyVecEnv([lambda: _DummyEnv()])  # type: ignore
                                    vecnorm = _SB3_VecNormalize.load(str(stats_path), dummy_env)  # type: ignore
                                    _vec_loaded = True
                                else:
                                    _vec_loaded = False
                            except Exception:
                                _vec_loaded = False
                        if not _vec_loaded:
                            raise Exception("VecNormalize load failed")
                        # Disable training mode to avoid updating running statistics at inference time
                        vecnorm.training = False  # type: ignore
                        vecnorm.norm_reward = False  # type: ignore
                        # Update global vars
                        globals()["_rl_vecnorm"] = vecnorm
                        globals()["_rl_vecnorm_mtime"] = cur_stats_mtime
                    except Exception as e:
                        # If loading fails, disable vecnorm and warn once
                        globals()["_rl_vecnorm"] = None
                        globals()["_rl_vecnorm_mtime"] = None
                        log.warning(f"RL: VecNormalize yüklenemedi: {e}")
            else:
                # Stats file missing or cannot import; disable vecnorm
                globals()["_rl_vecnorm"] = None
                globals()["_rl_vecnorm_mtime"] = None
        except Exception:
            # Any unexpected error: leave vecnorm unchanged
            pass


def _rl_score_for_symbol(symbol: str, base_decision: str) -> Dict[str, Any]:
    """
    PPO RL ajanından sembol bazlı ek risk skoru üretir.
    - RL yoksa / hata varsa / base_decision neutral ise 0.5 (nötr) döner.
    - Aksiyon:
        0 → flat  → 0.5
        1 → long  (base_dec=long ise 0.8, short ise 0.2)
        2 → short (base_dec=short ise 0.8, long ise 0.2)
    """
    _load_rl_agent()
    if _rl_model is None or _rl_df is None or base_decision not in ("long", "short"):
        return {"score": 0.5, "action": None, "why": "rl_disabled_or_neutral"}

    try:
        sym_norm = _rl_norm_symbol(symbol)
        g = _rl_df[_rl_df["symbol"] == sym_norm]
        if g.empty or len(g) <= _rl_window + 2:
            return {"score": 0.5, "action": None, "why": "not_enough_bars"}

        closes = g["close"].values.astype(np.float32)
        window_slice = closes[-_rl_window:]
        base_price = window_slice[0] if window_slice[0] != 0 else float(np.mean(window_slice) or 1.0)
        x = window_slice / base_price - 1.0
        obs = x.reshape(1, _rl_window, 1).astype(np.float32)
        # Apply VecNormalize stats (if loaded) to observation.  This uses the
        # same normalization as training.  If normalization fails, the raw
        # observation is used as a fallback.
        if _rl_vecnorm is not None:
            try:
                # stable-baselines3 VecNormalize returns normalized obs with
                # the same shape as input.  For RL inference we disable
                # reward normalization and training mode in _load_rl_agent().
                obs = _rl_vecnorm.normalize_obs(obs)  # type: ignore
            except Exception as e:
                log.warning(f"RL: VecNormalize normalize_obs hata: {e}")
                # do not modify obs on error

        action, _ = _rl_model.predict(obs, deterministic=True)
        if isinstance(action, (list, tuple, np.ndarray)):
            act = int(action[0])
        else:
            act = int(action)
    except Exception as e:
        log.warning(f"RL: predict hata ({symbol}): {e}")
        return {"score": 0.5, "action": None, "why": "rl_predict_failed"}

    if act == 0:
        score = 0.5
        why = "rl_flat"
    elif act == 1:
        if base_decision == "long":
            score = 0.8
            why = "rl_long_align"
        else:
            score = 0.2
            why = "rl_long_conflict"
    elif act == 2:
        if base_decision == "short":
            score = 0.8
            why = "rl_short_align"
        else:
            score = 0.2
            why = "rl_short_conflict"
    else:
        score = 0.5
        why = "rl_unknown_action"

    return {"score": float(score), "action": act, "why": why}

# ---------------------------------------------------------------------------
# Teknik / Sentiment skor fonksiyonları
# ---------------------------------------------------------------------------

def _tech_score(ta_pack: Dict[str, Any], base_decision: str) -> Dict[str, Any]:
    """
    Eski teknik skor hesaplaması: sadece 1h ve 4h MACD histogram hizalamasına bakar.
    Bu fonksiyon korunarak geriye dönük uyumluluk sağlanır.
    
    [GÜNCELLEME] Neutral base score 0.30 → 0.45 olarak artırıldı.
    Ek olarak RSI extreme değerlerinde ve EMA farkı varsa daha yüksek skor verilir.
    """
    trend = ta_pack.get("trend", {})
    h1 = trend.get("h1_macd_hist")
    h4 = trend.get("h4_macd_hist")
    
    # Neutral veya eksik MACD durumunda daha akıllı fallback
    if h1 is None or h4 is None or base_decision not in ("long", "short"):
        # EMA ve RSI'dan ek bilgi çıkarmaya çalış
        ema = ta_pack.get("ema") or {}
        rsi = ta_pack.get("rsi")
        
        # EMA farkı varsa zayıf trend olabilir
        try:
            ema_fast = float(ema.get("fast")) if ema.get("fast") is not None else None
            ema_slow = float(ema.get("slow")) if ema.get("slow") is not None else None
            if ema_fast is not None and ema_slow is not None and ema_slow != 0:
                diff_pct = abs(ema_fast - ema_slow) / abs(ema_slow) * 100
                if diff_pct > 0.5:
                    return {"score": 0.50, "why": f"weak trend detected (EMA diff={diff_pct:.2f}%)"}
        except Exception:
            pass
        
        # RSI extreme değerlerde daha yüksek skor
        try:
            if rsi is not None:
                rsi_val = float(rsi)
                if rsi_val < 30:
                    return {"score": 0.55, "why": "RSI oversold zone"}
                if rsi_val > 70:
                    return {"score": 0.55, "why": "RSI overbought zone"}
        except Exception:
            pass
        
        # Varsayılan neutral skor (0.30 → 0.45)
        return {"score": 0.45, "why": "neutral/missing MACD (improved baseline)"}

    d = 1 if base_decision == "long" else -1
    s1, s4 = (1 if h1 > 0 else -1), (1 if h4 > 0 else -1)
    if s1 == d and s4 == d:
        return {"score": 1.0, "why": "full trend alignment"}
    elif s1 == d or s4 == d:
        return {"score": 0.6, "why": "partial alignment"}
    else:
        return {"score": 0.1, "why": "trend misalignment"}


def _enhanced_tech_score(ta_pack: Dict[str, Any], base_decision: str) -> Dict[str, Any]:
    """
    Gelişmiş teknik skor:
    - Temel MACD hizalamasını (_tech_score) baz alır.
    - RSI ile overbought/oversold bölgelerinde sinyali zayıflatır veya hafif güçlendirir.
    - ADX ile trend gücünü dikkate alır (yanal piyasada sert düşürür).
    - EMA farkı ile trendin gücünü ince ayar yapar.
    Skor 0.0–1.0 aralığında tutulur.
    """
    base = _tech_score(ta_pack, base_decision)
    score = float(base.get("score", 0.0))
    reasons = [base.get("why", "").strip()] if base.get("why") else []

    # RSI
    rsi = None
    try:
        rsi = float(ta_pack.get("rsi")) if ta_pack.get("rsi") is not None else None
    except Exception:
        rsi = None

    if rsi is not None and base_decision in ("long", "short"):
        if base_decision == "long":
            # Long sinyali için RSI eşiklerini uygula; overbought durumunda skoru zayıflat,
            # oversold durumunda ise hafif güçlendir.
            # RSI limitlerini daha duyarlı hale getir: klasik 30/70 eşikleri yerine 25/75
            if rsi > 80:
                score *= 0.6
                reasons.append("RSI>80: aşırı alım, long sert zayıflatıldı")
            elif rsi > 75:
                score *= 0.8
                reasons.append("RSI>75: aşırı alım, long zayıflatıldı")
            # Çok oversold ise hafif destek (25 yerine)
            elif rsi < 25:
                score = min(1.0, score * 1.1)
                reasons.append("RSI<25: oversold, long hafif güçlendirildi")
        elif base_decision == "short":
            # Short sinyali için RSI eşiklerini uygula; oversold (aşırı satım) durumunda skoru zayıflat,
            # overbought (aşırı alım) durumunda ise hafif güçlendir.
            if rsi < 20:
                score *= 0.6
                reasons.append("RSI<20: aşırı satım, short sert zayıflatıldı")
            elif rsi < 25:
                score *= 0.8
                reasons.append("RSI<25: aşırı satım, short zayıflatıldı")
            elif rsi > 75:
                score = min(1.0, score * 1.1)
                reasons.append("RSI>75: overbought, short hafif güçlendirildi")
                reasons.append("RSI<25: aşırı satım, short zayıflatıldı")
            # Çok overbought ise hafif destek (75 üzerine)
            elif rsi > 75:
                score = min(1.0, score * 1.1)
                reasons.append("RSI>75: overbought, short hafif güçlendirildi")

    # ADX: trend gücü
    adx = None
    try:
        adx = float(ta_pack.get("adx")) if ta_pack.get("adx") is not None else None
    except Exception:
        adx = None

    if adx is not None:
        # [GÜNCELLEME] ADX penalty'leri yumuşatıldı
        if adx < 15:
            score *= 0.70  # 0.5 → 0.70
            reasons.append("ADX<15: yanal piyasa, skor düşürüldü")
        elif adx < 20:
            score *= 0.85  # YENİ ARA KADEME
            reasons.append("15<=ADX<20: zayıf trend, skor hafif düşürüldü")
        elif adx < 25:
            score *= 0.92  # 0.8 → 0.92
            reasons.append("20<=ADX<25: orta trend, skor minimal düşürüldü")
        elif adx > 40:
            score = min(1.0, score * 1.15)  # 1.1 → 1.15
            reasons.append("ADX>40: güçlü trend, skor artırıldı")

    # EMA farkı ile ince ayar (trend gücü)
    ema = ta_pack.get("ema") or {}
    ema_fast = ema.get("fast")
    ema_slow = ema.get("slow")
    try:
        ema_fast_f = float(ema_fast) if ema_fast is not None else None
        ema_slow_f = float(ema_slow) if ema_slow is not None else None
    except Exception:
        ema_fast_f = ema_slow_f = None

    if ema_fast_f is not None and ema_slow_f is not None and base_decision in ("long", "short"):
        diff = ema_fast_f - ema_slow_f
        # [GÜNCELLEME] EMA penalty yumuşatıldı, bonus artırıldı
        if base_decision == "long":
            if diff < 0:
                score *= 0.82  # 0.7 → 0.82
                reasons.append("EMA fast<slow: long için ters eğilim, skor düşürüldü")
            elif diff > 0:
                score = min(1.0, score * 1.10)  # 1.05 → 1.10
                reasons.append("EMA fast>slow: long trend onayı, skor artırıldı")
        elif base_decision == "short":
            if diff > 0:
                score *= 0.82  # 0.7 → 0.82
                reasons.append("EMA fast>slow: short için ters eğilim, skor düşürüldü")
            elif diff < 0:
                score = min(1.0, score * 1.10)  # 1.05 → 1.10
                reasons.append("EMA fast<slow: short trend onayı, skor artırıldı")

    # Sınırlar - [GÜNCELLEME] minimum 0.20 garanti eklendi
    score = max(0.20, min(1.0, score))
    return {"score": score, "why": "; ".join([r for r in reasons if r]) or "enhanced tech score"}


def _sent_score(senti: Dict[str, Any] | None, base_decision: str) -> Dict[str, Any]:
    """
    Gelişmiş sentiment skoru:
    - Fear & Greed Index (0-100) → yönlü skor (long/short'a göre),
    - Funding rate → crowded trade tespiti,
    - Open interest değişimi (oi_change) → pozisyon akışı.

    Hepsi 0.3–0.9 bandına map edilir, sonra ağırlıklı ortalama alınır.
    Veri yoksa 0.5 (nötr) döner.
    """
    if base_decision not in ("long", "short"):
        return {"score": 0.5, "why": "neutral/no base_decision"}

    if not isinstance(senti, dict):
        senti = {}

    direction = base_decision

    # ---- FGI bileşeni (eski mantık korunuyor) ----
    fgi_raw = senti.get("fear_greed")
    try:
        v = float(fgi_raw) if fgi_raw is not None else None
    except Exception:
        v = None

    if v is not None:
        v = max(0.0, min(100.0, v))
        if direction == "long":
            raw_fgi = 1.0 - v / 100.0   # 0→1 (max), 100→0 (min)
        else:
            raw_fgi = v / 100.0         # 0→0, 100→1
        raw_fgi = max(0.0, min(1.0, raw_fgi))
        fgi_score = 0.3 + raw_fgi * 0.6
        fgi_txt = f"FGI={v:.1f}"
    else:
        fgi_score = 0.5
        fgi_txt = "FGI=None"

    # ---- Funding bileşeni ----
    funding_raw = senti.get("funding")
    try:
        fr = float(funding_raw) if funding_raw is not None else None
    except Exception:
        fr = None

    if fr is not None:
        # funding genelde küçük değer (ör: 0.0001–0.01)
        if direction == "long":
            # Negatif funding → short crowded, long lehine
            if fr <= -0.01:
                raw_f = 1.0
            elif fr <= 0.0:
                raw_f = 0.7
            elif fr <= 0.01:
                raw_f = 0.4
            else:
                raw_f = 0.2
        else:  # short
            # Pozitif funding → long crowded, short lehine
            if fr >= 0.01:
                raw_f = 1.0
            elif fr >= 0.0:
                raw_f = 0.7
            elif fr >= -0.01:
                raw_f = 0.4
            else:
                raw_f = 0.2

        raw_f = max(0.0, min(1.0, raw_f))
        funding_score = 0.3 + raw_f * 0.6
        funding_txt = f"funding={fr:.5f}"
    else:
        funding_score = 0.5
        funding_txt = "funding=None"

    # ---- OI change bileşeni ----
    oi_raw = senti.get("oi_change")
    try:
        oc = float(oi_raw) if oi_raw is not None else None
    except Exception:
        oc = None

    if oc is not None:
        # Eğer küçükse yüzde kabul et, büyükse direkt yüzde gibi kullan
        oc_pct = oc * 100.0 if abs(oc) < 1.0 else oc

        if direction == "long":
            if oc_pct >= 15:
                raw_oi = 0.8
            elif oc_pct >= 5:
                raw_oi = 0.6
            elif oc_pct >= 0:
                raw_oi = 0.5
            else:
                raw_oi = 0.3
        else:  # short
            if oc_pct <= -15:
                raw_oi = 0.8
            elif oc_pct <= -5:
                raw_oi = 0.6
            elif oc_pct <= 0:
                raw_oi = 0.5
            else:
                raw_oi = 0.3

        raw_oi = max(0.0, min(1.0, raw_oi))
        oi_score = 0.3 + raw_oi * 0.6
        oi_txt = f"oi%Δ={oc_pct:.2f}"
    else:
        oi_score = 0.5
        oi_txt = "oi_change=None"

    # ---- Sosyal bileşen ----
    social_raw = None
    social_score = 0.5
    social_txt = "social=None"
    # Öncelik verilen anahtar social_score, yoksa tweet/reddit/news üzerinden hesaplanır
    try:
        # social_score doğrudan geliyorsa kullan
        val = senti.get("social_score")
        if val is not None:
            social_score = float(val)
        else:
            # tweet/reddit/news değerlerini [-1,1] → [0,1] aralığına map et
            vals = []
            for k in ("tweet_sentiment", "reddit_sentiment", "news_sentiment"):
                v = senti.get(k)
                if v is None:
                    continue
                try:
                    f = float(v)
                    if f < -1.0:
                        f = -1.0
                    if f > 1.0:
                        f = 1.0
                    vals.append((f + 1.0) / 2.0)
                except Exception:
                    continue
            if vals:
                social_score = sum(vals) / len(vals)
    except Exception:
        social_score = 0.5
    # Yönlü çeviri: short için ters çevir (0 → 1, 1 → 0)
    if direction == "short":
        social_score = 1.0 - social_score
    # Sosyal skoru 0.3–0.9 bandına sıkıştır
    social_mapped = 0.3 + max(0.0, min(1.0, social_score)) * 0.6
    social_txt = f"social={social_score:.2f}"

    # ---- Ağırlıklı birleşim (kalite duyarlı) ----
    # Veri yoksa "nötr" gibi davranmak yerine, o bileşenin etkisini düşürürüz.
    # Bu sayede sistem 0.0 gibi agresif bir fallback'e düşmez ama "veri varmış"
    # gibi de davranmaz.
    fgi_avail = v is not None
    funding_avail = fr is not None
    oi_avail = oc is not None
    social_avail = bool(senti.get("social_available", False)) or (senti.get("social_score") is not None)

    weights = [
        (fgi_score, 0.40, fgi_avail),
        (funding_score, 0.25, funding_avail),
        (oi_score, 0.15, oi_avail),
        (social_mapped, 0.20, social_avail),
    ]
    avail_w = sum(w for _, w, ok in weights if ok)
    quality = float(avail_w)
    if avail_w > 0:
        combined = sum(s * w for s, w, ok in weights if ok) / avail_w
    else:
        combined = 0.5
    combined = max(0.3, min(0.9, float(combined)))

    why = (
        f"{fgi_txt} → {fgi_score:.2f}; "
        f"{funding_txt} → {funding_score:.2f}; "
        f"{oi_txt} → {oi_score:.2f}; "
        f"{social_txt} → {social_mapped:.2f}; "
        f"combined={combined:.2f}, dir={direction}"
    )
    return {"score": combined, "quality": quality, "why": why}


def _lev_from_master(master_0_1: float) -> int:
    """
    master_0_1 [0..1] değerini kaldıraç değerine çevirir. Eğer bir risk
    takvimi (risk_schedule.json) tanımlı ise, master değerinin hangi aralığa
    düştüğüne bakarak oradan doğrudan kaldıraç döndürür. Aksi takdirde
    lineer bir dönüşüm uygulanır (65% altında kaldıraç 0, 65→5x, 100→25x).
    """
    # Önce risk modellerini yükle
    _load_risk_models()
    # risk schedule kullanılıyorsa doğrudan aralık eşleşmesi yap
    try:
        if _risk_schedule and isinstance(_risk_schedule, dict):
            bins = _risk_schedule.get("bins")
            if isinstance(bins, list):
                m = max(0.0, min(1.0, float(master_0_1)))
                for b in bins:
                    try:
                        lo = float(b.get("min", 0))
                        hi = float(b.get("max", 1))
                        lev = int(b.get("leverage", 0))
                    except Exception:
                        continue
                    if m >= lo and m < hi:
                        # Global clamp: 5x-25x aralığı (0 ise trade yok)
                        if lev <= 0:
                            return 0
                        return max(5, min(25, lev))
    except Exception:
        pass
    # fallback lineer dönüşüm (5x-25x)
    g = max(0.0, min(1.0, float(master_0_1))) * 100.0
    # 0.65 altındaki master değerlerinde kaldıraç açılmıyor.
    if g < 65:
        return 0
    ratio = (g - 65.0) / 35.0
    lev = 5 + ratio * (25 - 5)
    return max(5, min(25, int(round(lev))))


def _base_decision_from_ema(ema_fast, ema_slow):
    if ema_fast is None or ema_slow is None:
        return "neutral"
    return "long" if float(ema_fast) > float(ema_slow) else "short"


async def decide_batch(symbol_inputs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    symbol_inputs: [{
      'symbol','tf','price',
      'ta_pack': {'ema','rsi','adx','atr','trend': {'h1_macd_hist','h4_macd_hist'}, 'base_decision'},
      'senti': {'funding','oi_change','fear_greed'}
    }, ...]
    return: {symbol: {'action','master_confidence','lev','parts','reason','base_decision'}}
    """
    global _llm_tech_only_mode
    # Her çağrıda reset; bu loop'ta ne olduğunu yansıtacak
    _llm_tech_only_mode = False

    # Karar ağırlıklarını ve risk modellerini yükle
    _load_decision_weights()

    # AI model kalite faktörünü bir kez hesapla. Bu faktör BiLSTM ve RL
    # modellerinin doğruluk ve güncelliğine göre [0.5, 1.0] arasında bir değer
    # döndürür. AI confidence skoru bu faktörle çarpılarak ölçeklenecektir.
    ai_quality_factor = _ai_quality_factor()

    # Load additional runtime knobs (hybrid weights, RL master fusion) from config.json.
    cfg_runtime = {}
    try:
        if CONFIG_FILE.exists():
            with CONFIG_FILE.open('r', encoding='utf-8') as _f:
                cfg_runtime = json.load(_f) or {}
    except Exception:
        cfg_runtime = {}

    # Hybrid weights for AI composite score (DeepSeek/ChatGPT/BiLSTM/RL)
    _hw_default = {"deepseek": 0.45, "chatgpt": 0.40, "bilstm": 0.075, "ppo_rl": 0.075}
    hybrid_weights = dict(_hw_default)
    try:
        hw = cfg_runtime.get('hybrid_weights')
        if isinstance(hw, dict):
            for k, v in hw.items():
                try:
                    hybrid_weights[str(k)] = float(v)
                except Exception:
                    pass
    except Exception:
        pass

    # Feature flags
    disable_rl_master_fusion = False
    try:
        if isinstance(cfg_runtime.get('disable_rl_master_fusion'), bool):
            disable_rl_master_fusion = bool(cfg_runtime.get('disable_rl_master_fusion'))
        else:
            feats = cfg_runtime.get('features') or {}
            if isinstance(feats, dict) and isinstance(feats.get('disable_rl_master_fusion'), bool):
                disable_rl_master_fusion = bool(feats.get('disable_rl_master_fusion'))
    except Exception:
        disable_rl_master_fusion = False

    def _compute_ai_composite(_ai_part: dict, _rl_score: float, _base_dec: str) -> float:
        """Compute AI score as a weighted blend of DeepSeek, ChatGPT, BiLSTM and RL."""
        try:
            prov = _ai_part.get('providers') or {}
            cg = prov.get('chatgpt') or {}
            ds = prov.get('deepseek') or {}
            cg_conf = cg.get('confidence')
            ds_conf = ds.get('confidence')
            if cg_conf is None:
                cg_conf = _ai_part.get('confidence')
            if ds_conf is None:
                ds_conf = _ai_part.get('confidence')
            try:
                cg_conf = float(cg_conf)
            except Exception:
                cg_conf = 0.25
            try:
                ds_conf = float(ds_conf)
            except Exception:
                ds_conf = 0.25

            bil_conf = _ai_part.get('bilstm_confidence')
            if bil_conf is None:
                bil_prob = _ai_part.get('bilstm_prob')
                if bil_prob is not None:
                    try:
                        bil_prob = float(bil_prob)
                        bil_conf = bil_prob if str(_base_dec).lower() == 'long' else (1.0 - bil_prob)
                    except Exception:
                        bil_conf = 0.5
            try:
                bil_conf = float(bil_conf) if bil_conf is not None else 0.5
            except Exception:
                bil_conf = 0.5

            # Clamp all components to [0,1]
            cg_conf = max(0.0, min(1.0, cg_conf))
            ds_conf = max(0.0, min(1.0, ds_conf))
            bil_conf = max(0.0, min(1.0, bil_conf))
            rl_conf = max(0.0, min(1.0, float(_rl_score)))

            w_ds = float(hybrid_weights.get('deepseek', _hw_default['deepseek']))
            w_cg = float(hybrid_weights.get('chatgpt', _hw_default['chatgpt']))
            w_bl = float(hybrid_weights.get('bilstm', _hw_default['bilstm']))
            w_rl = float(hybrid_weights.get('ppo_rl', _hw_default['ppo_rl']))
            s = w_ds + w_cg + w_bl + w_rl
            if s <= 0:
                return 0.25
            val = (w_ds * ds_conf + w_cg * cg_conf + w_bl * bil_conf + w_rl * rl_conf) / s
            return max(0.0, min(1.0, float(val)))
        except Exception:
            # Fallback to legacy confidence
            try:
                return float(_ai_part.get('confidence', 0.25))
            except Exception:
                return 0.25
    # 1) AI confidence'ları paralel alabilmek için OpenAI'ye uygun minimal payload hazırlayalım
    ai_items = []
    for it in symbol_inputs:
        sym = it.get("symbol")
        if not sym:
            continue
        tf = it.get("tf", "5m")

        # Compatibility: accept either 'price' or 'current_price'.
        price = it.get("price")
        if price is None:
            price = it.get("current_price")
        if price is None:
            log.warning(f"[DECIDE] Missing price/current_price for {sym}; skipping decision item")
            continue

        # Accept either the expected key (ta_pack) or legacy payloads under "analysis".
        ta = it.get("ta_pack") or it.get("analysis") or {}
        ema = ta.get("ema") or {}
        base_decision = ta.get("base_decision") or _base_decision_from_ema(ema.get("fast"), ema.get("slow"))
        senti = it.get("senti", {})

        # Multi-timeframe features: if present, use them directly; else build a
        # simple feature dictionary based on the 5m ta_pack.  The LLM prompt
        # builder will detect the structure automatically.
        mtf = it.get("mtf_features")
        if isinstance(mtf, dict) and mtf:
            features_obj = mtf
            # Override tf to 'multi' for LLM; underlying 'tf' variable may still
            # reflect last frame but is no longer meaningful.
            tf = "multi"
        else:
            # Provide a richer technical snapshot to the LLM.  We still avoid
            # sending full OHLCV arrays, but include all high-signal scalar
            # features and small recent windows if available.
            features_obj = {
                "ema": ema,
                "rsi": ta.get("rsi"),
                "adx": ta.get("adx"),
                "atr": ta.get("atr"),
                "atr_ratio": ta.get("atr_ratio"),
                "macd": ta.get("macd"),
                "macd_signal": ta.get("macd_signal"),
                "macd_hist": ta.get("macd_hist"),
                "macd_h1": ta.get("macd_h1"),
                "macd_h4": ta.get("macd_h4"),
                "bb_upper": ta.get("bb_upper"),
                "bb_middle": ta.get("bb_middle"),
                "bb_lower": ta.get("bb_lower"),
                "stoch_k": ta.get("stoch_k"),
                "stoch_d": ta.get("stoch_d"),
                "volatility": ta.get("volatility"),
                "trend": ta.get("trend"),
                "support": ta.get("support"),
                "resistance": ta.get("resistance"),
                # optional short windows produced by analyzer/main loop
                "recent_closes": ta.get("recent_closes"),
                "recent_volumes": ta.get("recent_volumes"),
                "lookback_bars": ta.get("lookback_bars"),
            }

        ai_items.append({
            "symbol": sym,
            "tf": tf,
            "price": price,
            "features": features_obj,
            "sentiment": {
                "funding": senti.get("funding"),
                "oi_change": senti.get("oi_change"),
            },
            "base_decision": base_decision,
        })

    # 2) Tek batch'te tüm confidence sonuçlarını al
    llm_disabled = False
    ai_res: Dict[str, Any] = {}
    try:
        ai_res = await get_confidences_batch(ai_items)
    except Exception as e:
        msg = str(e).lower()
        if ("429" in msg or "too many requests" in msg or "rate limit" in msg or
                "402" in msg or "payment required" in msg or "quota" in msg):
            llm_disabled = True
            _llm_tech_only_mode = True
            log.warning(
                "[AI_SAFE] LLM (get_confidences_batch) 429/402 veya rate-limit/quota hatası: "
                "LLM sinyalleri devre dışı, sadece teknik/sentiment ile trade ediliyor."
            )
            ai_res = {}
        else:
            log.warning(f"get_confidences_batch hata: {e}. AI fallback 0.25 kullanılacak.")

    # Eğer hatadan dolayı devre dışı kalmadıysa, full-fallback batch durumunu kontrol et
    if not llm_disabled:
        try:
            if not ai_res:
                # Hiç yanıt yoksa pratikte LLM yok sayılır
                llm_disabled = True
                _llm_tech_only_mode = True
                log.warning(
                    "[AI_SAFE] LLM yanıtı boş döndü. "
                    "LLM sinyalleri devre dışı, sadece teknik/sentiment ile trade ediliyor."
                )
            else:
                # Phase-2: avoid fragile string matching for outage detection.
                # Disable LLM signals only if *all* symbols report that both
                # providers are down/disabled. A partial outage (e.g. DeepSeek
                # quota but ChatGPT OK) should keep LLM contribution enabled.
                all_down = True
                for v in ai_res.values():
                    if not isinstance(v, dict):
                        continue
                    ps = v.get("provider_status")
                    if isinstance(ps, str) and ps.lower() in ("ok", "partial"):
                        all_down = False
                        break
                    providers = v.get("providers")
                    if isinstance(providers, dict):
                        for pv in providers.values():
                            if isinstance(pv, dict):
                                st = pv.get("status")
                                if isinstance(st, str) and st.lower() == "ok":
                                    all_down = False
                                    break
                        if not all_down:
                            break
                if all_down:
                    llm_disabled = True
                    _llm_tech_only_mode = True
                    log.warning(
                        "[AI_SAFE] LLM sinyalleri fallback modunda. "
                        "Bu tur için LLM katkısı devre dışı, sadece teknik/sentiment ile trade ediliyor."
                    )
        except Exception:
            # Fallback algılama başarısızsa normal akışa devam
            pass

    # 3) Model parametrelerini yükle (kalibrasyon, logistic ağırlıkları, risk takvimi)
    _load_risk_models()

    # 4) Master skor + karar üret
    out = {}
    for it in symbol_inputs:
        sym = it["symbol"]
        price = it["price"]
        ta = it["ta_pack"]
        senti = it.get("senti", {})
        ema = ta.get("ema") or {}
        base_dec = ta.get("base_decision") or _base_decision_from_ema(ema.get("fast"), ema.get("slow"))

        # AI fallback 0.25 (ai_batch_manager ile uyumlu)
        ai_part = ai_res.get(sym, {"confidence": 0.25, "rationale": "fallback"})
        # AI composite score: DeepSeek + ChatGPT + BiLSTM + RL (RL is folded into AI to avoid double-counting)
        # NOTE: rl_score is computed below; we temporarily set a placeholder and recompute after rl_score is available.
        ai_score = float(ai_part.get("confidence", 0.25))

        # AI skorunu model kalitesine göre ölçekle. Bu ölçekleme faktörü
        # _ai_quality_factor() tarafından hesaplanır ve decide_batch başında
        # ai_quality_factor değişkeni olarak set edilir. Kalite faktörü 0.5–1.0 aralığında
        # olduğundan, kötü modeller AI etkisini azaltır.
        scaled_ai_score = ai_score * ai_quality_factor


        # LLM Direction Override
        # If the consensus LLM direction differs from the technical base decision
        # and the AI confidence is sufficiently high, allow the LLM direction
        # to override. This helps avoid trades in the wrong direction when the
        # models are strongly aligned. A high threshold (0.70) avoids flip-flops.
        try:
            agg_dir = ai_part.get("direction")
            if isinstance(agg_dir, str):
                agg_dir_l = agg_dir.lower()
            else:
                agg_dir_l = None
        except Exception:
            agg_dir_l = None
        try:
            override_threshold = 0.70
            if agg_dir_l in ("long", "short") and agg_dir_l != base_dec and ai_score >= override_threshold:
                log.info(
                    f"[LLM_DIR] {sym}: overriding base_decision {base_dec} → {agg_dir_l} based on high AI confidence ({ai_score:.2f})"
                )
                base_dec = agg_dir_l
        except Exception:
            pass

        # Gelişmiş teknik skor
        tech_part = _enhanced_tech_score(ta, base_dec)
        tech_score = tech_part["score"]

        # Gelişmiş sentiment skor (FGI + funding + OI + social_score)
        sent_part = _sent_score(senti, base_dec)
        sent_score = float(sent_part.get("score", 0.5))
        sent_quality = float(sent_part.get("quality", 1.0))

        # RL skoru (PPO multi-coin)
        rl_part = _rl_score_for_symbol(sym, base_dec)
        rl_score = float(rl_part.get("score", 0.5))

        # Recompute AI composite score now that rl_score is available.
        ai_score = _compute_ai_composite(ai_part, rl_score, base_dec)
        # Recompute scaled AI score with quality factor
        scaled_ai_score = ai_score * ai_quality_factor

        # If LLM consensus action is not 'enter', cap the AI score to avoid boosting master.
        try:
            ai_act = ai_part.get("action")
            if isinstance(ai_act, str) and ai_act.lower() != "enter":
                ai_score = min(ai_score, 0.4)
                scaled_ai_score = ai_score * ai_quality_factor
        except Exception:
            pass
        # Master skoru hesapla:
        # - llm_disabled ise AI ağırlığını 0 yap, teknik/sent daha baskın.
        if llm_disabled:
            local_dw = {"ai": 0.0, "tech": 0.8, "sent": 0.2}
        else:
            local_dw = _DECISION_WEIGHTS

        # If sentiment inputs are missing/unavailable, reduce their impact in a
        # deterministic way rather than pretending neutral data exists.
        eff_sent_w = float(local_dw.get("sent", 0.0)) * max(0.0, min(1.0, sent_quality))

        # 1) Lineer taban skor (AI + Teknik + Sentiment)
        # AI skorunu kalite faktörü ile ölçeklenmiş hali kullanılıyor (scaled_ai_score)
        base_linear = (
            float(local_dw.get("ai", 0.0)) * scaled_ai_score +
            float(local_dw.get("tech", 0.0)) * tech_score +
            eff_sent_w * sent_score
        )
        base_linear = max(0.0, min(1.0, float(base_linear)))

        # 2) Logistic weights varsa ve LLM aktifse, onlarla refine et; yoksa lineer kullan
        if _logistic_weights and not llm_disabled:
            try:
                w0 = float(_logistic_weights.get("w0", 0.0))
                w_ai = float(_logistic_weights.get("w_ai", 0.0))
                w_tech = float(_logistic_weights.get("w_tech", 0.0))
                w_sent = float(_logistic_weights.get("w_sent", 0.0))
                eff_w_sent = w_sent * max(0.0, min(1.0, sent_quality))
                # Logistik hesaplamada input'ları 0.5 etrafında merkezle ve [-1,1] aralığına genişlet.
                # Bu yaklaşım, master skorların dar bir banda "sıkışmasını" azaltır.
                def _center01(x: float) -> float:
                    x = max(0.0, min(1.0, float(x)))
                    return (x - 0.5) * 2.0

                ai_c = _center01(scaled_ai_score)
                tech_c = _center01(tech_score)
                sent_c = _center01(sent_score)

                raw_val = w0 + w_ai * ai_c + w_tech * tech_c + eff_w_sent * sent_c
                # Sigmoid hesaplamasında overflow'u önlemek için güvenli sigmoid kullan
                base_master = _safe_sigmoid(raw_val)
            except Exception:
                base_master = base_linear
        else:
            base_master = base_linear

        # 3) RL master fusion (optional)
        # RL is already folded into AI composite via hybrid_weights. To avoid double-counting,
        # you can disable this block via config.json: disable_rl_master_fusion=true.
        if disable_rl_master_fusion:
            master_raw = base_master
        else:
            # Adaptive RL blend (legacy) - small influence by default
            try:
                if rl_score >= 0.8 or rl_score <= 0.2:
                    rl_alpha = 0.25
                else:
                    rl_alpha = 0.10
                master_raw = (1.0 - rl_alpha) * base_master + rl_alpha * rl_score

                # Continuous RL gating (keeps variation)
                rl = float(rl_score)
                if rl >= 0.8:
                    boost = 0.04 + 0.04 * min(1.0, (rl - 0.8) / 0.2)
                    master_raw = min(1.0, master_raw + boost)
                elif rl <= 0.2:
                    pen = 0.04 + 0.04 * min(1.0, (0.2 - rl) / 0.2)
                    master_raw = max(0.0, master_raw - pen)
                else:
                    master_raw = min(1.0, max(0.0, master_raw + 0.03 * (rl - 0.5)))
            except Exception:
                master_raw = base_master

        # ------------------------------------------------------------------
        # ENHANCEMENT ENTEGRASYONU
        # ------------------------------------------------------------------
        # AI Agreement Bonus, MTF Confirmation vb. ek modüller
        enhancement_details = {}
        
        if _ENHANCEMENTS_AVAILABLE and not llm_disabled:
            try:
                # 1) AI Agreement Bonus
                # ChatGPT ve DeepSeek sonuçlarını al
                chatgpt_res = ai_part.get("providers", {}).get("chatgpt", {})
                deepseek_res = ai_part.get("providers", {}).get("deepseek", {})
                bilstm_prob = ai_part.get("bilstm_prob", 0.5)
                
                # Eğer providers yoksa ai_part'tan doğrudan al
                if not chatgpt_res:
                    chatgpt_res = {
                        "direction": ai_part.get("direction", ""),
                        "action": ai_part.get("action", ""),
                        "confidence": ai_part.get("confidence", 0.5)
                    }
                if not deepseek_res:
                    deepseek_res = chatgpt_res  # Fallback
                
                agreement = calculate_ai_agreement_bonus(
                    chatgpt_res, deepseek_res, bilstm_prob
                )
                
                if agreement["bonus"] > 0:
                    master_raw = min(1.0, master_raw + agreement["bonus"])
                    enhancement_details["ai_agreement"] = agreement
                    log.debug(f"{sym}: AI agreement bonus +{agreement['bonus']:.3f} ({agreement['agreement_level']})")
                
                # 2) MTF Confirmation (eğer ta_pack'te MTF verisi varsa)
                if ta.get("mtf") or ta.get("1h") or ta.get("4h"):
                    mtf_result = check_mtf_alignment(ta, base_dec)
                    enhancement_details["mtf"] = mtf_result
                    
                    if mtf_result["approved"] and mtf_result["score"] > 0.6:
                        # MTF onaylı ve güçlü: bonus
                        mtf_bonus = (mtf_result["score"] - 0.5) * 0.08
                        master_raw = min(1.0, master_raw + mtf_bonus)
                        log.debug(f"{sym}: MTF bonus +{mtf_bonus:.3f}")
                    elif not mtf_result["approved"] and mtf_result["score"] < 0.4:
                        # MTF reddetti: penalty
                        master_raw *= 0.95
                        log.debug(f"{sym}: MTF rejection penalty ×0.95")
                
            except Exception as enh_err:
                log.warning(f"{sym}: Enhancement hesaplama hatası: {enh_err}")

        # ------------------------------------------------------------------
        # Meta-Politika: Piyasa Rejimi ve Zaman Dilimi Çoklama
        #
        # Rejim (BULL/BEAR/SIDEWAYS) ve zaman dilimi (tf) bilgilerini
        # master confidence üzerinde ölçekleme yapmak için kullanıyoruz.
        # BULL rejimi long sinyalleri güçlendirirken short sinyalleri
        # zayıflatır; BEAR rejimi bunun tersini yapar; SIDEWAYS rejimi
        # tüm sinyallerde daha temkinli olunmasını sağlar.  Ayrıca
        # scalp (5m/15m) işlemlerde master biraz azaltılır, swing/position
        # işlemlerde ise bir miktar artırılır.  Bu ayarlamalar,
        # master_raw üzerinde yapıldığından, kalibrasyon ve risk
        # katmanları daha sonra uygulanmaya devam edecektir.

        def _regime_multiplier(dec: str, regime: str | None) -> float:
            if not regime or not isinstance(regime, str):
                return 1.0
            regime_u = regime.upper()
            dec_l = dec.lower() if isinstance(dec, str) else ""
            try:
                if regime_u == "BULL":
                    return 1.05 if dec_l == "long" else 0.95
                if regime_u == "BEAR":
                    return 1.05 if dec_l == "short" else 0.95
                if regime_u == "SIDEWAYS":
                    return 0.90
            except Exception:
                return 1.0
            return 1.0

        def _horizon_multiplier(timeframe: str) -> float:
            try:
                tf = str(timeframe).lower()
            except Exception:
                return 1.0
            if tf in ("5m", "15m"):
                return 0.90
            if tf in ("30m", "1h", "2h", "4h"):
                return 1.00
            if tf in ("6h", "8h", "12h", "1d", "1day"):
                return 1.10
            return 1.0

        def _is_horizon_allowed(timeframe: str, regime: str | None) -> bool:
            """
            Çoklu zaman aralığı modülü: Belirli rejimlerde yalnızca bazı
            zaman dilimlerinin aktif olmasına izin verir.  Bu işlev,
            kullanıcının belirlediği basit kurallara göre bir zaman diliminin
            kullanılabilir olup olmadığını döndürür.  Eğer zaman dilimi veya
            rejim tanımsızsa True döner.

            Kurallar:
              - BULL rejimi: Trend takip eden aralıklar (30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d)
              - SIDEWAYS rejimi: Sadece scalp/mean‑reversion (5m, 15m, 30m)
              - BEAR rejimi: Kısa horizonlar (5m, 15m, 30m, 1h)
              - Diğer veya tanımsız rejim: tüm zaman dilimleri

            Bu ayarları değiştirmek için fonksiyonun içini düzenleyebilirsiniz.
            """
            try:
                t = str(timeframe).lower() if timeframe else ""
            except Exception:
                return True
            try:
                r = str(regime).upper() if regime else None
            except Exception:
                r = None
            # Bull market: longer horizons for trend trades
            if r == "BULL":
                return t in ("30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1day")
            # Sideways: scalp only
            if r == "SIDEWAYS":
                return t in ("5m", "15m", "30m")
            # Bear: scalp + short intraday
            if r == "BEAR":
                return t in ("5m", "15m", "30m", "1h")
            return True

        regime_val: str | None = None
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

        tf_for_horizon = it.get("tf", "")
        try:
            r_mult = _regime_multiplier(base_dec, regime_val)
            h_mult = _horizon_multiplier(tf_for_horizon)
            adj_master = master_raw * float(r_mult) * float(h_mult)
            if adj_master < 0.0:
                adj_master = 0.0
            if adj_master > 1.0:
                adj_master = 1.0
            master_raw = adj_master
        except Exception:
            pass

        # ------------------------------------------------------------------
        # RL Adjustment: apply per-symbol/regime factor to the raw master.
        # If an RL adjustment factor is defined for this symbol and regime,
        # multiply master_raw by it.  For example, BTC might get boosted in a
        # bull regime while altcoins may be dampened.  The factor is loaded
        # from data/rl_adjustments.json via _get_rl_factor.
        try:
            rl_factor = _get_rl_factor(sym, regime_val)
            if isinstance(rl_factor, (float, int)) and rl_factor != 1.0:
                master_raw = float(master_raw) * float(rl_factor)
                # Clamp
                if master_raw < 0.0:
                    master_raw = 0.0
                if master_raw > 1.0:
                    master_raw = 1.0
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Makro olay filtresi: yaklaşan veya devam eden ekonomik olaylar
        # varsa risk çarpanını uygula. multiplier <1.0 ise master_raw
        # azaltılır.  Dosya yoksa multiplier 1.0 döner.
        try:
            macro_mult = _get_macro_risk_multiplier()
            master_raw = float(master_raw) * float(macro_mult)
            # Clamp master_raw to [0,1]
            if master_raw < 0.0:
                master_raw = 0.0
            if master_raw > 1.0:
                master_raw = 1.0
        except Exception:
            pass

        # ------------------------------------------------------------------
        # On‑chain sentiment: zincir verilerinden elde edilen sentiment
        # değerini master confidence ile harmanla.  On‑chain skoru 0.5
        # nötrdür.  Ağırlıklı ortalama ile hafif bir etki uygularız (10%).
        try:
            onchain_val = _get_onchain_sentiment(sym)
            # Basit harman: %90 mevcut master_raw, %10 onchain skoru
            master_raw = 0.9 * float(master_raw) + 0.1 * float(onchain_val)
            if master_raw < 0.0:
                master_raw = 0.0
            if master_raw > 1.0:
                master_raw = 1.0
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Anomali kontrolü: _analyze_one fonksiyonundan gelen 'anomaly'
        # bayrağı, mevcut fiyatın ATR'ye göre aşırı sıçradığını gösterir.
        # Eğer bu bayrak True ise master_raw'u 0.0'a çekerek sinyali
        # tamamen pas geçeriz.
        try:
            anom_flag = bool(it.get("anomaly"))
            if anom_flag:
                master_raw = 0.0
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Order book imbalance adjustment: if the input dict contains an
        # "imbalance" key (supplied by main_bot_async), use it to tweak
        # master_raw.  Positive imbalance favours long positions and
        # suppresses shorts; negative imbalance does the opposite.  If the
        # helper is not available or imbalance is None, this call returns
        # master_raw unchanged.
        try:
            imb_val = it.get("imbalance")  # may be None
            master_raw = adjust_confidence_with_imbalance(master_raw, imb_val, base_dec)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Meta‑labeling adjustment: compute the probability that this
        # trade setup is profitable using the meta model.  If the meta
        # probability is below a threshold (e.g. 0.55), suppress the
        # signal entirely.  Otherwise, scale the master by the meta
        # probability.  If no meta model is available, this call returns
        # a default of 1.0.
        try:
            meta_prob = compute_meta_probability(master_raw, scaled_ai_score, tech_score, sent_score)
            # Use a conservative threshold to filter out low-probability trades
            if meta_prob < 0.55:
                master_raw = 0.0
            else:
                master_raw = master_raw * float(meta_prob)
            # Clamp
            if master_raw < 0.0:
                master_raw = 0.0
            if master_raw > 1.0:
                master_raw = 1.0
        except Exception:
            pass

        # Horizon filtresi: Rejim ve zaman dilimine göre trade açmayı engelle
        # Belirli rejimlerde bazı zaman aralıkları tercih edilmiyorsa master
        # skoru sıfırlayarak sinyali atla.
        try:
            if not _is_horizon_allowed(tf_for_horizon, regime_val):
                master_raw = 0.0
        except Exception:
            pass

        # If upstream logic suppresses the signal (e.g., meta_prob < threshold
        # or horizon filter), keep it as a hard zero. Do NOT feed hard-zeros
        # through calibration sigmoids, otherwise they turn into small non-zero
        # clusters (~0.05–0.10) that look like "weak" signals in logs.
        if float(master_raw) <= 0.0:
            return {
                # `symbol` is not defined in this scope; the local identifier is `sym`.
                # This previously raised: NameError: name 'symbol' is not defined
                "symbol": sym,
                "action": "SKIP",
                "master": 0.0,
                "lev": 0,
                # base_dec already holds the final directional decision ("long"/"short")
                "base": base_dec,
                "mode": "suppressed",
                "ai_score": float(ai_score),
                "tech_score": float(tech_score),
                "sent_score": float(sent_score),
                "rl_score": float(rl_score),
                "provider_status": provider_flags,
                "reason": "master_raw_suppressed",
            }

        # Use the raw master confidence as starting point for final score
        master = master_raw

        # ------------------------------------------------------------------
        # Skor kalibrasyonu
        #
        # Kalibrasyon modeli (calibration.json) master_raw'ı olasılık
        # tahminine dönüştürmek için kullanılır.  Ayrıca kalibrasyon
        # parametreleri 'segments' anahtarı içeriyorsa, mevcut volatilite
        # kategorisi için segment spesifik a,b değerleri uygulanır.  Bu
        # sayede farklı rejimlerde güven skorları daha doğru dağıtılabilir.
        #
        # Önce vol_category hesaplanır: ATR/fiyat oranına göre low/med/high.
        vol_category = None
        try:
            atr_tmp = ta.get("atr")
            price_tmp = price
            atr_tmp = float(atr_tmp) if atr_tmp is not None else None
            # Only compute if both atr and price are positive
            if atr_tmp is not None and price_tmp is not None:
                _v_tmp = classify_volatility(atr_tmp, price_tmp)
                if _v_tmp and isinstance(_v_tmp, dict):
                    vol_category = _v_tmp.get("category")
        except Exception:
            vol_category = None

        if _calibration_params and isinstance(_calibration_params, dict):
            try:
                ctype = _calibration_params.get("type")
                if ctype == "logistic":
                    # Global parameters
                    a = float(_calibration_params.get("a", 1.0))
                    b = float(_calibration_params.get("b", 0.0))
                    # If segments defined, override a,b for current volatility category
                    seg_params = None
                    try:
                        segs = _calibration_params.get("segments")
                        if isinstance(segs, dict) and vol_category:
                            seg_key = f"VOL_{str(vol_category).upper()}"
                            seg_params = segs.get(seg_key)
                    except Exception:
                        seg_params = None
                    if seg_params and isinstance(seg_params, dict):
                        try:
                            a_seg = float(seg_params.get("a", a))
                            b_seg = float(seg_params.get("b", b))
                            # Clamp segment params to safe ranges
                            a = max(0.25, min(4.0, a_seg))
                            b = max(-2.0, min(2.0, b_seg))
                        except Exception:
                            pass
                    # Platt scaling: logistic transform via logit
                    a = float(max(0.25, min(4.0, a)))
                    b = float(max(-2.0, min(2.0, b)))
                    p = float(max(1e-6, min(1.0 - 1e-6, master)))
                    logit_val = math.log(p / (1.0 - p))
                    master = float(_safe_sigmoid(a * logit_val + b))
                elif ctype == "isotonic":
                    try:
                        # Default global isotonic params
                        thresholds = _calibration_params.get("thresholds")
                        preds = _calibration_params.get("preds")
                        # If segments exist, override thresholds/preds based on vol_category
                        seg_thresholds = None
                        seg_preds = None
                        try:
                            segs = _calibration_params.get("segments")
                            if isinstance(segs, dict) and vol_category:
                                seg_key = f"VOL_{str(vol_category).upper()}"
                                seg_params = segs.get(seg_key)
                                if seg_params and isinstance(seg_params, dict):
                                    seg_thresholds = seg_params.get("thresholds")
                                    seg_preds = seg_params.get("preds")
                        except Exception:
                            pass
                        # Use segment-specific thresholds if available
                        if seg_thresholds and seg_preds:
                            thresholds = seg_thresholds
                            preds = seg_preds
                        # Ensure lists are present
                        if not thresholds or not preds or len(thresholds) != len(preds):
                            # Identity mapping if incomplete
                            raise ValueError("Invalid isotonic calibration params")
                        # Convert to floats
                        try:
                            th_arr = [float(x) for x in thresholds]
                            pr_arr = [float(x) for x in preds]
                        except Exception:
                            th_arr = [0.0, 1.0]
                            pr_arr = [0.0, 1.0]
                        # Apply piecewise linear interpolation
                        try:
                            # Clip input into [0,1]
                            p_val = float(max(0.0, min(1.0, master)))
                            master = float(_np.interp(p_val, th_arr, pr_arr))
                        except Exception:
                            pass
                    except Exception:
                        # If isotonic calibration fails, leave master unchanged
                        pass
                # Additional methods (e.g. beta / temperature) can be added here
            except Exception:
                pass

        # Bireysel alanları sınırla (0–1 aralığı)
        master = max(0.0, min(1.0, float(master)))

        # Master score dağılımını aç ("sıkışma" etkisini azalt).
        # Lineer bir "spread" uygulayarak 0.5 etrafındaki değerleri
        # genişletir; sıralamayı bozmaz, sadece aralığı büyütür.
        try:
            spread = float(os.getenv("MASTER_SCORE_SPREAD", "1.8"))
            if spread > 1.0:
                master = 0.5 + (master - 0.5) * spread
                master = max(0.0, min(1.0, float(master)))
        except Exception:
            pass

        # Yüksek confidence değerlerini 0.85 üzeri bölgede sıkıştır
        master = _compress_high_confidence(master)

        # ------------------------------------------------------------------
        # RL Penalty removed: When RL score is neutral (≈0.5) or the RL
        # component is effectively disabled, we no longer apply a penalty
        # to the master confidence.  Earlier versions multiplied the master
        # by 0.85 when rl_score≈0.5.  This logic has been removed to avoid
        # unduly punishing trades when RL input is unavailable or neutral.

        # ------------------------------------------------------------------
        # Risk penalty: use logistic risk weights (if available) to estimate
        # the probability of adverse outcomes.  Compute a risk score from
        # the same logistic weights used to refine the base master score.
        # Multiply the current master by (1 - risk_score) so that higher
        # risk reduces confidence.  If weights or scores are missing, this
        # section has no effect.
        try:
            if _logistic_weights and not llm_disabled:
                w0 = float(_logistic_weights.get("w0", 0.0))
                w_ai = float(_logistic_weights.get("w_ai", 0.0))
                w_tech = float(_logistic_weights.get("w_tech", 0.0))
                w_sent = float(_logistic_weights.get("w_sent", 0.0))
                # Calculate risk z-score using scaled AI score, technical and sentiment scores
                z_risk = w0 + w_ai * float(scaled_ai_score) + w_tech * float(tech_score) + w_sent * float(sent_score)
                # Use safe sigmoid to convert to probability
                risk_prob = _safe_sigmoid(z_risk)
                #
                # The raw risk probability returned by the logistic model can be
                # overly pessimistic, causing the confidence to be almost
                # completely zeroed out even for moderately risky setups.  To
                # temper this effect, apply a scaling factor so that the risk
                # penalty only reduces confidence by at most half of the
                # estimated risk.  For example, a risk_prob of 0.8 yields a
                # penalty of 1 − (0.8 × 0.5) = 0.6 rather than 0.2.  This
                # adjustment preserves the relative ordering of risks while
                # keeping master_confidence above meaningful levels.
                # Reduce risk penalty influence further to allow master scores to exceed threshold.
                # Using 0.2 here means the risk penalty can reduce confidence by at most 20%.
                # Temper the risk penalty further to allow a wider distribution of
                # master scores.  By reducing the scaling factor from 0.2 to
                # 0.1, high risk still reduces confidence but less
                # aggressively, preventing scores from clustering tightly
                # around the mid‑range.
                risk_prob_adj = float(risk_prob) * 0.1
                penalty = 1.0 - risk_prob_adj
                if penalty < 0.0:
                    penalty = 0.0
                if penalty > 1.0:
                    penalty = 1.0
                master = master * penalty
                # clamp after applying risk penalty
                if master < 0.0:
                    master = 0.0
                if master > 1.0:
                    master = 1.0
        except Exception:
            pass

        # --- SINYAL LOG KAYDI ---
        # Her sembol için master_raw (kalibrasyonsuz) ve bileşen skorlarını kaydet.
        try:
            # Rejim bilgisini data/market_regime.json dosyasından al (varsa)
            regime_val = None
            from pathlib import Path
            import json as _json
            regime_file = Path(__file__).resolve().parent / "data" / "market_regime.json"
            if regime_file.exists():
                try:
                    _reg_txt = regime_file.read_text(encoding="utf-8")
                    _reg_data = _json.loads(_reg_txt) if _reg_txt else {}
                    regime_val = _reg_data.get("REGIME")
                except Exception:
                    regime_val = None
            # Sinyal loglama fonksiyonu çağrısı
            # scaled_ai_score kullanmak: AI skoruna kalite faktörünün uygulanmış hali
            log_signal(
                symbol=sym,
                tf=tf,
                master_conf_raw=master_raw,
                ai_score=scaled_ai_score,
                tech_score=tech_score,
                sent_score=sent_score,
                rl_score=rl_score,
                base_decision=base_dec,
                price_entry_planned=price,
                regime=regime_val,
            )
        except Exception:
            # loglama hatası olursa sessizce yut
            pass

        # Kaldıraç ve aksiyon, master confidence'e göre belirlenecek. Ancak önce
        # volatilite risk katsayısı ile master'ı güncelleyelim. ATR ve fiyat
        # bilgisi mevcutsa, risk_manager.classify_volatility fonksiyonu ile
        # volatilite kategorisini belirleyip risk_factor ile master'ı çarpıyoruz.
        # Bu, yüksek volatilitede güvenin düşmesini sağlar.

        # ATR değeri (önceden alınmamışsa yükle)
        atr_value = ta.get("atr")
        try:
            atr_value = float(atr_value) if atr_value is not None else None
        except Exception:
            atr_value = None

        try:
            if atr_value is not None and price is not None:
                v_info = classify_volatility(atr_value, price)
                if v_info and isinstance(v_info, dict):
                    rf = v_info.get("risk_factor")
                    if isinstance(rf, (float, int)):
                        master = master * float(rf)
                        # Clamp after scaling
                        master = max(0.0, min(1.0, float(master)))
        except Exception:
            pass

        # Global sentiment adjustment: gently nudge master up or down
        try:
            gsent = _load_global_sentiment()
            master = _apply_global_sentiment_adjustment(master, gsent, strength=0.10)
        except Exception:
            # if anything fails, ignore and keep master as is
            pass

        # Varsayılan kaldıraç, seans çarpanı uygulamadan önce hesaplanır
        try:
            lev = _lev_from_master(master)
            # Safety: if LLM signals are disabled, do NOT open new positions.
            if llm_disabled and action == "enter":
                action = "skip"
            if llm_disabled:
                lev = 0

        except Exception:
            lev = 0
        # Seans bazlı risk katsayısı uygula
        try:
            # Eğer işlem yapılabilir değilse, aksiyonu skip olarak ayarla
            if not _is_trading_enabled():
                # Seans işlem yapmaya izin vermiyorsa sadece skip
                action = "skip"
                # Kaldıraç değerini 0 olarak ayarla
                lev = 0
            else:
                # Seans risk çarpanı 0.0–2.0 aralığında tutulur. Master'a uygula.
                sess_mult = _get_session_multiplier()
                try:
                    m = float(master) * float(sess_mult)
                except Exception:
                    m = master
                # Clamp after session multiplier
                master = max(0.0, min(1.0, m))
                # Seans çarpanından sonra kaldıraç değerini yeniden hesapla
                lev = _lev_from_master(master)
                # Safety: if LLM signals are disabled, do NOT open new positions.
                if llm_disabled and action == "enter":
                    action = "skip"
                if llm_disabled:
                    lev = 0

                # Sembol bazlı threshold'u yükle
                threshold = _get_symbol_threshold(sym, default=0.65)
                action = "enter" if master >= threshold else "skip"
        except Exception:
            # Bir hata durumunda default davranış
            action = "enter" if master >= 0.65 else "skip"
            try:
                # Default threshold üzerinden kaldıraç hesapla
                lev = _lev_from_master(master)
                # Safety: if LLM signals are disabled, do NOT open new positions.
                if llm_disabled and action == "enter":
                    action = "skip"
                if llm_disabled:
                    lev = 0

            except Exception:
                lev = 0

        # GÜVENLİK: Eğer base_decision 'neutral' ise risk manager hata verir.
        if base_dec not in ("long", "short"):
            action = "skip"

        # FGI
        fgi_value = None
        try:
            fgi_value = float(senti.get("fear_greed")) if senti.get("fear_greed") is not None else None
        except Exception:
            fgi_value = None

        # ADX ve RSI
        adx_value = None
        rsi_value = None
        try:
            adx_value = float(ta.get("adx")) if ta.get("adx") is not None else None
        except Exception:
            adx_value = None
        try:
            rsi_value = float(ta.get("rsi")) if ta.get("rsi") is not None else None
        except Exception:
            rsi_value = None

        # EMA değerleri
        ema_fast_val = None
        ema_slow_val = None
        try:
            ema_fast_val = float(ema.get("fast")) if ema.get("fast") is not None else None
        except Exception:
            ema_fast_val = None
        try:
            ema_slow_val = float(ema.get("slow")) if ema.get("slow") is not None else None
        except Exception:
            ema_slow_val = None

        reason_str = "ai-batch"
        if llm_disabled:
            reason_str = "tech-only-fallback (LLM disabled/full-fallback)"

        used_dw = local_dw
        mode_tag = "llm_ok"
        if llm_disabled:
            mode_tag = "llm_disabled_tech_only"

        # log için güvenli upper
        action_u = str(action).upper()
        action_str = action_u

        # Derive provider flags for logging and calibration
        provider_flags: Dict[str, Any] = {}
        try:
            # Was the LLM completely disabled this batch?
            provider_flags["llm_disabled"] = bool(llm_disabled)
            # Provider status aggregated
            ps = ai_part.get("provider_status") if isinstance(ai_part, dict) else None
            if isinstance(ps, str):
                provider_flags["ai_provider_status"] = ps
            # Per‑provider statuses if available
            provs = ai_part.get("providers") if isinstance(ai_part, dict) else None
            if isinstance(provs, dict):
                for _pname, _pinfo in provs.items():
                    if isinstance(_pinfo, dict):
                        st = _pinfo.get("status")
                        if isinstance(st, str):
                            provider_flags[f"ai_{_pname}_status"] = st
            # Sentiment availability
            provider_flags["sentiment_available"] = bool(sent_score is not None)
            # On‑chain availability
            provider_flags["onchain_available"] = bool(onchain_val is not None)
        except Exception:
            # On any error, still produce a dict
            provider_flags = {"llm_disabled": bool(llm_disabled)}

        out[sym] = {
            "symbol": sym,
            "action": action_u,
            # Unified final decision score
            "score": master,
            # Backward compatibility alias
            "master_confidence": master,
            # Raw confidence before calibration and risk penalties
            "raw_score": master_raw,
            "lev": lev,
            "parts": {
                "ai": {"score": ai_score, **ai_part},
                "tech": tech_part,
                "sent": sent_part,
                "rl": rl_part,
            },
            "reason": reason_str,
            "base_decision": base_dec,
            "price": price,
            "atr": atr_value,
            # Ek alanlar logging için: fgi, adx, rsi, ema
            "fgi": fgi_value,
            "adx": adx_value,
            "rsi": rsi_value,
            "ema_fast": ema_fast_val,
            "ema_slow": ema_slow_val,
            # AI, teknik, sentiment ve RL skorları direkt ekle
            "ai_score": ai_score,
            "tech_score": tech_score,
            "sent_score": sent_score,
            "rl_score": rl_score,
            "rl_action": rl_part.get("action"),
            "rl_why": rl_part.get("why"),
            # Additional context for calibration and debugging
            "vol_category": vol_category,
            "provider_flags": provider_flags,
            "provider_status": provider_flags.get("ai_provider_status", "ok"),
        }
        log.info(
            f"{sym}: {action_str} | master={round(master*100,2)}% | "
            f"lev={lev}x | base={base_dec} | weights(ai/tech/sent)="
            f"({used_dw['ai']:.2f}/{used_dw['tech']:.2f}/{used_dw['sent']:.2f})"
            f" | rl_score={rl_score:.2f} | mode={mode_tag}"
        )

    # --- Persist full decision snapshot for dashboards/debugging ---
    try:
        import os, json
        from datetime import datetime
        from pathlib import Path
        os.makedirs('metrics', exist_ok=True)
        snapshot = {
            'generated_at_utc': datetime.utcnow().isoformat() + 'Z',
            'n_symbols_in': len(symbol_inputs),
            'n_symbols_out': len(out),
            'decisions': list(out.values()),
        }
        with open(os.path.join('metrics', 'last_decisions.json'), 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"[DECIDE] could not write metrics/last_decisions.json: {e}")

    # Brief summary to make it obvious in logs how many symbols were scored
    try:
        n_enter = sum(1 for d in out.values() if d.get('action') == 'ENTER')
        n_skip  = sum(1 for d in out.values() if d.get('action') == 'SKIP')
        n_hold  = sum(1 for d in out.values() if d.get('action') == 'HOLD')
        log.info(f"[DECIDE] Scored {len(out)}/{len(symbol_inputs)} symbols | ENTER={n_enter} SKIP={n_skip} HOLD={n_hold}")
    except Exception:
        pass
    return out
