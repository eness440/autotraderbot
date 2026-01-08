# -*- coding: utf-8 -*-
"""
ai_batch_manager.py
- ChatGPT + DeepSeek entegrasyonu (%50-%50 ağırlık)
- Token tasarrufu için 20’lik batch analiz sistemi
- Async ve 429-limit güvenli (rate-limit/quota durumunda yukarıya hata fırlatır)
- Tahmin loglama (metrics/ai_predictions.json otomatik)
- Canlı sinyal kaydı (metrics/signals_last.json)
"""
import asyncio
import os
import json
import re
import pathlib
import datetime
import time
import random
from typing import Dict, Any, List

# Import BiLSTM inference utility.  This module provides a lightweight
# ``predict_prob`` function that returns the probability of an upward move
# for a given symbol.  The model and data are cached and hot‑reloaded on
# disk changes.  See ``ml/bilstm_inference.py`` for details.
try:
    from ml.bilstm_inference import predict_prob as _bilstm_predict_prob  # type: ignore
except Exception:
    # If import fails, define a stub that returns neutral probability
    def _bilstm_predict_prob(symbol: str, window: int = 60) -> float:  # type: ignore
        return 0.5

# Load hybrid weights from config.json if present.  These weights determine
# the relative influence of each AI component (ChatGPT, DeepSeek, BiLSTM)
# when computing the combined confidence.  If the config is missing or
# malformed, sensible defaults are used.
try:
    _cfg_path = pathlib.Path("config.json")
    _cfg_data: Dict[str, Any] = {}
    if _cfg_path.exists():
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _cfg_data = json.load(_f)
    _hw = {
        "chatgpt": 0.4,
        "deepseek": 0.4,
        "bilstm": 0.1,
        "ppo_rl": 0.1,
    }
    if isinstance(_cfg_data.get("hybrid_weights"), dict):
        for k, v in _cfg_data["hybrid_weights"].items():
            try:
                _hw[k] = float(v)
            except Exception:
                pass
    HYBRID_WEIGHTS = _hw
except Exception:
    HYBRID_WEIGHTS = {"chatgpt": 0.4, "deepseek": 0.4, "bilstm": 0.1, "ppo_rl": 0.1}

from openai import AsyncOpenAI
from logger import get_logger
from settings import (
    OPENAI_API_KEY, OPENAI_DECISION_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_DECISION_MODEL
)

# ────────────────────────────────────────────────────────────────
# Logger
# ────────────────────────────────────────────────────────────────
log = get_logger("ai_batch_manager")

# ────────────────────────────────────────────────────────────────
# Kurumsal sistem prompt (ChatGPT + DeepSeek ortak)
# ────────────────────────────────────────────────────────────────
# Varsayılan sistem promptu. Aşağıdaki satırlar, ChatGPT ve DeepSeek modellerinin
# karmaşık durumlarda bile tutarlı ve risk odaklı kararlar vermesini sağlar.
TRADING_SYSTEM_PROMPT = """
You are an institutional-grade quantitative crypto futures strategist.

Goal:
- Evaluate long/short opportunities on OKX USDT perpetual swaps.
- Focus on *risk-adjusted* returns and capital preservation, *not* trade frequency or unrealistic profit targets.
- Be conservative by default; only high-quality setups should be marked as "enter". It is better to skip a mediocre trade than to chase unproven setups.

Inputs per symbol (already preprocessed for you):
- Symbol and current price
- Multi‑timeframe technicals (5m, 15m, 1h, 4h): trend, momentum, volatility, key EMAs and indicators (e.g. MACD, Bollinger Bands, Ichimoku, Fibonacci levels)
- Market structure information and basic on-chain/derivatives sentiment where available
- Internal hybrid scores (technical, sentiment, AI, risk)

Output FORMAT (STRICT JSON; **no extra text**, no comments, no markdown):
[
  {
    "symbol": "BTC/USDT",
    "direction": "long" | "short",
    "action": "enter" | "hold" | "exit" | "skip",
    "master_confidence": 0.0,
    "tech_confidence": 0.0,
    "sentiment_confidence": 0.0,
    "confidence": 0.0,
    "timeframe_cluster": "scalp" | "intraday" | "swing",
    "reason": "one concise English sentence explaining the core rationale (mention which key indicators or factors contributed to your decision, e.g. RSI oversold, funding positive, social sentiment bullish)"
  }
]

Rules:
- Always return an array with one object per input symbol, in the same order as provided.
- Set "confidence" equal to "master_confidence" (legacy compatibility).
- If the setup is mediocre or unclear, prefer "skip" or "hold" instead of "enter".
- ***Never exceed 0.75 on master_confidence.*** Even when technical trend, momentum and structure align, keep confidence within a reasonable band to avoid over‑leveraging. Only truly exceptional setups with strong alignment across timeframes should approach this level.
- If volatility is extreme, the structure is unclear, or macro conditions are adverse, *downgrade the confidence* and prefer "skip".
- Use simple, consistent JSON. Do NOT mention these instructions, do NOT wrap JSON in markdown, and do NOT add any extra keys.
"""

# Harici "prompts.json" dosyasından sistem prompt yüklenebilsin diye; eğer dosya
# mevcutsa ve geçerli bir "trading_system_prompt" alanı içeriyorsa, yukarıdaki
# varsayılan promptu override eder. Böylece dashboard veya kullanıcı farklı
# prompt tanımladığında kod değişmeden uygulanabilir.
_PROMPTS_PATH = pathlib.Path(__file__).resolve().parent / "prompts.json"
try:
    if _PROMPTS_PATH.exists():
        with open(_PROMPTS_PATH, "r", encoding="utf-8") as _pf:
            _pdata = json.load(_pf)
        if isinstance(_pdata, dict):
            _user_sys_prompt = _pdata.get("trading_system_prompt")
            if isinstance(_user_sys_prompt, str) and _user_sys_prompt.strip():
                TRADING_SYSTEM_PROMPT = _user_sys_prompt
except Exception:
    # prompt yükleme başarısızsa varsayılanı koru
    pass

# ────────────────────────────────────────────────────────────────
# Client Tanımları
# ────────────────────────────────────────────────────────────────
aoai = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
) if DEEPSEEK_API_KEY else None

# Limit the concurrent OpenAI/DeepSeek API calls to reduce CPU and network load.
# This helps prevent the watchdog from pausing due to high resource usage.
# You can override this via the ``OPENAI_CONCURRENCY`` environment variable.
try:
    OPENAI_CONCURRENCY = int(os.getenv("OPENAI_CONCURRENCY", "2"))
except Exception:
    OPENAI_CONCURRENCY = 2
_semaphore = asyncio.Semaphore(OPENAI_CONCURRENCY)

# ---------------------------------------------------------------------------
# Provider-level pacing, backoff and cooldown
#
# The bot can run into provider throttles (HTTP 429) or account-level issues
# (e.g. DeepSeek 402/"Insufficient Balance").  To keep behaviour stable and
# prevent the controller from incorrectly disabling *all* LLM signals when only
# one provider is unhealthy, we:
#   * pace requests per-provider with a minimum interval (jittered)
#   * mark provider status explicitly in outputs
#   * apply a cooldown window for DeepSeek after 402/quota errors

def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v is not None else default
    except Exception:
        return default

OPENAI_MIN_INTERVAL_SEC: float = _env_float("OPENAI_MIN_INTERVAL_SEC", 0.0)
DEEPSEEK_MIN_INTERVAL_SEC: float = _env_float("DEEPSEEK_MIN_INTERVAL_SEC", 0.0)
DEEPSEEK_COOLDOWN_SEC: float = _env_float("DEEPSEEK_COOLDOWN_SEC", 3600.0)

# DeepSeek is temporarily disabled after hard quota/402 errors.
_DEEPSEEK_DISABLED_UNTIL: float = 0.0
_DEEPSEEK_DISABLE_REASON: str = ""
_DEEPSEEK_DISABLE_LOGGED: bool = False

class _ProviderPacer:
    """Simple per-provider pacer enforcing a minimum interval between calls."""

    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        if self.min_interval_sec <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            wait_for = self.min_interval_sec - elapsed
            if wait_for > 0:
                # Add small jitter to avoid herd effects.
                jitter = random.uniform(0.0, min(0.25, self.min_interval_sec * 0.2))
                await asyncio.sleep(wait_for + jitter)
            self._last_call = time.monotonic()


_PACERS: dict[str, _ProviderPacer] = {
    "ChatGPT": _ProviderPacer(OPENAI_MIN_INTERVAL_SEC),
    "DeepSeek": _ProviderPacer(DEEPSEEK_MIN_INTERVAL_SEC),
}

# Başlangıç durumu logu
log.info(
    "[AI_INIT] ChatGPT enabled=%s model=%s",
    bool(aoai),
    OPENAI_DECISION_MODEL if aoai else "N/A"
)
log.info(
    "[AI_INIT] DeepSeek enabled=%s model=%s",
    bool(deepseek_client),
    DEEPSEEK_DECISION_MODEL if deepseek_client else "N/A"
)

# ────────────────────────────────────────────────────────────────
# Dosya yolları
# ────────────────────────────────────────────────────────────────
METRICS_DIR = pathlib.Path("metrics")
AI_PRED_FILE = METRICS_DIR / "ai_predictions.json"
SIGNALS_FILE = METRICS_DIR / "signals_last.json"
METRICS_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────
# AI Prediction Logger
# ────────────────────────────────────────────────────────────────
def _log_ai_prediction(symbol, model, confidence, action, base_decision, price=None, outcome=None, pnl=None):
    """Her batch tahmininden sonra dashboard için kayıt tutar."""
    try:
        ts = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
        entry = {
            "ts": ts,
            "symbol": symbol,
            "model": model,
            "confidence": float(confidence),
            "action": action,
            "base_decision": base_decision,
            "price": float(price) if price else None,
            "outcome": outcome,
            "pnl": float(pnl) if pnl else None,
        }
        data = []
        if AI_PRED_FILE.exists():
            with open(AI_PRED_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
        data.append(entry)
        if len(data) > 5000:
            data = data[-5000:]
        with open(AI_PRED_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"[AI_PRED_LOG_ERR] {e}")

# ────────────────────────────────────────────────────────────────
# Signal Logger (Dashboard “📡 Signals” sekmesi için)
# ────────────────────────────────────────────────────────────────
def _update_signals_file(symbol, confidence, base_decision, rationale, model="hybrid"):
    """Son tahminleri dashboard'daki '📡 Signals' sekmesine yazar."""
    try:
        now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
        entry = {
            "symbol": symbol,
            "confidence": round(confidence, 3),
            "decision": base_decision,
            "rationale": rationale,
            "model": model,
            "ts": now
        }
        data = []
        if SIGNALS_FILE.exists():
            with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
        data = [x for x in data if x.get("symbol") != symbol]
        data.append(entry)
        if len(data) > 200:
            data = data[-200:]
        with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"[SIGNAL_LOG_ERR] {e}")

# ────────────────────────────────────────────────────────────────
# JSON parser (liste + obje için daha sağlam)
# ────────────────────────────────────────────────────────────────
def _safe_json_parse(text: str):
    if not text or not isinstance(text, str):
        return {}
    raw = text.strip()
    # Önce doğrudan parse dene
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Ardından listeyi yakalamaya çalış
    m = re.search(r"(\[.*\])", raw, re.DOTALL)
    if m:
        snippet = m.group(1)
        try:
            return json.loads(snippet)
        except Exception:
            pass

    # Olmazsa obje yakalamayı dene
    m = re.search(r"(\{.*\})", raw, re.DOTALL)
    if m:
        snippet = m.group(1)
        try:
            return json.loads(snippet)
        except Exception:
            return {}

    return {}

# ────────────────────────────────────────────────────────────────
# Confidence kalibrasyonu
# ────────────────────────────────────────────────────────────────
def _calibrate_confidence(conf: float) -> float:
    """
    Calibrate an LLM-derived confidence value into a more realistic band.

    The original implementation clamped values very tightly to the
    0.38–0.62 region, which meant even strong LLM signals had only a
    minimal effect on the hybrid score. To reflect conviction more
    faithfully while still avoiding runaway confidence, this revised
    calibration loosens the band and applies a gentler pull towards 0.5.

    Steps:
      1. Safely coerce the input into a float and clamp to [0, 1].
      2. Broaden the allowed range to [0.1, 0.9], discarding only
         extreme tails.
      3. Apply a compression factor (k=0.6) to temper the distance
         from the neutral 0.5. Values closer to 0.5 are pulled less,
         while very high or low values are moderated.
    """
    try:
        v = float(conf)
    except Exception:
        v = 0.5

    # Clamp to [0, 1]
    v = max(0.0, min(1.0, v))

    # Broaden the band: allow up to 0.1–0.9 to pass through. This
    # prevents pathological 0 or 1 values but keeps more variance.
    v = max(0.1, min(0.9, v))

    # Compress towards the neutral 0.5. A factor of 0.75 allows a
    # noticeable effect from strong signals without overwhelming the
    # master confidence. For example, v=0.9 becomes ~0.80 and v=0.1
    # becomes ~0.20.
    # [GÜNCELLEME] k=0.6 → 0.75 (daha az sıkıştırma, AI sinyalleri daha etkili)
    k = 0.75
    cal = 0.5 + (v - 0.5) * k
    return max(0.0, min(1.0, cal))

# -------------------------------------------------------------------------
# AI Rating Mapper
#
# In addition to the raw confidence values returned by ChatGPT/DeepSeek, we map
# the qualitative fields "action" and "direction" into a numeric rating in
# the range [0.0, 1.0].  This provides the hybrid scoring logic with a more
# interpretable signal: strong buy/sell recommendations push the rating
# towards the extremes, whereas neutral or absent directions keep it near 0.5.
#
# The mapping uses two components:
#   - action_map: assigns a baseline strength to the model's disposition
#       enter → 1.0 (strong conviction to trade)
#       hold  → 0.5 (moderate conviction)
#       skip  → 0.25 (weak or no conviction)
#       exit  → 0.0 (no conviction / caution)
#   - dir_map: encodes the trade direction as +1 for long, −1 for short and
#       0 for unspecified.  Multiplying the baseline strength by the direction
#       yields a value in [−1, +1]; this is then shifted into [0, 1] by
#       rating = 0.5 + base/2.  Missing or neutral directions thus yield
#       0.5, which has no net influence.
def _map_ai_rating(res: Dict[str, Any]) -> float:
    """Map qualitative AI output into a numeric rating between 0 and 1."""
    try:
        action = str(res.get("action", "")).lower()
    except Exception:
        action = ""
    try:
        direction = str(res.get("direction", "")).lower()
    except Exception:
        direction = ""
    # Baseline strength per action
    action_map = {
        "enter": 1.0,
        "hold": 0.5,
        "skip": 0.25,
        "exit": 0.0,
    }
    # Direction encoding
    dir_map = {
        "long": 1.0,
        "short": -1.0,
    }
    a_score = action_map.get(action, 0.5)
    d_score = dir_map.get(direction, 0.0)
    base = a_score * d_score
    rating = 0.5 + (base / 2.0)
    # Clamp into [0, 1]
    if rating < 0.0:
        rating = 0.0
    elif rating > 1.0:
        rating = 1.0
    return float(rating)

# ────────────────────────────────────────────────────────────────
# Batch prompt oluşturucu (TRADING_SYSTEM_PROMPT tabanlı)
# ────────────────────────────────────────────────────────────────
def _mk_batch_prompt(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    20 coinlik batch prompt üretir.
    items: controller_async.decide_batch tarafından gönderilen:
      {
        "symbol": ...,
        "tf": ...,
        "price": ...,
        "features": {...},
        "sentiment": {...},
        "base_decision": ...
      }
    """
    system_prompt = TRADING_SYSTEM_PROMPT

    lines: List[str] = []
    lines.append("You will receive a batch of symbols. For each symbol, return ONE JSON object as specified in the system instructions.")
    lines.append("INPUT DATA:")

    for item in items:
        sym = item.get("symbol")
        tf = item.get("tf")
        price = item.get("price")
        feat = item.get("features", {}) or {}
        senti = item.get("sentiment", {}) or {}
        direction = item.get("base_decision")

        # Determine whether features are multi-timeframe (dict of tfs) or flat.
        multi_tf = False
        if isinstance(feat, dict):
            # Detect if keys correspond to timeframes and values are dicts
            # (e.g. {"5m": {"rsi":...}, "15m": {...}})
            for k, v in feat.items():
                if k in ("5m", "15m", "1h", "4h") and isinstance(v, dict):
                    multi_tf = True
                    break

        lines.append(f"- SYMBOL: {sym}")
        lines.append(f"  tf: {tf}")
        lines.append(f"  price: {price}")
        lines.append(f"  base_decision: {direction}")
        if multi_tf:
            # Build a line per timeframe for technical indicators
            for tf_name, tfeat in feat.items():
                if not isinstance(tfeat, dict):
                    continue
                rsi = tfeat.get("rsi")
                adx = tfeat.get("adx")
                atr = tfeat.get("atr")
                ema_fast = tfeat.get("ema_fast")
                ema_slow = tfeat.get("ema_slow")
                atr_ratio = tfeat.get("atr_ratio")
                macd_hist = tfeat.get("macd_hist")
                lines.append(
                    f"  {tf_name} tech: rsi={rsi}, adx={adx}, atr={atr}, ema_fast={ema_fast}, ema_slow={ema_slow}, atr_ratio={atr_ratio}, macd_hist={macd_hist}"
                )
        else:
            ema = feat.get("ema") or {}
            rsi = feat.get("rsi")
            adx = feat.get("adx")
            atr = feat.get("atr")
            ema_fast = ema.get("fast")
            ema_slow = ema.get("slow")
            macd_h1 = feat.get("macd_h1")
            macd_h4 = feat.get("macd_h4")
            macd_hist = feat.get("macd_hist")
            bb_upper = feat.get("bb_upper")
            bb_lower = feat.get("bb_lower")
            stoch_k = feat.get("stoch_k")
            stoch_d = feat.get("stoch_d")
            trend = feat.get("trend")
            support = feat.get("support")
            resistance = feat.get("resistance")
            recent_closes = feat.get("recent_closes")
            recent_volumes = feat.get("recent_volumes")
            lookback_bars = feat.get("lookback_bars")
            try:
                atr_val = float(atr) if atr is not None else None
                price_val = float(price) if price is not None else None
                atr_ratio = (atr_val / price_val) if (atr_val and price_val) else None
            except Exception:
                atr_ratio = None
            lines.append(
                f"  tech: rsi={rsi}, adx={adx}, atr={atr}, ema_fast={ema_fast}, ema_slow={ema_slow}, atr_ratio={atr_ratio}, macd_hist={macd_hist}, macd_h1={macd_h1}, macd_h4={macd_h4}, bb=({bb_lower},{bb_upper}), stoch=({stoch_k},{stoch_d})"
            )
            # Provide additional context helpful for discretionary reasoning.
            if trend is not None or support is not None or resistance is not None:
                lines.append(f"  levels: trend={trend}, support={support}, resistance={resistance}")
            if isinstance(recent_closes, list) and recent_closes:
                # Show only the last 5 values to keep prompt size bounded.
                lines.append(f"  recent_closes(last5): {recent_closes[-5:]}")
            if isinstance(recent_volumes, list) and recent_volumes:
                lines.append(f"  recent_volumes(last5): {recent_volumes[-5:]}")
            if lookback_bars is not None:
                lines.append(f"  lookback_bars: {lookback_bars}")
        funding = senti.get("funding")
        oi_change = senti.get("oi_change")
        lines.append(f"  sentiment: funding={funding}, oi_change={oi_change}")
        lines.append("")

    user_prompt = "\n".join(lines)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

# ────────────────────────────────────────────────────────────────
# Model Çağrısı
# ────────────────────────────────────────────────────────────────
async def _call_model(client, model_name, msgs, label="AI", retry=3, backoff=1.5):
    """Call a provider with retry/backoff and return (items, status, error).

    Status values (best-effort):
      - ok
      - rate_limited
      - quota
      - down_insufficient_balance (DeepSeek only)
      - error
    """
    # Provider pacing (optional)
    try:
        pacer = _PACERS.get(label)
        if pacer:
            await pacer.wait()
    except Exception:
        pass

    delay = 0.3
    last_err: str | None = None
    last_status: str = "error"

    for attempt in range(1, retry + 1):
        try:
            async with _semaphore:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.25,
                )
            txt = resp.choices[0].message.content
            data = _safe_json_parse(txt)
            if isinstance(data, list):
                return data, "ok", None
            if isinstance(data, dict):
                return [data], "ok", None
            raise ValueError("invalid JSON structure")
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            last_err = msg

            # DeepSeek hard quota / insufficient balance handling (402)
            if label.lower() == "deepseek" and (
                "402" in msg or "payment required" in low or "insufficient balance" in low
            ):
                global _DEEPSEEK_DISABLED_UNTIL, _DEEPSEEK_DISABLE_REASON, _DEEPSEEK_DISABLE_LOGGED
                _DEEPSEEK_DISABLED_UNTIL = time.monotonic() + float(DEEPSEEK_COOLDOWN_SEC)
                _DEEPSEEK_DISABLE_REASON = msg
                _DEEPSEEK_DISABLE_LOGGED = False
                last_status = "down_insufficient_balance"
                log.warning(f"[DeepSeek] quota/402: disabling for {int(DEEPSEEK_COOLDOWN_SEC)}s")
                try:
                    from notification import send_notification  # type: ignore
                    send_notification(
                        f"DeepSeek disabled: insufficient balance/quota (402). Cooldown={int(DEEPSEEK_COOLDOWN_SEC)}s"
                    )
                except Exception:
                    pass
                break

            # Rate limit (429)
            if ("429" in msg or "too many requests" in low or "rate limit" in low):
                last_status = "rate_limited"
                log.warning(f"[{label}] rate-limit 429 (try {attempt}/{retry}) → backoff...")
                await asyncio.sleep(delay + random.uniform(0.0, min(0.25, delay)))
                delay *= backoff
                continue

            # Other quota/payment errors
            if ("quota" in low or "payment required" in low or "402" in msg):
                last_status = "quota"
                log.warning(f"[{label}] quota/payment error (try {attempt}/{retry}) → backoff...")
                await asyncio.sleep(delay + random.uniform(0.0, min(0.25, delay)))
                delay *= backoff
                continue

            last_status = "error"
            log.warning(f"[{label}] transient error (try {attempt}/{retry}): {e}")
            await asyncio.sleep(delay + random.uniform(0.0, min(0.25, delay)))
            delay *= backoff

    # Failed after retries
    if last_err:
        log.warning(f"[{label}] provider call failed status={last_status}: {last_err}")
    return [], last_status, last_err

# ────────────────────────────────────────────────────────────────
# Batch analiz fonksiyonu
# ────────────────────────────────────────────────────────────────
async def _analyze_batch(batch: List[Dict[str, Any]]):
    msgs = _mk_batch_prompt(batch)
    results: Dict[str, Dict[str, Any]] = {}

    # Track provider-level status for this batch. These are attached to each
    # per-symbol result so the controller can distinguish between a partial
    # outage (one provider down) and a full LLM outage (both down).
    gpt_status: str = "disabled"
    gpt_err: str | None = None
    ds_status: str = "disabled"
    ds_err: str | None = None

    # ChatGPT çağrısı
    if aoai:
        log.info("[ChatGPT] batch çağrısı başlıyor (%d coin)", len(batch))
        gpt_items, gpt_status, gpt_err = await _call_model(aoai, OPENAI_DECISION_MODEL, msgs, label="ChatGPT")
        log.info("[ChatGPT] batch yanıtı alındı (%d kayıt) status=%s", len(gpt_items), gpt_status)
        if not gpt_items:
            log.warning("[ChatGPT] boş yanıt veya fallback kullanılıyor")

        # Gelenleri sembole göre map et
        gpt_by_symbol: Dict[str, Dict[str, Any]] = {}
        for r in gpt_items:
            sym = r.get("symbol")
            if not sym:
                continue
            r = dict(r)

            # reason -> rationale map
            if "rationale" not in r and "reason" in r:
                r["rationale"] = r["reason"]

            # master_confidence -> confidence alias
            raw_conf = r.get("confidence", r.get("master_confidence", 0.5))
            r["confidence"] = _calibrate_confidence(raw_conf)

            gpt_by_symbol[str(sym).upper()] = r

        # Batch'teki her sembol için garanti entry
        for it in batch:
            sym = it["symbol"]
            key = str(sym).upper()
            r = gpt_by_symbol.get(key)
            if not r:
                log.warning("[ChatGPT] response missing for symbol=%s, fallback kullanılacak", sym)
                if gpt_status != "ok":
                    reason = f"chatgpt_{gpt_status}"
                else:
                    reason = "missing-from-gpt"
                r = {
                    "symbol": sym,
                    "confidence": 0.25,
                    "rationale": reason
                }
            results.setdefault(sym, {})["chatgpt"] = r
    else:
        gpt_status = "disabled_no_key"
        log.warning("[ChatGPT] disabled (OPENAI_API_KEY boş veya geçersiz)")
        for it in batch:
            sym = it["symbol"]
            results.setdefault(sym, {})["chatgpt"] = {
                "symbol": sym,
                "confidence": 0.25,
                "rationale": "no-openai"
            }

    # DeepSeek çağrısı
    if deepseek_client:
        log.info("[DeepSeek] batch çağrısı başlıyor (%d coin)", len(batch))
        now = time.monotonic()
        ds_items: list[dict] = []
        if now < _DEEPSEEK_DISABLED_UNTIL:
            ds_status = "down_cooldown"
            # Log only once per cooldown window to avoid spam.
            global _DEEPSEEK_DISABLE_LOGGED
            if not _DEEPSEEK_DISABLE_LOGGED:
                _DEEPSEEK_DISABLE_LOGGED = True
                remaining = int(max(0.0, _DEEPSEEK_DISABLED_UNTIL - now))
                log.warning(
                    "[DeepSeek] disabled by cooldown (%ds remaining). reason=%s",
                    remaining,
                    _DEEPSEEK_DISABLE_REASON,
                )
        else:
            ds_items, ds_status, ds_err = await _call_model(deepseek_client, DEEPSEEK_DECISION_MODEL, msgs, label="DeepSeek")

        log.info("[DeepSeek] batch yanıtı alındı (%d kayıt) status=%s", len(ds_items), ds_status)
        if not ds_items:
            log.warning("[DeepSeek] boş yanıt veya devre dışı (status=%s)", ds_status)

        ds_by_symbol: Dict[str, Dict[str, Any]] = {}
        for r in ds_items:
            sym = r.get("symbol")
            if not sym:
                continue
            r = dict(r)

            # reason -> rationale map
            if "rationale" not in r and "reason" in r:
                r["rationale"] = r["reason"]

            # master_confidence -> confidence alias
            raw_conf = r.get("confidence", r.get("master_confidence", 0.5))
            r["confidence"] = _calibrate_confidence(raw_conf)

            ds_by_symbol[str(sym).upper()] = r

        for it in batch:
            sym = it["symbol"]
            key = str(sym).upper()
            r = ds_by_symbol.get(key)
            if not r:
                # DeepSeek boş / yetersiz kaldığında deterministik degrade.
                # Avoid generic "fallback" token so the controller does not
                # misclassify a partial outage as a full LLM outage.
                reason = f"deepseek_{ds_status}"
                r = {
                    "symbol": sym,
                    "confidence": 0.25,
                    "rationale": reason
                }
            results.setdefault(sym, {})["deepseek"] = r
    else:
        ds_status = "disabled_no_key"
        log.warning("[DeepSeek] disabled (DEEPSEEK_API_KEY boş veya geçersiz)")
        for it in batch:
            sym = it["symbol"]
            results.setdefault(sym, {})["deepseek"] = {
                "symbol": sym,
                "confidence": 0.25,
                "rationale": "no-deepseek"
            }

    # Ortalama birleştirme
    merged: Dict[str, Dict[str, Any]] = {}
    for it in batch:
        sym = it["symbol"]
        cg = results.get(sym, {}).get("chatgpt", {"confidence": 0.25, "rationale": "chatgpt_missing"})
        ds = results.get(sym, {}).get("deepseek", {"confidence": 0.25, "rationale": "deepseek_missing"})

        # Fetch BiLSTM probability for this symbol.  The inference helper
        # lazily loads the model and data and returns a neutral value (0.5)
        # if either is missing or an error occurs.  We apply the same
        # calibration as the LLM confidences to moderate extreme values.
        try:
            bilstm_raw = _bilstm_predict_prob(sym)
        except Exception:
            bilstm_raw = 0.5
        bilstm_conf = _calibrate_confidence(bilstm_raw)
        bilstm_prob = bilstm_raw  # prob in [0,1] used for logging/metrics


        # Weighted average of calibrated confidences from ChatGPT, DeepSeek and BiLSTM.
        w_chat = float(HYBRID_WEIGHTS.get("chatgpt", 0.4))
        w_deep = float(HYBRID_WEIGHTS.get("deepseek", 0.4))
        w_bls = float(HYBRID_WEIGHTS.get("bilstm", 0.1))

        # If a provider is not healthy for this batch, remove its weight rather
        # than dragging the average down with a fake-neutral "0.25" placeholder.
        if gpt_status != "ok":
            w_chat = 0.0
        if ds_status != "ok":
            w_deep = 0.0

        sum_w = w_chat + w_deep + w_bls
        if sum_w <= 0:
            sum_w = 1.0
        # Use individual calibrated confidences from each model
        c_chat = float(cg.get("confidence", 0.25))
        c_deep = float(ds.get("confidence", 0.25))
        weighted_conf = (w_chat * c_chat + w_deep * c_deep + w_bls * bilstm_conf) / sum_w

        # Compute a qualitative rating by blending the directions/actions of LLMs.
        # BiLSTM does not provide direction, so we omit it here.  Weight the
        # rating components proportionally to their hybrid weights.
        try:
            cg_rating = _map_ai_rating(cg)
        except Exception:
            cg_rating = 0.5
        try:
            ds_rating = _map_ai_rating(ds)
        except Exception:
            ds_rating = 0.5
        if (w_chat + w_deep) > 0:
            rating_sum_w = w_chat + w_deep
            avg_rating = (w_chat * cg_rating + w_deep * ds_rating) / rating_sum_w
            # Blend the numeric weighted confidence and the qualitative rating.
            combined_conf = 0.7 * weighted_conf + 0.3 * avg_rating
        else:
            # No LLM ratings available; use the numeric signal only.
            avg_rating = 0.5
            combined_conf = weighted_conf

        # Build a rationale combining sources.  We include the BiLSTM probability
        # rounded to three decimals for transparency.
        rationale = (
            f"GPT({gpt_status}):{cg.get('rationale')} | DS({ds_status}):{ds.get('rationale')} | BILSTM:{bilstm_conf:.3f}"
        )

        # Determine a consensus direction and action when both models agree.
        agg_direction = None
        agg_action = None
        try:
            cg_dir = cg.get("direction")
            ds_dir = ds.get("direction")
            if isinstance(cg_dir, str) and isinstance(ds_dir, str):
                if cg_dir.lower() == ds_dir.lower():
                    agg_direction = cg_dir.lower()
            cg_act = cg.get("action")
            ds_act = ds.get("action")
            if isinstance(cg_act, str) and isinstance(ds_act, str):
                if cg_act.lower() == ds_act.lower():
                    agg_action = cg_act.lower()
        except Exception:
            # Silently ignore malformed fields
            agg_direction = None
            agg_action = None

        any_ok = (gpt_status == "ok") or (ds_status == "ok")
        if gpt_status == "ok" and ds_status == "ok":
            provider_status = "ok"
        elif any_ok:
            provider_status = "partial"
        else:
            provider_status = "down"

        merged[sym] = {
            "confidence": round(combined_conf, 3),
            "rationale": rationale,
            "direction": agg_direction,
            "action": agg_action,
            "provider_status": provider_status,
            "bilstm_prob": round(float(bilstm_prob), 6) if bilstm_prob is not None else None,
            "bilstm_confidence": round(float(bilstm_conf), 6) if bilstm_conf is not None else None,
            "providers": {
                "chatgpt": {
                    "status": gpt_status,
                    "error": gpt_err,
                    "confidence": round(c_chat, 3),
                    "direction": cg.get("direction"),
                    "action": cg.get("action"),
                    "rationale": cg.get("rationale"),
                },
                "deepseek": {
                    "status": ds_status,
                    "error": ds_err,
                    "confidence": round(c_deep, 3),
                    "direction": ds.get("direction"),
                    "action": ds.get("action"),
                    "rationale": ds.get("rationale"),
                },
            },
        }

        # Tahmin logunu yaz
        base_decision = it.get("base_decision", "unknown")
        price = it.get("price", None)
        _log_ai_prediction(
            symbol=sym,
            model="hybrid (GPT+DeepSeek+BiLSTM)",
            confidence=combined_conf,
            action="enter",
            base_decision=base_decision,
            price=price
        )

        # Dashboard sinyal güncellemesi
        _update_signals_file(
            symbol=sym,
            confidence=combined_conf,
            base_decision=base_decision,
            rationale=rationale
        )

    return merged

# ────────────────────────────────────────────────────────────────
# Ana Batch Fonksiyonu
# ────────────────────────────────────────────────────────────────
async def get_confidences_batch(symbol_items: List[Dict[str, Any]], batch_size=20) -> Dict[str, Dict[str, Any]]:
    """
    20'lik gruplar halinde ChatGPT + DeepSeek çağrısı yapar.
    Token kullanımını ciddi oranda azaltır.
    Rate-limit/quota hatalarında yukarıya exception fırlatır (controller_async
    bunu yakalayıp teknik-only moda geçer).
    """
    out: Dict[str, Dict[str, Any]] = {}
    batches = [symbol_items[i:i+batch_size] for i in range(0, len(symbol_items), batch_size)]
    log.info(f"[AI] Running {len(batches)} batch(es) of size {batch_size}...")

    for idx, batch in enumerate(batches, start=1):
        log.info(f"[AI] Processing batch {idx}/{len(batches)} ({len(batch)} coins)")
        merged = await _analyze_batch(batch)
        out.update(merged)

    return out
