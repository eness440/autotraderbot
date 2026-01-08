# -*- coding: utf-8 -*-
"""
ChatGPT client wrapper (Hybrid: 4o-mini for filtering, 4o for decision).
- Reads API key from env: OPENAI_API_KEY
- Safe retries + timeouts
- Small cost guardrails (token limits)
"""

import os
import time
from typing import List, Dict, Any, Optional

# OpenAI official SDK (>=1.0)
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "OpenAI SDK bulunamadı. Kurulum: `py -3.14 -m pip install --upgrade openai`"
    ) from e


DEFAULT_FILTER_MODEL = os.getenv("OPENAI_FILTER_MODEL", "gpt-4o-mini")
DEFAULT_DECISION_MODEL = os.getenv("OPENAI_DECISION_MODEL", "gpt-4o")
API_KEY = os.getenv("OPENAI_API_KEY", "")

if not API_KEY:
    # Bilerek Exception atmıyoruz; proje demo/paper modda anahtar olmayabilir.
    print("[UYARI] OPENAI_API_KEY set edilmemiş. ChatGPT çağrıları başarısız olabilir.")

class ChatGPTClient:
    def __init__(
        self,
        filter_model: str = DEFAULT_FILTER_MODEL,
        decision_model: str = DEFAULT_DECISION_MODEL,
        request_timeout: int = 30,
        max_retries: int = 2,
    ) -> None:
        # Eğer LLM kotası aşıldıysa veya ortam değişkeni ile devre dışı
        # bırakıldıysa, client'ı None olarak ayarla ve çağrıları
        # pasif moda geçir. LLM_DISABLE veya CHATGPT_DISABLE=1
        disable_env = os.getenv("CHATGPT_DISABLE", "0")
        self.enabled = disable_env not in ("1", "true", "True")
        if self.enabled:
            self.client = OpenAI(api_key=API_KEY)
        else:
            self.client = None
        self.filter_model = filter_model
        self.decision_model = decision_model
        self.request_timeout = request_timeout
        self.max_retries = max_retries

    # ---- Internal helper ----
    def _chat(self, model: str, messages: List[Dict[str, str]]) -> str:
        """
        Call OpenAI Chat Completions with simple retry.
        """
        # LLM kullanımının devre dışı bırakılması durumunda, boş bir
        # yanıt döndür ve sisteme geriye düşmesine izin ver. Bu, 'tech-only'
        # veya 'sentiment-only' modunda botun çalışmasını sağlar.
        if not getattr(self, 'enabled', True) or self.client is None:
            return ""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    timeout=self.request_timeout,
                )
                # New SDK returns .choices[0].message.content
                if resp and resp.choices:
                    return resp.choices[0].message.content or ""
                return ""
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"OpenAI chat hatası: {last_err}")

    # ---- Public APIs ----
    def filter_signal(self, snapshot: Dict[str, Any], base_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lightweight, cheap classification with GPT-4o-mini (2. katman).
        Returns dict: {allow: bool, reason: str, risk_flag: str}
        """
        # Minimize prompt size: include only essential fields
        sym = snapshot.get("symbol")
        tf = snapshot.get("timeframe")
        features = snapshot.get("features", {})
        risk = snapshot.get("risk", {})
        base_decision = base_signal.get("decision")

        sys_prompt = (
            "You are a strict risk filter for a crypto futures bot. "
            "Your job: accept or reject the base signal to avoid bad trades. "
            "Prefer safety. Output must be JSON with keys: allow (bool), risk_flag (str), reason (str <= 200 chars)."
        )
        # Kullanıcı prompt'unu f-string olarak oluştururken sözlük
        # gösterimleri için çift süslü parantez kullanarak f-string
        # süslü parantezlerinden kaçınırız. Böylece f-string hata vermez.
        user_prompt = (
            f"symbol={sym}, timeframe={tf}\n"
            f"base_decision={base_decision}\n"
            f"features={{'ema': {features.get('ema')}, 'rsi': {features.get('rsi')}, 'adx': {features.get('adx')}, 'fibo': {features.get('fibo')}, 'atr': {features.get('atr')}}}\n"
            f"risk={{'vol': {risk.get('vol')}, 'dd': {risk.get('dd')}, 'lev': {risk.get('lev')}}}\n"
            "Rules:\n"
            "- Reject if ADX<12 or RSI in [45,55] and momentum weak.\n"
            "- Reject if leverage>50 and ATR spike>2x baseline.\n"
            "- Otherwise allow.\n"
            "Respond STRICTLY with JSON only."
        )

        content = self._chat(
            self.filter_model,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Best-effort JSON parsing
        import json
        try:
            data = json.loads(content)
            return {
                "allow": bool(data.get("allow", False)),
                "risk_flag": str(data.get("risk_flag", "unknown")),
                "reason": str(data.get("reason", ""))[:200],
            }
        except Exception:
            return {"allow": False, "risk_flag": "parse_error", "reason": content[:200]}

    def decide_trade(self, snapshot: Dict[str, Any], filtered: Dict[str, Any]) -> Dict[str, Any]:
        """
        Heavier reasoning with GPT-4o (3. katman karar revizyonu).
        Returns dict: {action: "enter|skip|modify", tp: float?, sl: float?, lev: int?, reason: str}
        """
        sym = snapshot.get("symbol")
        tf = snapshot.get("timeframe")
        px = snapshot.get("price")
        atr = snapshot.get("features", {}).get("atr")
        gscore = snapshot.get("gscore")
        funding = snapshot.get("sentiment", {}).get("funding")
        oi = snapshot.get("sentiment", {}).get("oi_change")
        filt_allow = filtered.get("allow")

        sys_prompt = (
            "You are the final decision layer for a crypto futures bot (3rd layer). "
            "Use conservative leverage if volatility is high. Prefer capital preservation. "
            "Output must be JSON with keys: action('enter'|'skip'|'modify'), "
            "tp (float), sl (float), lev (int), reason (<=220 chars)."
        )
        user_prompt = (
            f"symbol={sym}, timeframe={tf}, price={px}, atr={atr}, gscore={gscore}, "
            f"funding={funding}, oi_change={oi}, filter_allow={filt_allow}.\n"
            "Rules:\n"
            "- If filter_allow=false => skip.\n"
            "- If gscore<55 => skip.\n"
            "- If enter: set SL = price - 0.6*ATR (long) or price + 0.6*ATR (short) depending on direction from gscore/momentum.\n"
            "- Leverage map: g>=95->40x, g>=85->25x, g>=75->15x, g>=65->10x, else 5x.\n"
            "- If funding extreme negative and momentum up, allow long bias; vice versa.\n"
            "Respond STRICTLY with JSON only."
        )

        content = self._chat(
            self.decision_model,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        import json
        try:
            data = json.loads(content)
            out = {
                "action": str(data.get("action", "skip")),
                "tp": float(data.get("tp")) if data.get("tp") is not None else None,
                "sl": float(data.get("sl")) if data.get("sl") is not None else None,
                "lev": int(data.get("lev")) if data.get("lev") is not None else None,
                "reason": str(data.get("reason", ""))[:220],
            }
            return out
        except Exception:
            return {"action": "skip", "tp": None, "sl": None, "lev": None, "reason": content[:220]}
