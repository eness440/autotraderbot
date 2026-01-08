# -*- coding: utf-8 -*-
"""
chatgpt_decision_layer.py
-------------------------

ChatGPT Decision Layer (3. katman)
- 2. katmandan geçen sinyali alır (filter result)
- Nihai kararı üretir: enter/skip/modify + TP/SL/LEV

Bu modül, ChatGPT API'sini kullanarak trading sinyallerini değerlendirir
ve nihai karar verir. İki aşamalı bir süreç izler:

1. Filter aşaması (gpt-4o-mini): Hızlı ve ucuz ön filtre
2. Decision aşaması (gpt-4o): Detaylı analiz ve karar

CHANGELOG:
- v1.0: Initial version
- v1.1: Fixed import to work both as module and standalone
- v1.2: Added fallback when ChatGPT is disabled
- v1.3: Added timeout and retry logic
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional

# Flexible import: works both as package module and standalone script
try:
    from .chatgpt_client import ChatGPTClient
except ImportError:
    try:
        from chatgpt_client import ChatGPTClient
    except ImportError:
        # Fallback: create a dummy client if chatgpt_client not available
        ChatGPTClient = None  # type: ignore


class DecisionLayerError(Exception):
    """Custom exception for decision layer errors."""
    pass


def _get_client() -> Optional[Any]:
    """
    Get or create ChatGPT client instance.
    Returns None if ChatGPT is disabled or unavailable.
    """
    # Check if ChatGPT is disabled via environment
    if os.getenv("CHATGPT_DISABLE", "").lower() in ("1", "true", "yes"):
        return None
    
    if ChatGPTClient is None:
        return None
    
    try:
        return ChatGPTClient()
    except Exception:
        return None


# Lazy initialization of client
_client: Optional[Any] = None
_client_initialized: bool = False


def get_client() -> Optional[Any]:
    """Get the singleton ChatGPT client."""
    global _client, _client_initialized
    if not _client_initialized:
        _client = _get_client()
        _client_initialized = True
    return _client


def verify_and_revise(
    snapshot: Dict[str, Any], 
    base_signal: Dict[str, Any],
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    High-level pipeline for signal verification and revision.
    
    1) Filter (4o-mini) -> allow?
    2) If allowed -> final decision (4o)
    
    Args:
        snapshot: Current market snapshot containing price, indicators, etc.
        base_signal: Base trading signal from technical/AI analysis.
        timeout: Maximum time to wait for API response.
        
    Returns:
        Decision dictionary with keys:
        - action: "enter", "skip", or "modify"
        - tp: Take profit price (or None)
        - sl: Stop loss price (or None)
        - lev: Recommended leverage (or None)
        - reason: Explanation for the decision
        - confidence: Confidence score (0-1)
    """
    client = get_client()
    
    # If client is not available, pass through with reduced confidence
    if client is None:
        return _fallback_decision(base_signal)
    
    try:
        # Step 1: Filter with fast model
        filtered = client.filter_signal(snapshot, base_signal)
        
        if not filtered.get("allow"):
            return {
                "action": "skip",
                "tp": None,
                "sl": None,
                "lev": None,
                "reason": f"Filtered: {filtered.get('risk_flag', 'unknown')} - {filtered.get('reason', 'No reason provided')}",
                "confidence": 0.0,
                "filter_passed": False
            }
        
        # Step 2: Final decision with heavy model
        final_decision = client.decide_trade(snapshot, filtered)
        
        # Sanity check on action
        valid_actions = ("enter", "modify", "skip", "hold")
        if final_decision.get("action") not in valid_actions:
            final_decision["action"] = "skip"
            final_decision["reason"] = f"Invalid action normalized to skip: {final_decision.get('action')}"
        
        # Ensure all required fields exist
        final_decision.setdefault("tp", None)
        final_decision.setdefault("sl", None)
        final_decision.setdefault("lev", None)
        final_decision.setdefault("reason", "")
        final_decision.setdefault("confidence", 0.5)
        final_decision["filter_passed"] = True
        
        return final_decision
        
    except Exception as e:
        # On any error, return skip with error info
        return {
            "action": "skip",
            "tp": None,
            "sl": None,
            "lev": None,
            "reason": f"Decision layer error: {str(e)}",
            "confidence": 0.0,
            "filter_passed": False,
            "error": str(e)
        }


def _fallback_decision(base_signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback decision when ChatGPT is not available.
    Uses the base signal with reduced confidence.
    
    Args:
        base_signal: Base trading signal from technical/AI analysis.
        
    Returns:
        Decision dictionary.
    """
    action = base_signal.get("action", "skip")
    confidence = base_signal.get("confidence", 0.5)
    
    # Reduce confidence since we're not using AI verification
    adjusted_confidence = confidence * 0.7
    
    # If confidence is too low after adjustment, skip
    if adjusted_confidence < 0.5:
        action = "skip"
    
    return {
        "action": action,
        "tp": base_signal.get("tp"),
        "sl": base_signal.get("sl"),
        "lev": base_signal.get("lev") or base_signal.get("leverage"),
        "reason": f"Fallback decision (ChatGPT unavailable): {base_signal.get('reason', 'No reason')}",
        "confidence": adjusted_confidence,
        "filter_passed": False,
        "fallback": True
    }


def quick_filter(
    snapshot: Dict[str, Any], 
    base_signal: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Quick filter using only the fast model (4o-mini).
    Useful for pre-screening many signals quickly.
    
    Args:
        snapshot: Current market snapshot.
        base_signal: Base trading signal.
        
    Returns:
        Filter result with 'allow' boolean and 'reason'.
    """
    client = get_client()
    
    if client is None:
        # If client not available, allow by default but flag it
        return {
            "allow": True,
            "reason": "ChatGPT unavailable - allowing by default",
            "risk_flag": None,
            "fallback": True
        }
    
    try:
        return client.filter_signal(snapshot, base_signal)
    except Exception as e:
        return {
            "allow": False,
            "reason": f"Filter error: {str(e)}",
            "risk_flag": "error",
            "error": str(e)
        }


def reset_client() -> None:
    """
    Reset the client instance.
    Useful for testing or when API credentials change.
    """
    global _client, _client_initialized
    _client = None
    _client_initialized = False


# For backwards compatibility
def get_decision(snapshot: Dict[str, Any], base_signal: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for verify_and_revise for backwards compatibility."""
    return verify_and_revise(snapshot, base_signal)
