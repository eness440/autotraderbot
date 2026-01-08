"""
whale_alert_provider.py
-----------------------

Large on-chain transfers, colloquially known as whale alerts, can
significantly impact market sentiment.  This module defines a simple
interface for retrieving recent whale alerts from on-chain data services.

The ``fetch_whale_alerts`` function is designed to query an external
service (such as Whale Alert or CryptoQuant) using an API key provided
via environment variables.  It returns a list of alerts, where each
alert is represented as a dictionary containing at least the transfer
amount, the token symbol and a timestamp.

Example usage::

    from whale_alert_provider import fetch_whale_alerts
    alerts = fetch_whale_alerts(token="BTC", min_value_usd=500000)
    for alert in alerts:
        print(alert["timestamp"], alert["amount"], alert["symbol"])

Note:  Because third-party services may impose rate limits, consider
caching results with ``cache_manager.file_cache`` if you call this
function frequently.
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict, Any, Optional, Tuple

# Optional dependencies
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

# Import retry and caching decorators.  If unavailable they fall back to
# identity functions so that this module remains functional in minimal
# environments.
try:
    from retry_utils import retry  # type: ignore
except Exception:
    def retry(*args, **kwargs):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator
try:
    from cache_manager import file_cache  # type: ignore
except Exception:
    def file_cache(cache_name: str, ttl: int = 3600):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token price caching and fallbacks

# In-memory cache to avoid repeatedly querying price APIs.  This is scoped to
# the module and not persisted across runs.  Keys are token symbols in upper
# case and values are floats (USD price).  The cache is refreshed on each
# call to ``_get_token_price`` if expired or missing.
_token_price_cache: Dict[str, float] = {}

# Fallback USD prices for commonly monitored tokens.  These are used when
# price retrieval fails or an API key is unavailable.  Adjust values as
# appropriate for your deployment environment.
_token_price_fallbacks: Dict[str, float] = {
    "ETH": 2000.0,
    "BNB": 250.0,
    "MATIC": 1.0,
    "ARB": 1.0,
}

def _get_token_price(token: str) -> float:
    """Return the current USD price for a given token symbol.

    This helper first checks an in-memory cache, then attempts to fetch
    a fresh quote from CoinMarketCap using the ``COINMARKETCAP_API_KEY``
    or ``CMC_API_KEY`` environment variables.  If the API call fails or
    the key is missing, a fallback constant from ``_token_price_fallbacks``
    is returned.

    Parameters
    ----------
    token : str
        Token symbol (e.g. "ETH", "BNB").

    Returns
    -------
    float
        Latest token price in USD, or a fallback value.
    """
    sym = token.upper()
    # Use cached value if available
    if sym in _token_price_cache:
        return _token_price_cache[sym]
    # Attempt to query CoinMarketCap if API key present
    api_key = os.getenv("COINMARKETCAP_API_KEY") or os.getenv("CMC_API_KEY")
    if api_key and requests is not None:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        params = {"symbol": sym}
        headers = {"X-CMC_PRO_API_KEY": api_key}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            price = (
                data.get("data", {}).get(sym, {}).get("quote", {}).get("USD", {}).get("price")
            )
            if price is not None:
                _token_price_cache[sym] = float(price)
                return _token_price_cache[sym]
        except Exception:
            pass
    # Use fallback if API unavailable
    fallback = _token_price_fallbacks.get(sym)
    if fallback is not None:
        _token_price_cache[sym] = fallback
        return fallback
    # Default fallback: 1.0 USD
    _token_price_cache[sym] = 1.0
    return 1.0

@file_cache("whale_alerts_cache.json", ttl=600)
@retry()
def fetch_whale_alerts(token: str = "BTC", min_value_usd: float = 1000000.0, lookback_hours: int = 1) -> List[Dict[str, Any]]:
    """Retrieve recent whale alerts across one or more chains.

    This function aggregates large on-chain transfers from multiple
    blockchains defined by the ``WHALE_CHAINS`` environment variable.
    Supported chains include ``ETH``, ``BSC``, ``MATIC`` and ``ARB``.
    Watchlist addresses are read from the ``WHALE_WATCHLIST`` env var.

    Parameters
    ----------
    token : str
        Unused; retained for backwards compatibility.
    min_value_usd : float
        Minimum USD value for transfers to be reported.
    lookback_hours : int
        Maximum age (in hours) for transactions to consider.

    Returns
    -------
    list of dict
        List of events, each containing ``timestamp``, ``from``, ``to``,
        ``value_token``, ``value_usd``, ``token``, ``hash`` and ``chain``.
    """
    # Optional official Whale Alert API (not available without subscription)
    api_key = os.getenv("WHALE_ALERT_API_KEY")
    if api_key:
        logger.debug("WHALE_ALERT_API_KEY provided but integration is not implemented.")
        # You could integrate the official Whale Alert service here
        return []
    # Determine chains to monitor (comma-separated list)
    chains_env = os.getenv("WHALE_CHAINS", "ETH").strip()
    chains: List[str] = [c.strip().upper() for c in chains_env.split(",") if c.strip()]
    if not chains:
        chains = ["ETH"]
    # Watchlist addresses
    watchlist = os.getenv("WHALE_WATCHLIST", "")
    addresses = [a.strip() for a in watchlist.split(",") if a.strip()]
    if not addresses:
        logger.debug("No watchlist addresses defined; whale alerts disabled")
        return []
    all_events: List[Dict[str, Any]] = []
    for chain_id in chains:
        try:
            events = fetch_large_transfers(
                addresses=addresses,
                min_value_usd=min_value_usd,
                chain=chain_id,
                lookback_hours=lookback_hours,
            )
            if events:
                all_events.extend(events)
        except Exception as exc:
            logger.warning("Failed to fetch whale alerts for chain %s: %s", chain_id, exc)
            continue
    # Sort events by USD value descending
    all_events.sort(key=lambda e: e.get("value_usd", 0), reverse=True)
    return all_events


# ---------------------------------------------------------------------------
# Custom whale transfer tracker using Etherscan

@file_cache("eth_price_cmc.json", ttl=300)
@retry()
def _get_eth_price_cmc() -> Optional[float]:
    """Return the latest ETH price in USD using the CoinMarketCap API.

    If ``COINMARKETCAP_API_KEY`` is set in the environment, this helper
    fetches a fresh quote for ETH from the CMC API.  When the API key is
    missing or an error occurs, ``None`` is returned.  Note that free
    tiers of CMC limit the number of requests per minute.  Consider
    caching this value via ``file_cache`` when integrating into your bot.

    Returns
    -------
    float or None
        Latest ETH price in USD, or None on failure.
    """
    api_key = os.getenv("COINMARKETCAP_API_KEY") or os.getenv("CMC_API_KEY")
    if not api_key or requests is None:
        return None
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    params = {"symbol": "ETH"}
    headers = {"X-CMC_PRO_API_KEY": api_key}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        price = (
            data
            .get("data", {})
            .get("ETH", {})
            .get("quote", {})
            .get("USD", {})
            .get("price")
        )
        return float(price) if price is not None else None
    except Exception:
        return None


def fetch_large_transfers(
    addresses: List[str],
    min_value_usd: float = 100000.0,
    chain: str = "ETH",
    lookback_hours: int = 1,
    limit_per_address: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch large on‑chain transfers for a list of addresses.

    This helper supports multiple blockchains using explorers similar to
    Etherscan.  Supported chains are ``ETH`` (Ethereum), ``BSC`` (Binance
    Smart Chain), ``MATIC`` (Polygon) and ``ARB`` (Arbitrum).  For each
    chain, it calls the respective explorer's public API to retrieve
    recent transactions, converts the token value into USD using
    ``_get_token_price`` and filters transfers below ``min_value_usd``.

    Parameters
    ----------
    addresses : list of str
        Addresses to monitor.  Checks both inbound and outbound transfers.
    min_value_usd : float
        Minimum USD value for a transfer to be considered.
    chain : str
        Chain identifier (case‑insensitive).  Supported values: ETH, BSC,
        MATIC, ARB.  Unknown values default to ETH.
    lookback_hours : int
        Maximum age of transactions to consider, in hours.
    limit_per_address : int
        Number of recent transactions to retrieve for each address.

    Returns
    -------
    list of dict
        List of events.  Each event includes ``timestamp``, ``from``, ``to``,
        ``value_token``, ``value_usd``, ``token``, ``hash`` and ``chain``.
    """
    if requests is None:
        logger.debug("requests library unavailable; cannot fetch whale transfers")
        return []
    chain_id = (chain or "ETH").strip().upper()
    # Map chain identifiers to API base URL, API key environment variable and
    # token symbol.  These explorers follow Etherscan's API conventions.
    chain_map = {
        "ETH": {
            "base_url": "https://api.etherscan.io/api",
            "api_key_var": "ETHERSCAN_API_KEY",
            "token": "ETH",
            "decimals": 18,
        },
        "BSC": {
            "base_url": "https://api.bscscan.com/api",
            "api_key_var": "BSCSCAN_API_KEY",
            "token": "BNB",
            "decimals": 18,
        },
        "MATIC": {
            "base_url": "https://api.polygonscan.com/api",
            "api_key_var": "POLYGONSCAN_API_KEY",
            "token": "MATIC",
            "decimals": 18,
        },
        "ARB": {
            "base_url": "https://api.arbiscan.io/api",
            "api_key_var": "ARBISCAN_API_KEY",
            "token": "ARB",
            "decimals": 18,
        },
    }
    info = chain_map.get(chain_id, chain_map["ETH"])
    api_key = os.getenv(info["api_key_var"], "").strip()
    if not api_key:
        # Fallback to ETHERSCAN_API_KEY for all chains if chain‑specific key missing
        api_key = os.getenv("ETHERSCAN_API_KEY", "").strip()
    if not api_key:
        logger.debug("API key missing for chain %s; cannot fetch transfers", chain_id)
        return []
    token_symbol = info["token"]
    decimals = info["decimals"]
    base_url = info["base_url"]
    # Fetch token price in USD using helper
    token_price = _get_token_price(token_symbol)
    now_ts = int(__import__("time").time())
    max_age = lookback_hours * 3600
    events: List[Dict[str, Any]] = []
    for addr in addresses:
        addr_lower = addr.lower()
        try:
            params = {
                "module": "account",
                "action": "txlist",
                "address": addr_lower,
                "startblock": 0,
                "endblock": 99999999,
                "page": 1,
                "offset": limit_per_address,
                "sort": "desc",
                "apikey": api_key,
            }
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            txs = resp.json().get("result", [])
            for tx in txs:
                try:
                    # Only consider transfers with value > 0
                    value_raw = int(tx.get("value", "0"))
                    if value_raw <= 0:
                        continue
                    # Convert to token units
                    value_token = value_raw / float(10 ** decimals)
                    value_usd = value_token * token_price
                    if value_usd < min_value_usd:
                        continue
                    ts = int(tx.get("timeStamp", "0"))
                    if now_ts - ts > max_age:
                        break
                    events.append({
                        "timestamp": ts,
                        "from": tx.get("from"),
                        "to": tx.get("to"),
                        "value_token": float(value_token),
                        "value_usd": float(value_usd),
                        "token": token_symbol,
                        "hash": tx.get("hash"),
                        "chain": chain_id,
                    })
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("%s scan fetch error for %s: %s", chain_id, addr_lower, exc)
            continue
    events.sort(key=lambda e: e.get("value_usd", 0), reverse=True)
    return events