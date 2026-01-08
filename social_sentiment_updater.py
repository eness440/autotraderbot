"""
social_sentiment_updater.py
--------------------------------
This module provides a simple service for fetching social sentiment
information from external sources like Twitter, Reddit and news APIs and
writing the aggregated scores to `metrics/social_sentiment.json`.  It is
intended to run periodically (e.g. via cron or a scheduled task).  The
actual API integration points are left as placeholders because API keys
and network connectivity are environment‑specific.  To enable live
sentiment updates, fill in the `fetch_*_sentiment` functions with
real API calls using your own credentials.

Example usage:

    python3 social_sentiment_updater.py

Environment variables used (if implemented):
  TWITTER_BEARER_TOKEN:  Twitter API bearer token for v2 search endpoint
  REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT: credentials for
      Reddit API or alternative aggregator
  NEWS_API_KEY: API key for a news sentiment provider (e.g. NewsAPI.org)

The resulting JSON file will contain keys 'tweet_sentiment',
'reddit_sentiment' and 'news_sentiment' with values in the range [-1, 1].

Note: In this repository, the sentiment functions return neutral (0.0)
values by default to avoid external API calls.  Replace the TODO
sections with real implementations to use live sentiment.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import requests
import logging

logger = logging.getLogger(__name__)

# Load .env variables at import time to populate API keys.  Use
# find_dotenv() to search for the .env file in parent directories.  If
# python-dotenv is not available, ignore the error.
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv())
except Exception:
    pass

# Import retry decorator from our utilities.  This decorator retries a
# function call with exponential backoff in case of transient errors such
# as network timeouts.  If ``retry_utils.py`` is modified or missing,
# import failure will silently disable retries.
try:
    from retry_utils import retry
except Exception:  # pragma: no cover - safe fallback when retry_utils is absent
    def retry(exceptions, tries=3, base_delay=0.5, max_delay=5.0):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

# Helper function for naive sentiment analysis.  We perform a simple
# lexicon‑based score by counting occurrences of predefined positive
# and negative words.  The resulting score is normalised to the range
# ``[-1, 1]``.  This is intentionally simplistic but avoids pulling in
# heavy NLP libraries like TextBlob or spaCy.
POS_WORDS = set([
    "bull", "bullish", "buy", "moon", "pump", "green", "up", "rise", "rally", "profit",
    "gain", "positive", "strong"
])
NEG_WORDS = set([
    "bear", "bearish", "sell", "dump", "red", "down", "fall", "drop", "loss",
    "negative", "weak", "fear"
])


def _aggregate_sentiment(texts: List[str]) -> float:
    """
    Compute an aggregate sentiment score from a list of strings.  Each
    string is tokenised on whitespace and punctuation.  Positive and
    negative words are counted using the lexicon defined above.  The
    final score is the difference between positive and negative word
    counts divided by the total number of sentiment words.  If no
    sentiment words are found, returns 0.0 (neutral).

    Args:
        texts: A list of sentences or documents to analyse.

    Returns:
        A float in the range [-1, 1].  Positive values indicate
        bullish/buying sentiment while negative values indicate
        bearish/selling sentiment.
    """
    pos_count = 0
    neg_count = 0
    for text in texts:
        # Lowercase and split on non‑alphabetic boundaries
        tokens = [tok.lower() for tok in text.split()]
        for tok in tokens:
            # Strip common punctuation
            tok = tok.strip(".,!?:;\"'()[]{}<>\n\r")
            if not tok:
                continue
            if tok in POS_WORDS:
                pos_count += 1
            elif tok in NEG_WORDS:
                neg_count += 1
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    score = (pos_count - neg_count) / total
    # Ensure the score is within [-1, 1]
    if score < -1.0:
        score = -1.0
    if score > 1.0:
        score = 1.0
    return float(score)

# ---------------------------------------------------------------------------
# Extended per‑symbol sentiment updater
#
# In many trading strategies it is beneficial to compute sentiment scores on
# a per‑asset basis rather than a single aggregated value.  The function
# below uses available API keys (NewsAPI, LunarCrush and Fear & Greed
# Index) to fetch sentiment for each provided symbol.  The NewsAPI
# endpoint is queried for recent articles mentioning the coin; a simple
# lexicon is used to score the text.  LunarCrush provides a ``galaxy
# score`` which is normalised to the range [‑1, 1].  The Fear & Greed
# Index (global crypto sentiment) is also included as a minor factor.
# These components are combined using weighted averages to produce a
# sentiment value in [‑1, 1] for each symbol.  The resulting mapping is
# written to ``metrics/social_sentiment.json`` and returned to the caller.

from typing import Iterable

def update_social_sentiment_for_symbols(symbols: Iterable[str], weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Compute sentiment scores for multiple symbols and persist to disk.

    Args:
        symbols: Iterable of symbols in trading pair format (e.g. "BTC/USDT").
        weights: Optional dict specifying weights for the components
            ``news``, ``lunar`` and ``fng``.  Defaults to
            ``{"news": 0.6, "lunar": 0.3, "fng": 0.1}``.

    Returns:
        A dictionary mapping each symbol to its sentiment score in the
        range [‑1, 1].  If no sentiment could be computed for a symbol,
        it will default to 0.0.
    """
    # Default weights
    if weights is None:
        weights = {"news": 0.6, "lunar": 0.3, "fng": 0.1}
    # Ensure weights sum to 1.0
    try:
        total_w = sum(float(w) for w in weights.values())
        if total_w > 0:
            weights = {k: float(v) / total_w for k, v in weights.items()}
    except Exception:
        weights = {"news": 0.6, "lunar": 0.3, "fng": 0.1}
    sentiments: Dict[str, float] = {}
    # Fetch the Fear & Greed Index once
    fng_score: float = 0.0
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=4)
        data = resp.json()
        val_str = data.get("data", [{}])[0].get("value")
        if isinstance(val_str, str):
            fng_val = float(val_str)
            # Map 0–100 to [‑1,1]
            fng_score = max(-1.0, min(1.0, (fng_val - 50.0) / 50.0))
    except Exception:
        fng_score = 0.0
    # Prepare API keys
    news_key = os.getenv("NEWS_API_KEY")
    lunar_key = os.getenv("LUNARCRUSH_API_KEY") or os.getenv("LUNARCRUSH_API")
    # Process each symbol
    for sym in symbols:
        # Normalise symbol for API queries (remove slash and uppercase for CC)
        sym_clean = sym.replace("/", "").upper()
        # Compute news sentiment
        news_sent: float | None = None
        if news_key:
            try:
                url = "https://newsapi.org/v2/everything"
                # Query the coin symbol (e.g. BTC or ETH) plus crypto keyword to focus on cryptocurrency context
                params = {
                    "q": f"{sym_clean} cryptocurrency",
                    "language": "en",
                    "pageSize": 25,
                    "apiKey": news_key,
                }
                r = requests.get(url, params=params, timeout=6)
                r.raise_for_status()
                articles = r.json().get("articles", [])
                texts: List[str] = []
                for art in articles:
                    if not isinstance(art, dict):
                        continue
                    title = art.get("title") or ""
                    desc = art.get("description") or ""
                    text = f"{title} {desc}".strip()
                    if text:
                        texts.append(text)
                if texts:
                    news_sent = _aggregate_sentiment(texts)
            except Exception as e:
                logger.warning("News sentiment fetch failed for %s: %s", sym_clean, e)
                news_sent = None
        # Compute LunarCrush sentiment via galaxy score
        lunar_sent: float | None = None
        if lunar_key:
            try:
                # LunarCrush v4 API: galaxy score for a specific asset is available via
                # ``/public/coins/{symbol}/v1``.  We build the URL dynamically and
                # authenticate via Bearer token.  See docs for details:
                # https://lunarcrush.com/developers/api/public/coins/coin/v1
                url = f"https://lunarcrush.com/api4/public/coins/{sym_clean}/v1"
                headers = {"Authorization": f"Bearer {lunar_key}"}
                resp_lc = requests.get(url, headers=headers, timeout=6)
                # On payment required (402), forbidden (403), not found (404), or rate limiting (429)
                # we treat as missing sentiment.  These codes indicate the plan does not
                # allow this endpoint or that we are being throttled.
                if resp_lc.status_code in (402, 403, 404, 429):
                    lunar_sent = None
                else:
                    resp_lc.raise_for_status()
                    js = resp_lc.json()
                    data_obj = js.get("data") or js
                    galaxy = None
                    if isinstance(data_obj, list) and data_obj:
                        galaxy = data_obj[0].get("galaxy_score")
                    elif isinstance(data_obj, dict):
                        galaxy = data_obj.get("galaxy_score")
                    if isinstance(galaxy, (int, float)):
                        # Normalise Galaxy Score (0–100) to [‑1, 1]
                        g_val = max(0.0, min(100.0, float(galaxy)))
                        lunar_sent = (g_val - 50.0) / 50.0
            except Exception as e:
                # Sanitize the exception message to avoid leaking the API key.  If the
                # LunarCrush key appears in the error, replace it with a placeholder.
                exc_str = str(e)
                if lunar_key:
                    exc_str = exc_str.replace(lunar_key, "[REDACTED]")
                logger.warning("LunarCrush sentiment fetch failed for %s: %s", sym_clean, exc_str)
                lunar_sent = None
        # Combine components using weights.  If a component is missing, skip its weight and re‑normalise
        parts: List[float] = []
        part_weights: List[float] = []
        for k, val in [("news", news_sent), ("lunar", lunar_sent), ("fng", fng_score)]:
            try:
                w = weights.get(k, 0.0)
            except Exception:
                w = 0.0
            if val is None:
                continue
            parts.append(float(val) * w)
            part_weights.append(w)
        if parts and part_weights:
            # Normalise weights to sum to 1 for available parts
            total_w = sum(part_weights)
            if total_w > 0:
                score = sum(parts) / total_w
            else:
                score = sum(parts)
        else:
            score = 0.0
        # Ensure score is within [‑1, 1]
        score = max(-1.0, min(1.0, float(score)))
        sentiments[sym] = score
    # Persist sentiment scores to metrics/social_sentiment.json
    try:
        metrics_dir = Path(__file__).resolve().parent / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_path = metrics_dir / "social_sentiment.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(sentiments, fh, indent=2)
        logger.info("Per‑symbol sentiment updated for %d symbols", len(sentiments))
    except Exception as e:
        logger.error("Failed to write per‑symbol social sentiment: %s", e)
    return sentiments


from cache_manager import file_cache
logger = logging.getLogger(__name__)

@file_cache("twitter_sentiment.json", ttl=300)
def fetch_twitter_sentiment(query: str = "crypto") -> Optional[float]:
    """Fetch sentiment from Twitter or fallback to CoinMarketCap community data.

    If a bearer token is provided via the ``TWITTER_BEARER_TOKEN`` environment
    variable, the function queries the Twitter API v2 recent search endpoint
    and performs a very simple sentiment analysis based on word counts.
    
    When no Twitter token is present, falls back to CoinMarketCap community
    trending data using ``COINMARKETCAP_API_KEY``. This provides social
    momentum signals as an alternative to Twitter sentiment.

    Args:
        query: The search term to query tweets for (e.g. "bitcoin").

    Returns:
        A float in the range ``[-1, 1]`` on success, or ``None`` when
        sentiment could not be determined.
    """
    token = os.getenv("TWITTER_BEARER_TOKEN")
    
    # Primary: Twitter API
    if token:
        headers = {"Authorization": f"Bearer {token}"}
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            "query": query,
            "max_results": 30,
            "tweet.fields": "text,lang"
        }

        @retry((Exception,), tries=3, base_delay=1.0, max_delay=4.0)
        def _query() -> List[str]:
            resp = requests.get(url, params=params, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            texts: List[str] = []
            for item in data.get("data", []):
                text = item.get("text")
                if isinstance(text, str) and len(text) > 0:
                    texts.append(text)
            return texts

        try:
            texts = _query()
            if texts:
                return _aggregate_sentiment(texts)
        except Exception as e:
            logger.warning("Twitter API call failed: %s", e)
    
    # Fallback: CoinMarketCap Community Trending
    cmc_key = os.getenv("COINMARKETCAP_API_KEY")
    if cmc_key:
        try:
            # CoinMarketCap trending API - returns most searched/visited coins
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/trending/most-visited"
            headers = {"X-CMC_PRO_API_KEY": cmc_key, "Accept": "application/json"}
            resp = requests.get(url, headers=headers, timeout=6)
            
            if resp.status_code in (402, 403, 429):
                logger.warning("CoinMarketCap API rate limited or forbidden")
            else:
                resp.raise_for_status()
                data = resp.json().get("data", [])
                
                # Calculate sentiment based on trending coins' price changes
                if data:
                    positive_moves = 0
                    negative_moves = 0
                    for coin in data[:20]:  # Top 20 trending
                        quote = coin.get("quote", {}).get("USD", {})
                        pct_24h = quote.get("percent_change_24h", 0) or 0
                        if pct_24h > 0:
                            positive_moves += 1
                        elif pct_24h < 0:
                            negative_moves += 1
                    
                    total = positive_moves + negative_moves
                    if total > 0:
                        # Convert to [-1, 1] range
                        sentiment = (positive_moves - negative_moves) / total
                        logger.info("CoinMarketCap trending sentiment: %.2f", sentiment)
                        return float(max(-1.0, min(1.0, sentiment)))
        except Exception as e:
            logger.warning("CoinMarketCap trending fetch failed: %s", e)
    
    # Secondary Fallback: LunarCrush social volume for general crypto
    lunar_key = os.getenv("LUNARCRUSH_API_KEY") or os.getenv("LUNARCRUSH_API")
    if lunar_key:
        try:
            # Get BTC social sentiment as proxy for general crypto sentiment
            url = "https://lunarcrush.com/api4/public/coins/BTC/v1"
            headers = {"Authorization": f"Bearer {lunar_key}"}
            resp = requests.get(url, headers=headers, timeout=5)
            
            if resp.status_code not in (402, 403, 404, 429):
                resp.raise_for_status()
                js = resp.json()
                data_obj = js.get("data") or js
                
                # Get social volume and sentiment
                social_score = None
                if isinstance(data_obj, dict):
                    social_score = data_obj.get("social_score") or data_obj.get("galaxy_score")
                elif isinstance(data_obj, list) and data_obj:
                    social_score = data_obj[0].get("social_score") or data_obj[0].get("galaxy_score")
                
                if social_score is not None:
                    # Normalize 0-100 to [-1, 1]
                    gs = max(0.0, min(100.0, float(social_score)))
                    sentiment = (gs / 50.0) - 1.0
                    logger.info("LunarCrush social fallback sentiment: %.2f", sentiment)
                    return float(sentiment)
        except Exception as e:
            exc_str = str(e)
            if lunar_key:
                exc_str = exc_str.replace(lunar_key, "[REDACTED]")
            logger.warning("LunarCrush social fallback failed: %s", exc_str)
    
    return None


@file_cache("reddit_sentiment.json", ttl=300)
def fetch_reddit_sentiment(subreddit: str = "cryptocurrency") -> Optional[float]:
    """
    Fetch sentiment from Reddit for a given subreddit.

    When valid Reddit API credentials (``REDDIT_CLIENT_ID``, ``REDDIT_SECRET``
    and ``REDDIT_USER_AGENT``) are available, this function queries the
    official Reddit API via OAuth to obtain recent posts from the specified
    subreddit.  If credentials are missing it falls back to the public
    JSON feed.  If that also fails, it uses CoinGecko trending as a final
    fallback. Retrieved titles are scored using a naive lexicon‑based
    sentiment analyser.

    Args:
        subreddit: Name of the subreddit to analyse.

    Returns:
        A float in the range ``[-1, 1]`` on success, or ``None`` if no
        data is available.
    """
    # Attempt to fetch the latest posts from the subreddit using the
    # official Reddit API if credentials are set.  Otherwise, fall back
    # to the public JSON endpoint.
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "social-sentiment-agent/0.1")
    texts: List[str] = []
    
    # Method 1: Reddit API with OAuth
    if client_id and client_secret:
        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
        headers = {"User-Agent": user_agent}
        data = {"grant_type": "client_credentials"}
        try:
            # Get OAuth token
            resp = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers, timeout=5)
            resp.raise_for_status()
            token = resp.json().get("access_token")
        except Exception as e:
            logger.warning(f"Reddit auth failed: {e}")
            token = None
        if token:
            headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}
            url = f"https://oauth.reddit.com/r/{subreddit}/new"
            params = {"limit": 30}
            @retry((Exception,), tries=3, base_delay=1.0, max_delay=4.0)
            def _query_oauth() -> List[str]:
                r = requests.get(url, headers=headers, params=params, timeout=5)
                r.raise_for_status()
                data = r.json()
                posts = data.get("data", {}).get("children", [])
                res: List[str] = []
                for p in posts:
                    body = p.get("data", {}).get("title") or ""
                    res.append(body)
                return res
            try:
                texts = _query_oauth()
            except Exception as e:
                logger.warning(f"Reddit API fetch failed: {e}")
    
    # Method 2: Public Reddit JSON feed (no auth required)
    if not texts:
        url = f"https://www.reddit.com/r/{subreddit}/new.json"
        params = {"limit": 30}
        headers = {"User-Agent": user_agent}
        @retry((Exception,), tries=3, base_delay=1.0, max_delay=4.0)
        def _query_json() -> List[str]:
            r = requests.get(url, params=params, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            posts = data.get("data", {}).get("children", [])
            res: List[str] = []
            for p in posts:
                body = p.get("data", {}).get("title") or ""
                res.append(body)
            return res
        try:
            texts = _query_json()
        except Exception as e:
            logger.warning(f"Reddit JSON fetch failed: {e}")
            texts = []
    
    # Method 3: Fallback to CoinGecko trending (free, no API key)
    if not texts:
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                coins = data.get("coins", [])
                
                # Calculate sentiment from trending coin price changes
                positive = 0
                negative = 0
                for coin_data in coins[:10]:
                    item = coin_data.get("item", {})
                    # Get price change if available
                    price_btc = item.get("price_btc", 0)
                    score = item.get("score", 0)
                    # Higher score = more trending = bullish signal
                    if score is not None and score <= 3:  # Top 3 trending
                        positive += 1
                    elif score is not None and score > 5:
                        negative += 1
                
                if positive + negative > 0:
                    sentiment = (positive - negative) / (positive + negative)
                    logger.info("CoinGecko trending fallback sentiment: %.2f", sentiment)
                    return float(max(-1.0, min(1.0, sentiment)))
        except Exception as e:
            logger.warning(f"CoinGecko trending fallback failed: {e}")
    
    # Method 4: Final fallback using LunarCrush social dominance
    if not texts:
        lunar_key = os.getenv("LUNARCRUSH_API_KEY") or os.getenv("LUNARCRUSH_API")
        if lunar_key:
            try:
                url = "https://lunarcrush.com/api4/public/coins/list/v1"
                headers = {"Authorization": f"Bearer {lunar_key}"}
                params = {"sort": "social_dominance", "limit": 10}
                resp = requests.get(url, headers=headers, params=params, timeout=5)
                
                if resp.status_code not in (402, 403, 404, 429):
                    resp.raise_for_status()
                    js = resp.json()
                    data_list = js.get("data", [])
                    
                    if data_list:
                        # Average sentiment from top social dominance coins
                        sentiments = []
                        for coin in data_list[:5]:
                            gs = coin.get("galaxy_score") or coin.get("social_score")
                            if gs is not None:
                                normalized = (float(gs) / 50.0) - 1.0
                                sentiments.append(normalized)
                        
                        if sentiments:
                            avg_sent = sum(sentiments) / len(sentiments)
                            logger.info("LunarCrush social dominance fallback: %.2f", avg_sent)
                            return float(max(-1.0, min(1.0, avg_sent)))
            except Exception as e:
                exc_str = str(e)
                if lunar_key:
                    exc_str = exc_str.replace(lunar_key, "[REDACTED]")
                logger.warning(f"LunarCrush social dominance fallback failed: {exc_str}")
    
    if not texts:
        return None
    return _aggregate_sentiment(texts)


@file_cache("news_sentiment.json", ttl=600)
def fetch_news_sentiment(keyword: str = "crypto") -> Optional[float]:
    """
    Fetch sentiment from news articles using an external API or RSS feeds.

    When a ``NEWS_API_KEY`` is set in the environment, this function
    queries NewsAPI.org for recent articles matching the supplied
    keyword.  If the key is not set or the request fails, it falls
    back to the Google News RSS feed.  On success it returns a
    sentiment score in ``[-1, 1]`` based on the naive lexicon scorer.

    Importantly, this function returns ``None`` when no articles
    could be retrieved or an error occurs.  The calling code is
    expected to handle ``None`` values by skipping the news component
    altogether rather than defaulting to a dummy neutral sentiment.

    Args:
        keyword: Keyword to search news articles for (e.g. "bitcoin").

    Returns:
        A float in ``[-1, 1]`` on success, or ``None`` on failure/no
        data.
    """
    api_key = os.getenv("NEWS_API_KEY")
    texts: List[str] = []
    # Primary: NewsAPI if API key is available
    if api_key:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": keyword,
            "language": "en",
            "pageSize": 30,
            "apiKey": api_key,
            "sortBy": "relevancy",
        }
        headers: Dict[str, str] = {}

        @retry((Exception,), tries=3, base_delay=1.0, max_delay=4.0)
        def _query_newsapi() -> List[str]:
            r = requests.get(url, params=params, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            articles = data.get("articles", [])
            res: List[str] = []
            for article in articles:
                title = article.get("title") or ""
                desc = article.get("description") or ""
                res.append(f"{title}. {desc}")
            return res

        try:
            texts = _query_newsapi()
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
    # Fallback: Google News RSS
    if not texts:
        import xml.etree.ElementTree as ET
        feed_url = (
            f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
        )

        @retry((Exception,), tries=3, base_delay=1.0, max_delay=4.0)
        def _query_rss() -> List[str]:
            r = requests.get(feed_url, timeout=5)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            res: List[str] = []
            for item in root.iter("item"):
                title = item.findtext("title") or ""
                description = item.findtext("description") or ""
                res.append(f"{title}. {description}")
            return res[:30]

        try:
            texts = _query_rss()
        except Exception as e:
            logger.warning(f"RSS news fetch failed: {e}")
            texts = []

    if not texts:
        # No data available
        return None
    return _aggregate_sentiment(texts)

# ---------------------------------------------------------------------------
# Fear & Greed index helper

FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"

@file_cache("lunarcrush_sentiment.json", ttl=600)
@retry((Exception,), tries=3, base_delay=1.0, max_delay=4.0)
def fetch_lunarcrush_sentiment(symbol: str = "BTC") -> Optional[float]:
    """
    Fetch social sentiment from the LunarCrush API for a given symbol.

    The LunarCrush API provides the "Galaxy Score" and other metrics
    summarising community engagement and price performance for a
    cryptocurrency.  The free API tier can be accessed using the
    `LUNARCRUSH_API_KEY` environment variable.  If no key is set or
    the request fails, this function returns 0.0 (neutral sentiment).

    The Galaxy Score ranges from 0 to 100.  We map it to the range
    ``[-1, 1]`` by dividing by 50 and subtracting 1.  Thus 0 becomes
    -1 (bearish) and 100 becomes +1 (bullish).

    Args:
        symbol: A cryptocurrency ticker (e.g. "BTC" or "ETH").

    Returns:
        Normalised Galaxy Score in ``[-1, 1]``.
    """
    # Read the LunarCrush API key.  ``LUNARCRUSH_API_KEY`` is the
    # recommended name, but ``LUNARCRUSH_API`` is supported for
    # backwards‑compatibility with older configurations.
    key = os.getenv("LUNARCRUSH_API_KEY") or os.getenv("LUNARCRUSH_API")
    if not key:
        # No API key – cannot fetch data
        return None
    try:
        # Updated LunarCrush endpoint and authentication.  The Galaxy Score for a
        # specific asset is available via ``/public/coins/{symbol}/v1``.  The
        # endpoint returns an object containing market data and social metrics,
        # including ``galaxy_score``.  See API docs for details:
        # https://lunarcrush.com/developers/api/public/coins/coin/v1
        sym = symbol.upper()
        url = f"https://lunarcrush.com/api4/public/coins/{sym}/v1"
        headers = {"Authorization": f"Bearer {key}"}
        # Perform request with basic retry logic; if ``retry_utils`` wraps this
        # function, the decorator will handle exceptions.
        resp = requests.get(url, headers=headers, timeout=5)
        # If the API returns a non-success status that indicates payment required,
        # forbidden, not found, or rate limit exceeded, treat as missing sentiment.
        # 402 = Payment Required (subscription needed), 403 = Forbidden (invalid key),
        # 404 = Not Found (endpoint unavailable), 429 = Too Many Requests (rate limit).
        if resp.status_code in (402, 403, 404, 429):
            return None
        resp.raise_for_status()
        js = resp.json()
        # The API may return a list under ``data`` or directly a dict.  We
        # normalise to a dict representing one coin's data.
        if isinstance(js, dict):
            data_obj = js.get("data") or js
        else:
            return None
        # The galaxy_score may be under the ``data`` object or top-level
        galaxy_score = None
        if isinstance(data_obj, list) and data_obj:
            galaxy_score = data_obj[0].get("galaxy_score")
        elif isinstance(data_obj, dict):
            galaxy_score = data_obj.get("galaxy_score")
        if galaxy_score is None:
            return None
        try:
            gs = float(galaxy_score)
        except Exception:
            return None
        # The Galaxy Score is nominally 0–100.  For safety, clamp to [0, 100].
        gs = max(0.0, min(100.0, gs))
        # Map to [-1, 1]
        return (gs / 50.0) - 1.0
    except Exception as e:
        # Sanitize the exception message to avoid leaking the API key.  If the
        # LunarCrush key is present in the exception string, replace it with a
        # placeholder before logging.  Returning None signals missing data.
        exc_str = str(e)
        if key:
            exc_str = exc_str.replace(key, "[REDACTED]")
        logger.warning("LunarCrush sentiment fetch failed: %s", exc_str)
        return None

def fetch_fear_and_greed_index(timeout: float = 5.0) -> Optional[float]:
    """
    Fetch the current Crypto Fear & Greed Index from alternative.me and map
    the raw 0–100 value to a 0–1 range.  Returns 0.5 (neutral) on error.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        Normalised sentiment value in [0, 1] where lower values indicate fear
        and higher values indicate greed.
    """
    try:
        resp = requests.get(FNG_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        raw_val = float(data.get("data", [{}])[0].get("value"))
        raw_val = max(0.0, min(100.0, raw_val))
        return raw_val / 100.0
    except Exception as e:
        logger.warning(f"[sentiment_updater] Fear & Greed fetch failed: {e}")
        # Return None to signal missing data rather than a dummy neutral value
        return None

def combine_global_sentiment(news_score: float, fng_score: float,
                             w_news: float = 0.6, w_fng: float = 0.4) -> float:
    """
    Combine the news sentiment and the Fear & Greed index into a single
    global sentiment score.  This legacy helper exists for backward
    compatibility and is used when only news and F&G are available.  Newer
    callers should prefer the weighting logic in ``update_social_sentiment``
    which supports combining Twitter and Reddit sentiment as well.

    All inputs are clamped to their expected ranges and then mapped to
    ``[0,1]`` before applying the given weights.  The output is also clamped
    to ``[0,1]``.

    Args:
        news_score: Aggregate news sentiment in ``[-1, 1]``.  If outside this
            range values are clamped.
        fng_score: Fear & Greed index normalised to ``[0, 1]``.
        w_news: Weight for the news score.  Defaults to 0.6.
        w_fng: Weight for the fear & greed score.  Defaults to 0.4.

    Returns:
        Weighted combination in ``[0, 1]``.
    """
    try:
        n = float(news_score)
    except Exception:
        n = 0.0
    # Clamp news to [-1, 1] and normalise to [0,1]
    n = max(-1.0, min(1.0, n))
    news_norm = (n + 1.0) / 2.0
    try:
        f = float(fng_score)
    except Exception:
        f = 0.5
    # Clamp FNG to [0,1]
    f = max(0.0, min(1.0, f))
    total_w = max(1e-6, w_news + w_fng)
    combined = (news_norm * w_news + f * w_fng) / total_w
    # Clamp final result
    return max(0.0, min(1.0, combined))


def update_social_sentiment(
    tweet_query: str = "crypto",
    reddit_sub: str = "cryptocurrency",
    news_keyword: str = "crypto",
    out_file: Optional[Path] = None,
) -> dict:
    """
    Fetch sentiment values from Twitter, Reddit and news sources and
    write them into a JSON file within the `metrics/` directory.  If
    no output file is provided, defaults to metrics/social_sentiment.json.

    Args:
        tweet_query: Search query for Twitter sentiment.
        reddit_sub: Subreddit to analyze for Reddit sentiment.
        news_keyword: Keyword for news sentiment search.
        out_file: Optional path to the output JSON file.

    Returns:
        A dictionary with keys 'tweet_sentiment', 'reddit_sentiment',
        'news_sentiment'.  Values are in [-1, 1].
    """
    # Fetch individual sentiment components.  Each fetcher returns
    # either a sentiment value or ``None`` if data could not be
    # retrieved.  ``None`` values will be ignored when computing
    # the aggregate sentiment.
    tw = fetch_twitter_sentiment(tweet_query)
    rd = fetch_reddit_sentiment(reddit_sub)
    nw = fetch_news_sentiment(news_keyword)
    base_sym = news_keyword.split("/")[0] if isinstance(news_keyword, str) else "BTC"
    lc = fetch_lunarcrush_sentiment(base_sym)
    fng = fetch_fear_and_greed_index()

    # Weight coefficients for each sentiment source.  You can tune
    # these values to emphasise or de‑emphasise certain channels.
    weights = {
        "tw": 0.25,
        "rd": 0.15,
        "nw": 0.30,
        "lc": 0.15,
        "fng": 0.15,
    }

    # Helper to normalise a score from [-1,1] into [0,1].  If x is
    # ``None`` then no normalisation is performed and the caller should
    # skip weighting.
    def _norm(x: float) -> float:
        f = float(x)
        if f < -1.0:
            f = -1.0
        if f > 1.0:
            f = 1.0
        return (f + 1.0) / 2.0

    # Build lists of available normalised values and their weights
    norm_values: list[float] = []
    norm_weights: list[float] = []
    if tw is not None:
        norm_values.append(_norm(tw))
        norm_weights.append(weights["tw"])
    if rd is not None:
        norm_values.append(_norm(rd))
        norm_weights.append(weights["rd"])
    if nw is not None:
        norm_values.append(_norm(nw))
        norm_weights.append(weights["nw"])
    if lc is not None:
        # LunarCrush returns a value in [-1,1]
        norm_values.append(_norm(lc))
        norm_weights.append(weights["lc"])
    if fng is not None:
        # Fear & Greed index already in [0,1]; ensure numeric
        try:
            fng_norm = float(fng)
        except Exception:
            fng_norm = 0.5
        fng_norm = max(0.0, min(1.0, fng_norm))
        norm_values.append(fng_norm)
        norm_weights.append(weights["fng"])
    # Compute weighted average if there are available components
    if norm_weights:
        total_w = sum(norm_weights)
        global_sent = sum(v * w for v, w in zip(norm_values, norm_weights)) / total_w
        # Clamp to [0,1]
        global_sent = max(0.0, min(1.0, global_sent))
    else:
        # If nothing is available, default to neutral
        global_sent = 0.5

    data = {
        "tweet_sentiment": None if tw is None else float(tw),
        "reddit_sentiment": None if rd is None else float(rd),
        "news_sentiment": None if nw is None else float(nw),
        "lunarcrush_sentiment": None if lc is None else float(lc),
        # Preserve the raw FNG index; if missing return None
        "fng_value": None if fng is None else float(fng),
        "global_sentiment": float(global_sent),
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }

    target = out_file or Path("metrics/social_sentiment.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[sentiment-updater] Failed to write social sentiment file: {e}")
    return data


if __name__ == "__main__":
    update_social_sentiment()