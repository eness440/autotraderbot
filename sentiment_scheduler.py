"""
sentiment_scheduler.py
---------------------------------
This script provides a simple scheduler for updating social sentiment
scores at regular intervals using the APScheduler library.  It uses
the ``update_social_sentiment`` function from
``social_sentiment_updater`` to fetch Twitter, Reddit and news
sentiment data and write it to ``metrics/social_sentiment.json``.

Configuration is driven by environment variables:

  SENTIMENT_UPDATE_INTERVAL (int): update frequency in minutes (default 30)
  SENTIMENT_QUERY (str): query term for Twitter and news sentiment (default "crypto")
  SENTIMENT_SUBREDDIT (str): subreddit for Reddit sentiment (default "cryptocurrency")

To run the scheduler, execute this script directly.  It will start a
blocking scheduler that runs until interrupted.  Ensure that
``apscheduler`` is installed in your Python environment.
"""

import os
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

from social_sentiment_updater import update_social_sentiment


def _update_job() -> None:
    """Wrapper to call update_social_sentiment with environment overrides."""
    tweet_query = os.getenv("SENTIMENT_QUERY", "crypto")
    reddit_sub = os.getenv("SENTIMENT_SUBREDDIT", "cryptocurrency")
    news_kw = tweet_query
    data = update_social_sentiment(tweet_query, reddit_sub, news_kw)
    print(
        f"[SentimentScheduler] {datetime.utcnow().isoformat()}Z updated sentiment: "
        f"{data}"
    )


def run_scheduler() -> None:
    """Create and start the blocking scheduler."""
    # Reduce the default update interval from 30 to 15 minutes to
    # better capture rapid shifts in market sentiment.  The interval
    # remains configurable via the SENTIMENT_UPDATE_INTERVAL environment
    # variable (value in minutes).  If unset, a 15‑minute default is used.
    interval_min = int(os.getenv("SENTIMENT_UPDATE_INTERVAL", "15"))
    scheduler = BlockingScheduler()
    scheduler.add_job(
        _update_job,
        'interval',
        minutes=interval_min,
        next_run_time=datetime.utcnow(),
    )
    print(
        f"[SentimentScheduler] starting: interval={interval_min} min, query={os.getenv('SENTIMENT_QUERY', 'crypto')}, subreddit={os.getenv('SENTIMENT_SUBREDDIT', 'cryptocurrency')}"
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("[SentimentScheduler] stopped.")


if __name__ == "__main__":
    run_scheduler()