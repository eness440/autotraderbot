"""
db_utils.py
-----------

Utilities for SQLite database integration.  These helpers create the
database schema and provide functions to insert trade, prediction and
metrics records.  The schema is intentionally simple for easy
inspection and portability.  SQLite is chosen as the default backend
for ease of distribution; for more advanced deployments consider
switching to PostgreSQL or another relational database.

To initialise the database and migrate existing JSON logs, run
``migrate_to_db.py`` located in the project root.  After migration you
can query the database using standard SQL tools or integrate it into
reporting dashboards.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Path to the SQLite file.  Resides in ``data/bot.db`` under the project
# root.  If the ``data`` directory does not exist it will be created by
# initialise_db().
DB_PATH = Path("data/bot.db")


def initialise_db(db_path: Optional[Path] = None) -> None:
    """
    Create the database file and tables if they do not already exist.

    Args:
        db_path: Optional override for the database file location.  If
            omitted, uses the module‑level ``DB_PATH``.
    """
    path = db_path or DB_PATH
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        c = conn.cursor()
        # Trades table
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                timestamp_open TEXT,
                timestamp_close TEXT,
                ai_score REAL,
                tech_score REAL,
                sent_score REAL,
                master_confidence REAL,
                leverage INTEGER,
                rl_score REAL,
                wallet_allocation_percent REAL,
                risk_usd REAL,
                tf TEXT,
                base_decision TEXT
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp_close ON trades(timestamp_close)"
        )
        # Predictions table
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                ai_confidence REAL,
                tech_score REAL,
                sent_score REAL,
                master_confidence REAL,
                provider TEXT
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)"
        )
        # Metrics table
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                api_calls INTEGER,
                api_errors INTEGER,
                api_retries INTEGER,
                total_latency REAL,
                realized_pnl REAL,
                unrealized_pnl REAL,
                exposure_usd REAL,
                expectancy REAL,
                trade_count INTEGER
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
        conn.commit()


def insert_trade(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    """
    Insert a trade record into the trades table.  Missing keys are
    accepted and default to NULL.

    Args:
        conn: SQLite connection.
        record: Dictionary representing a trade.
    """
    fields = [
        "symbol",
        "side",
        "entry_price",
        "exit_price",
        "size",
        "pnl_usd",
        "pnl_pct",
        "timestamp_open",
        "timestamp_close",
        "ai_score",
        "tech_score",
        "sent_score",
        "master_confidence",
        "leverage",
        "rl_score",
        "wallet_allocation_percent",
        "risk_usd",
        "tf",
        "base_decision",
    ]
    values = [record.get(k) for k in fields]
    conn.execute(
        f"INSERT INTO trades ({', '.join(fields)}) VALUES ({', '.join(['?'] * len(fields))})",
        values,
    )


def insert_prediction(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    """
    Insert a prediction record into the predictions table.

    Args:
        conn: SQLite connection.
        record: Dictionary representing a prediction.
    """
    fields = [
        "symbol",
        "timestamp",
        "ai_confidence",
        "tech_score",
        "sent_score",
        "master_confidence",
        "provider",
    ]
    values = [record.get(k) for k in fields]
    conn.execute(
        f"INSERT INTO predictions ({', '.join(fields)}) VALUES ({', '.join(['?'] * len(fields))})",
        values,
    )


def insert_metrics(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    """
    Insert a metrics record into the metrics table.

    Args:
        conn: SQLite connection.
        record: Dictionary representing metrics.
    """
    fields = [
        "timestamp",
        "api_calls",
        "api_errors",
        "api_retries",
        "total_latency",
        "realized_pnl",
        "unrealized_pnl",
        "exposure_usd",
        "expectancy",
        "trade_count",
    ]
    values = [record.get(k) for k in fields]
    conn.execute(
        f"INSERT INTO metrics ({', '.join(fields)}) VALUES ({', '.join(['?'] * len(fields))})",
        values,
    )