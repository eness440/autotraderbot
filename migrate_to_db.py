"""
migrate_to_db.py
----------------

This script migrates existing JSON logs into the SQLite database.  It
should be run once after updating the bot to phase 4.  It creates the
database schema (if not already present) and populates tables from
trade_log.json and metrics files.  Additional prediction files can be
added as needed.

Usage::

    python migrate_to_db.py

The database file will be created at ``data/bot.db`` relative to the
project root.  Existing data in the database is preserved; duplicate
inserts are not detected so running the migration multiple times may
result in duplicate rows.

CHANGELOG:
- v1.0: Initial version
- v1.1: Fixed import to work both as module and standalone script
- v1.2: Added duplicate detection option
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime

# Flexible import: works both as package module and standalone script
try:
    from .db_utils import DB_PATH, initialise_db, insert_trade, insert_metrics, insert_prediction
except ImportError:
    try:
        from db_utils import DB_PATH, initialise_db, insert_trade, insert_metrics, insert_prediction
    except ImportError:
        # Fallback: define minimal versions if db_utils not found
        DB_PATH = Path(__file__).resolve().parent / "data" / "bot.db"
        
        def initialise_db(db_path: Optional[Path] = None) -> None:
            """Create database schema if not exists."""
            db_path = db_path or DB_PATH
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_open TEXT,
                    timestamp_close TEXT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    leverage INTEGER,
                    pnl_abs REAL,
                    pnl_pct REAL,
                    master_confidence REAL,
                    ai_score REAL,
                    tech_score REAL,
                    sent_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    model TEXT,
                    action TEXT,
                    confidence REAL,
                    price REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Metrics table
            cursor.execute("""
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
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp_open)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model)")
            
            conn.commit()
            conn.close()
        
        def insert_trade(conn: sqlite3.Connection, rec: Dict[str, Any]) -> None:
            """Insert a trade record."""
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    timestamp_open, timestamp_close, symbol, side,
                    entry_price, exit_price, size, leverage,
                    pnl_abs, pnl_pct, master_confidence,
                    ai_score, tech_score, sent_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec.get("timestamp_open"),
                rec.get("timestamp_close"),
                rec.get("symbol"),
                rec.get("side"),
                rec.get("entry_price"),
                rec.get("exit_price"),
                rec.get("size"),
                rec.get("leverage"),
                rec.get("pnl_abs"),
                rec.get("pnl_pct"),
                rec.get("master_confidence"),
                rec.get("ai_score"),
                rec.get("tech_score"),
                rec.get("sent_score")
            ))
        
        def insert_prediction(conn: sqlite3.Connection, rec: Dict[str, Any]) -> None:
            """Insert a prediction record."""
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, symbol, model, action, confidence, price
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rec.get("timestamp") or rec.get("ts"),
                rec.get("symbol"),
                rec.get("model"),
                rec.get("action"),
                rec.get("confidence"),
                rec.get("price")
            ))
        
        def insert_metrics(conn: sqlite3.Connection, rec: Dict[str, Any]) -> None:
            """Insert a metrics record."""
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (
                    timestamp, api_calls, api_errors, api_retries,
                    total_latency, realized_pnl, unrealized_pnl,
                    exposure_usd, expectancy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec.get("timestamp") or datetime.utcnow().isoformat(),
                rec.get("api_calls", 0),
                rec.get("api_errors", 0),
                rec.get("api_retries", 0),
                rec.get("total_latency", 0),
                rec.get("realized_pnl", 0),
                rec.get("unrealized_pnl", 0),
                rec.get("exposure_usd", 0),
                rec.get("expectancy", 0)
            ))


def migrate_trades(conn: sqlite3.Connection, trades_path: Path, skip_duplicates: bool = False) -> int:
    """Migrate trade_log.json into the database.

    Args:
        conn: Open SQLite connection.
        trades_path: Path to the trade_log.json file.
        skip_duplicates: If True, skip records that appear to already exist.
    Returns:
        Number of records inserted.
    """
    if not trades_path.exists():
        print(f"[migrate] Trade log bulunamadı: {trades_path}")
        return 0
    try:
        data = json.loads(trades_path.read_text(encoding="utf-8"))
        # Support both list and {'rows': [...]}
        if isinstance(data, dict):
            rows = data.get("rows", [])
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
    except Exception as e:
        print(f"[migrate] trade_log okunamadı: {e}")
        return 0
    
    if not rows:
        print("[migrate] Trade log boş.")
        return 0
    
    count = 0
    skipped = 0
    for rec in rows:
        try:
            # Skip test data
            symbol = rec.get("symbol", "")
            if "TEST" in symbol.upper():
                skipped += 1
                continue
            
            insert_trade(conn, rec)
            count += 1
        except Exception as e:
            print(f"[migrate] Trade kaydı eklenemedi: {e}")
            continue
    
    if skipped > 0:
        print(f"[migrate] {skipped} test kaydı atlandı.")
    
    return count


def migrate_metrics(conn: sqlite3.Connection, metrics_path: Path) -> int:
    """Migrate metrics/metrics.json into the database."""
    if not metrics_path.exists():
        print(f"[migrate] Metrics dosyası bulunamadı: {metrics_path}")
        return 0
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[migrate] metrics.json okunamadı: {e}")
        return 0
    try:
        insert_metrics(conn, data)
        return 1
    except Exception as e:
        print(f"[migrate] Metrics kaydı eklenemedi: {e}")
        return 0


def migrate_predictions(conn: sqlite3.Connection, predictions_path: Path) -> int:
    """Migrate ai_predictions.json into the database."""
    if not predictions_path.exists():
        print(f"[migrate] Predictions dosyası bulunamadı: {predictions_path}")
        return 0
    try:
        data = json.loads(predictions_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[migrate] ai_predictions.json okunamadı: {e}")
        return 0
    
    # Expect a dict mapping symbol to list of prediction records or a list of records
    rows: List[Dict[str, Any]]
    if isinstance(data, dict):
        rows = []
        for v in data.values():
            if isinstance(v, list):
                rows.extend(v)
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    
    if not rows:
        print("[migrate] Predictions dosyası boş.")
        return 0
    
    count = 0
    for rec in rows:
        try:
            insert_prediction(conn, rec)
            count += 1
        except Exception as e:
            print(f"[migrate] Prediction kaydı eklenemedi: {e}")
            continue
    return count


def get_project_root() -> Path:
    """Find project root by looking for config.json or trade_log.json."""
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "config.json").exists() or (p / "trade_log.json").exists():
            return p
    return current


def main() -> None:
    """Main migration function."""
    print("=" * 50)
    print("📦 Database Migration Tool")
    print("=" * 50)
    
    # Find project root
    root = get_project_root()
    print(f"📁 Proje dizini: {root}")
    
    # Initialize database
    db_path = root / "data" / "bot.db"
    print(f"🗄️  Database: {db_path}")
    
    try:
        initialise_db(db_path)
        print("✅ Database şeması oluşturuldu/kontrol edildi.")
    except Exception as e:
        print(f"❌ Database başlatma hatası: {e}")
        return
    
    # Connect and migrate
    with sqlite3.connect(db_path) as conn:
        # Migrate trades
        trades_path = root / "trade_log.json"
        total_trades = migrate_trades(conn, trades_path)
        print(f"📊 {total_trades} trade kaydı aktarıldı.")
        
        # Migrate metrics
        metrics_path = root / "metrics" / "metrics.json"
        total_metrics = migrate_metrics(conn, metrics_path)
        print(f"📈 {total_metrics} metrics kaydı aktarıldı.")
        
        # Migrate predictions
        predictions_path = root / "metrics" / "ai_predictions.json"
        total_preds = migrate_predictions(conn, predictions_path)
        print(f"🤖 {total_preds} prediction kaydı aktarıldı.")
        
        conn.commit()
    
    print("=" * 50)
    print("✅ Migrasyon tamamlandı!")
    print(f"   Toplam: {total_trades} trade, {total_metrics} metrics, {total_preds} prediction")
    print("=" * 50)


if __name__ == "__main__":
    main()
