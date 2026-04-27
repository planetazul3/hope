#!/usr/bin/env python3
"""
Deriv Database Migration Utility
--------------------------------
Enforces robust UNIQUE(symbol, epoch) constraints
on legacy tick databases to guarantee data integrity.
"""

import argparse
import logging
import os
import sqlite3
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MigrateDB")

def migrate(db_path: str) -> None:
    if not os.path.exists(db_path):
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    logger.info("Connecting to database: %s", db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    try:
        # Check if 'ticks' table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ticks'")
        if not cursor.fetchone():
            logger.info("Table 'ticks' does not exist in %s. Nothing to migrate.", db_path)
            return

        # Check if index already exists to avoid redundant migrations
        cursor = conn.execute("PRAGMA index_list(ticks)")
        indexes = [row[1] for row in cursor.fetchall()]
        if "idx_symbol_epoch" in indexes:
            logger.info("Migration already applied. UNIQUE(symbol, epoch) index found. Nothing to do.")
            return

        logger.info("Migrating UNIQUE constraint to (symbol, epoch)...")
        # executescript performs an implicit COMMIT before starting, so we wrap it
        # in an explicit transaction for atomicity.
        conn.executescript("""
            BEGIN TRANSACTION;
            
            CREATE TABLE ticks_new (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT    NOT NULL,
                epoch  INTEGER NOT NULL,
                quote  REAL    NOT NULL,
                UNIQUE(symbol, epoch)
            );
            
            INSERT OR IGNORE INTO ticks_new (symbol, epoch, quote)
                SELECT symbol, epoch, quote FROM ticks ORDER BY id ASC;
                
            DROP TABLE ticks;
            ALTER TABLE ticks_new RENAME TO ticks;
            
            CREATE INDEX idx_symbol_epoch ON ticks(symbol, epoch);
            
            COMMIT;
        """)
        
        count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
        logger.info("Migration complete. %d distinct rows retained.", count)
    except Exception:
        logger.exception("Migration failed")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Database migration utility for Deriv tick storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default="data/tick_store.db", help="SQLite database path")
    args = parser.parse_args()
    
    migrate(args.db)

if __name__ == "__main__":
    main()
