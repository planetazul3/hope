"""Migration: Fix UNIQUE constraint from (symbol,epoch,quote) to (symbol,epoch)."""
import sqlite3
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def migrate(db_path):
    if not os.path.exists(db_path):
        logger.error(f"DB not found: {db_path}")
        sys.exit(1)
        
    logger.info(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # Enforce architectural standards for SQLite I/O
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    logger.info("Migrating UNIQUE constraint to (symbol, epoch)...")
    try:
        conn.executescript("""
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
            CREATE INDEX IF NOT EXISTS idx_symbol_epoch ON ticks(symbol, epoch);
        """)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
        logger.info(f"Migration complete. {count} rows retained.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate(sys.argv[1] if len(sys.argv) > 1 else "data/tick_store.db")
