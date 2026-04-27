"""One‑time migration: enforce UNIQUE(symbol, epoch)."""
import logging
import sys
import os
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def migrate(db_path: str) -> None:
    if not os.path.exists(db_path):
        logger.error("DB not found: %s", db_path)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    try:
        # Check if old schema still exists
        cursor = conn.execute("PRAGMA index_list(ticks)")
        indexes = [row[1] for row in cursor.fetchall()]
        if "idx_symbol_epoch" in indexes:
            logger.info("Migration already applied. Nothing to do.")
            return

        logger.info("Migrating UNIQUE constraint to (symbol, epoch)...")
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
        logger.info("Migration complete. %d rows retained.", count)
    except Exception:
        logger.exception("Migration failed")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate(sys.argv[1] if len(sys.argv) > 1 else "data/tick_store.db")