"""Migration: Fix UNIQUE constraint from (symbol,epoch,quote) to (symbol,epoch)."""
import sqlite3, sys, os

def migrate(db_path):
    if not os.path.exists(db_path):
        logger.info(f"DB not found: {db_path}"); sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    logger.info("Migrating UNIQUE constraint...")
    try:
        conn.executescript("""
            CREATE TABLE ticks_new (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT    NOT NULL,
                epoch  REAL    NOT NULL,
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
        logger.info(f"Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate(sys.argv[1] if len(sys.argv) > 1 else "data/tick_store.db")
