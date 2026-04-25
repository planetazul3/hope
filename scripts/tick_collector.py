import asyncio
import sqlite3
import json
import argparse
import os
import sys
import time
import signal
from pathlib import Path

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets", file=sys.stderr)
    sys.exit(1)

# Default configuration
app_id = "1089"
DERIV_WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
BATCH_SIZE = 5000
RATE_LIMIT_SLEEP = 1.5
TICKS_PER_HOUR = 3600 # Approx for 1s indices

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch REAL NOT NULL,
            quote REAL NOT NULL,
            UNIQUE(epoch, quote)
        )
    """)
    conn.commit()
    return conn

def insert_batch(conn: sqlite3.Connection, epochs: list, quotes: list) -> int:
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT OR IGNORE INTO ticks (epoch, quote) VALUES (?, ?)",
        zip(epochs, quotes)
    )
    conn.commit()
    return cursor.rowcount

async def fetch_batch(ws, symbol: str, end_epoch: int = None) -> dict:
    request = {
        "ticks_history": symbol,
        "style": "ticks",
        "end": end_epoch if end_epoch is not None else "latest",
        "count": BATCH_SIZE,
        "req_id": int(time.time() * 1000) % 10000,
    }

    await ws.send(json.dumps(request))
    raw = await ws.recv()
    response = json.loads(raw)

    if "error" in response:
        raise RuntimeError(
            f"API error: {response['error']['code']} — {response['error']['message']}"
        )

    if response.get("msg_type") != "history":
        raise RuntimeError(f"Unexpected message type: {response.get('msg_type')}")

    return response["history"]

async def collect_ticks(symbol: str, target_count: int, db_path: str, hours: float = None):
    window_start_epoch = int(time.time()) - int(hours * 3600) if hours else None
    if hours:
        target_count = int(hours * TICKS_PER_HOUR)
        print(f"Time-window mode: {hours}h window → ~{target_count:,} target ticks")
    
    conn = init_db(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM ticks")
    existing = cursor.fetchone()[0]
    print(f"Existing ticks in DB: {existing:,}")

    total_inserted = 0
    end_epoch = None
    prev_oldest_epoch = None
    batch_num = 0

    print(f"Starting collection: symbol={symbol}, target={target_count:,}, db={db_path}")
    print("Press Ctrl+C to stop gracefully.")

    try:
        async with websockets.connect(DERIV_WS_URL) as ws:
            while total_inserted < target_count:
                batch_num += 1
                print(f"Batch {batch_num}: requesting {BATCH_SIZE} ticks"
                      + (f" ending at epoch {end_epoch}" if end_epoch is not None else " (latest)"))

                try:
                    history = await fetch_batch(ws, symbol, end_epoch)
                except Exception as e:
                    print(f"Fetch failed: {e}")
                    break

                epochs = history["times"]
                quotes = history["prices"]

                if not epochs:
                    print("No ticks returned — server history limit reached.")
                    break

                oldest_epoch = min(epochs)

                if oldest_epoch == prev_oldest_epoch:
                    print(f"Oldest epoch unchanged ({oldest_epoch}) — server history exhausted.")
                    break

                if window_start_epoch and oldest_epoch < window_start_epoch:
                    # Still insert the current batch but stop after
                    insert_batch(conn, epochs, quotes)
                    print(f"Reached window boundary (epoch {window_start_epoch}) — stopping.")
                    break

                inserted = insert_batch(conn, epochs, quotes)
                total_inserted += inserted

                print(
                    f"  → Received {len(epochs)} ticks, inserted {inserted} new | "
                    f"oldest epoch: {oldest_epoch} | total inserted: {total_inserted:,}"
                )

                prev_oldest_epoch = oldest_epoch
                end_epoch = oldest_epoch

                if len(epochs) < BATCH_SIZE:
                    print("Received fewer ticks than batch size — history exhausted.")
                    break

                if total_inserted < target_count:
                    print(f"  Sleeping {RATE_LIMIT_SLEEP}s (rate limit)...")
                    await asyncio.sleep(RATE_LIMIT_SLEEP)
                    
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\nInterrupt received, shutting down gracefully...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        final_count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
        conn.close()
        print(f"\nSummary: Batches: {batch_num}, new ticks inserted: {total_inserted:,}, "
              f"total in DB: {final_count:,}")

def main():
    parser = argparse.ArgumentParser(description="Deriv tick collector")
    parser.add_argument("--symbol", default="1HZ75V", help="Deriv symbol (default: 1HZ75V)")
    parser.add_argument("--count", type=int, default=86400, help="Target ticks (default: 86400 for 24h)")
    parser.add_argument("--db", default="data/tick_store.db", help="DB path")
    parser.add_argument("--hours", type=float, default=24, help="Last N hours")
    args = parser.parse_args()

    # Use a loop that can be interrupted
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Define a handler for signals
    def stop_loop():
        for task in asyncio.all_tasks(loop):
            task.cancel()

    try:
        loop.run_until_complete(collect_ticks(
            symbol=args.symbol,
            target_count=args.count,
            db_path=args.db,
            hours=args.hours,
        ))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

if __name__ == "__main__":
    main()
