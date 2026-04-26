import asyncio
import sqlite3
import json
import argparse
import sys
import os
import time
import signal
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets", file=sys.stderr)
    sys.exit(1)

# --- Configuration & Constants ---

DEFAULT_APP_ID = os.environ.get("DERIV_APP_ID", "1089")
BATCH_SIZE = 5000
RATE_LIMIT_SLEEP = 1.0
RECONNECT_DELAY = 5

# Set up professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Collector")

# --- Database Layer ---

class TickStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                epoch REAL NOT NULL,
                quote REAL NOT NULL,
                UNIQUE(symbol, epoch)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_epoch ON ticks(symbol, epoch)")
        self.conn.commit()

    def insert_batch(self, symbol: str, epochs: List[float], quotes: List[float]) -> int:
        if not epochs: return 0
        try:
            cursor = self.conn.cursor()
            data = [(symbol, e, q) for e, q in zip(epochs, quotes)]
            cursor.executemany(
                "INSERT OR IGNORE INTO ticks (symbol, epoch, quote) VALUES (?, ?, ?)",
                data
            )
            self.conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Database error: {e}")
            return 0

    def get_last_epoch(self, symbol: str) -> Optional[float]:
        cursor = self.conn.execute("SELECT MAX(epoch) FROM ticks WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        return row[0] if row else None

    def close(self):
        if self.conn:
            self.conn.close()

# --- API Interaction Layer ---

class DerivClient:
    def __init__(self, app_id: str):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self):
        logger.info(f"Connecting to Deriv (AppID: {self.app_id})...")
        self.ws = await websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10)

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ws: raise RuntimeError("Not connected")
        if "req_id" not in payload:
            payload["req_id"] = int(time.time() * 1000) % 10000
        await self.ws.send(json.dumps(payload))
        response = json.loads(await self.ws.recv())
        if "error" in response:
            raise RuntimeError(f"API Error: {response['error'].get('message')}")
        return response

# --- Logic Layer ---

class CollectionService:
    def __init__(self, client: DerivClient, store: TickStore, symbol: str):
        self.client = client
        self.store = store
        self.symbol = symbol
        self.shutdown_event = asyncio.Event()

    async def interruptible_sleep(self, seconds: float):
        """Sleep that can be interrupted by the shutdown event."""
        try:
            await asyncio.wait_for(self.shutdown_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass # Normal timeout, continue

    async def list_active_symbols(self):
        payload = {"active_symbols": "brief", "product_type": "basic", "landing_company": "svg"}
        resp = await self.client.request(payload)
        symbols = [s for s in resp["active_symbols"] if s["market"] == "synthetic_index"]
        print(f"\n{'Symbol':<15} | {'Display Name':<30}")
        print("-" * 50)
        for s in sorted(symbols, key=lambda x: x['symbol']):
            print(f"{s['symbol']:<15} | {s['display_name']:<30}")
        print("-" * 50)

    async def collect_history(self, target_count: Optional[int] = None, hours: Optional[float] = None):
        logger.info(f"Starting HISTORY collection for {self.symbol}")
        boundary = time.time() - (hours * 3600) if hours else None
        total_new, current_end, prev_oldest = 0, "latest", None
        empty_batches = 0

        while not self.shutdown_event.is_set():
            try:
                resp = await self.client.request({
                    "ticks_history": self.symbol, 
                    "style": "ticks", 
                    "end": current_end, 
                    "count": BATCH_SIZE
                })
                history = resp.get("history", {})
                times, prices = history.get("times", []), history.get("prices", [])

                if not times:
                    logger.info("History exhausted (empty response).")
                    break

                inserted = self.store.insert_batch(self.symbol, times, prices)
                total_new += inserted
                oldest = min(times)
                logger.info(f"[{self.symbol}] Batch: {len(times)} | New: {inserted} | Oldest: {datetime.fromtimestamp(oldest)}")

                # Termination conditions
                if oldest == prev_oldest:
                    logger.info("Oldest tick unchanged, stopping.")
                    break
                
                if prev_oldest and oldest > prev_oldest:
                    logger.warning(f"API returned forward-jumping data ({oldest} > {prev_oldest}). Possible rate limit reset or connection issue. Stopping history mode.")
                    break

                if inserted == 0:
                    empty_batches += 1
                    if empty_batches >= 2:
                        logger.info("Hit existing data (2 consecutive batches with 0 new ticks). Stopping.")
                        break
                else:
                    empty_batches = 0

                if (target_count and total_new >= target_count) or (boundary and oldest <= boundary):
                    logger.info("Requested target or time boundary reached.")
                    break

                if len(times) < BATCH_SIZE * 0.9:
                    logger.info(f"Received partial batch ({len(times)} < {BATCH_SIZE}). This is likely the end of available history.")
                    break

                current_end, prev_oldest = str(int(oldest)), oldest
                await self.interruptible_sleep(RATE_LIMIT_SLEEP)
            except Exception as e:
                logger.error(f"History fetch error: {e}")
                break

    async def run_live(self):
        logger.info(f"Starting LIVE subscription for {self.symbol}")
        while not self.shutdown_event.is_set():
            try:
                await self.client.ws.send(json.dumps({"ticks": self.symbol, "subscribe": 1}))
                async for message in self.client.ws:
                    if self.shutdown_event.is_set(): break
                    data = json.loads(message)
                    if data.get("msg_type") == "tick":
                        t = data["tick"]
                        self.store.insert_batch(self.symbol, [t["epoch"]], [t["quote"]])
                        if int(t["epoch"]) % 100 == 0:
                            logger.info(f"Live Tick: {self.symbol} @ {t['quote']}")
            except (ConnectionClosed, Exception) as e:
                if self.shutdown_event.is_set(): break
                logger.warning(f"Live connection lost ({e}). Reconnecting...")
                await asyncio.sleep(RECONNECT_DELAY)
                await self.client.connect()

    def signal_shutdown(self):
        logger.info("Shutdown signal received. Cleaning up...")
        self.shutdown_event.set()
        # To break out of blocking recv()
        if self.client.ws:
            asyncio.create_task(self.client.ws.close())

async def main():
    parser = argparse.ArgumentParser(
        description="Professional Deriv Tick Collector: A robust, asynchronous service for historical backfilling and live data ingestion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Operation Modes:
  history   Fetch historical ticks backwards from 'latest' until --count or --hours is reached.
  live      Subscribe to a real-time WebSocket stream for continuous data ingestion.
  both      Perform a historical backfill first, then seamlessly transition to live subscription.
  list      Query the Deriv API for all available Synthetic Index symbols and exit.

Examples:
  python3 scripts/tick_collector.py --mode list
  python3 scripts/tick_collector.py --symbol 1HZ100V --mode history --hours 24
  python3 scripts/tick_collector.py --symbol R_100 --mode both
        """
    )
    parser.add_argument("--symbol", default="1HZ100V", help="Deriv symbol to collect (default: 1HZ100V)")
    parser.add_argument("--db", default="data/tick_store.db", help="Path to SQLite database (default: data/tick_store.db)")
    parser.add_argument("--mode", choices=["history", "live", "both", "list"], default="history", help="Collection strategy")
    parser.add_argument("--hours", type=float, help="History window size in hours")
    parser.add_argument("--count", type=int, help="Target number of historical ticks to collect")
    args = parser.parse_args()

    client, store = DerivClient(DEFAULT_APP_ID), TickStore(args.db)
    service = CollectionService(client, store, args.symbol)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, service.signal_shutdown)

    try:
        await client.connect()
        if args.mode == "list": await service.list_active_symbols()
        elif args.mode == "history": await service.collect_history(target_count=args.count, hours=args.hours)
        elif args.mode == "live": await service.run_live()
        elif args.mode == "both":
            await service.collect_history(target_count=args.count, hours=args.hours)
            if not service.shutdown_event.is_set(): await service.run_live()
    finally:
        await client.disconnect()
        store.close()
        logger.info("Exited safely.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
