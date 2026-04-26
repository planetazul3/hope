import asyncio
import sqlite3
import json
import argparse
import sys
import os
import time
import signal
import logging
import random
from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets", file=sys.stderr)
    sys.exit(1)

# --- Configuration & Constants ---

DEFAULT_APP_ID = os.environ.get("DERIV_APP_ID", "1089")
BATCH_SIZE = 5000
RATE_LIMIT_SLEEP = 1.0
RECONNECT_DELAY_BASE = 5.0
RECONNECT_DELAY_MAX = 60.0
MAX_RECONNECT_ATTEMPTS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Collector")


def _backoff(attempt: int) -> float:
    """Jittered exponential backoff capped at RECONNECT_DELAY_MAX."""
    delay = min(RECONNECT_DELAY_BASE * (2 ** attempt), RECONNECT_DELAY_MAX)
    return delay * (0.5 + random.random() * 0.5)


# --- Database Layer ---

class TickStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-65536")  # 64 MB page cache
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT    NOT NULL,
                epoch  REAL    NOT NULL,
                quote  REAL    NOT NULL,
                UNIQUE(symbol, epoch, quote)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_epoch_quote ON ticks(symbol, epoch, quote)"
        )
        self.conn.commit()

    def insert_batch(self, symbol: str, epochs: List[float], quotes: List[float]) -> int:
        if not epochs:
            return 0
        valid = [
            (symbol, e, q)
            for e, q in zip(epochs, quotes)
            if isinstance(e, (int, float)) and isinstance(q, (int, float)) and q > 0
        ]
        if not valid:
            return 0
        try:
            cursor = self.conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO ticks (symbol, epoch, quote) VALUES (?, ?, ?)",
                valid,
            )
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.conn.rollback()
            return 0

    def get_count(self, symbol: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM ticks WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row[0] if row else 0

    def get_latest_epoch(self, symbol: str) -> Optional[float]:
        row = self.conn.execute(
            "SELECT MAX(epoch) FROM ticks WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


# --- API Interaction Layer ---

class DerivClient:
    """
    WebSocket client with a background message-listener that routes
    API responses to their awaiting futures by req_id.

    Without this, calling ws.recv() directly in request() would race
    against any concurrent live-tick messages, causing the wrong data
    to be returned or an exception to be raised silently.
    """

    def __init__(self, app_id: str):
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_counter = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._listener_task: Optional[asyncio.Task] = None
        # Queue for non-req_id messages (live ticks, subscription events)
        self.message_queue: asyncio.Queue = asyncio.Queue()

    def _next_req_id(self) -> int:
        self._req_counter = (self._req_counter % 9_999) + 1
        return self._req_counter

    @property
    def is_open(self) -> bool:
        return self.ws is not None and getattr(self.ws.state, 'name', '') == 'OPEN'

    async def connect(self):
        logger.info(f"Connecting to Deriv (AppID: {self.app_id})...")
        self.ws = await websockets.connect(
            self.ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10,
        )
        self._listener_task = asyncio.create_task(
            self._message_listener(), name="ws-listener"
        )
        logger.info("Connected.")

    async def disconnect(self):
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        # Reject all pending request futures
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

        # Drop stale messages from the queue
        self.message_queue = asyncio.Queue()

        if self.ws:
            await self.ws.close()
            self.ws = None

    async def _message_listener(self):
        """Background task: routes responses to futures, queues everything else."""
        try:
            async for raw in self.ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("Received non-JSON frame, skipping.")
                    continue

                req_id = data.get("req_id")
                if req_id and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        if "error" in data:
                            fut.set_exception(
                                RuntimeError(f"API error: {data['error'].get('message')}")
                            )
                        else:
                            fut.set_result(data)
                else:
                    await self.message_queue.put(data)

        except (ConnectionClosed, WebSocketException) as e:
            logger.debug(f"Listener closed: {e}")
        except asyncio.CancelledError:
            pass
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()

    async def request(self, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        if not self.is_open:
            raise RuntimeError("WebSocket not connected")
        req_id = self._next_req_id()
        payload["req_id"] = req_id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        try:
            await self.ws.send(json.dumps(payload))
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.cancel()
            raise RuntimeError(f"Request timed out after {timeout}s (req_id={req_id})")
        except Exception:
            self._pending.pop(req_id, None)
            raise


# --- Statistics ---

class Stats:
    def __init__(self):
        self.inserted = 0
        self.duplicates = 0
        self.batches = 0
        self.reconnects = 0
        self._t0 = time.monotonic()

    def log(self, symbol: str):
        elapsed = time.monotonic() - self._t0
        rate = self.inserted / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"[{symbol}] inserted={self.inserted:,} duplicates={self.duplicates:,} "
            f"batches={self.batches} reconnects={self.reconnects} "
            f"rate={rate:.1f}/s elapsed={elapsed:.0f}s"
        )


# --- Collection Service ---

class CollectionService:
    def __init__(self, client: DerivClient, store: TickStore, symbol: str):
        self.client = client
        self.store = store
        self.symbol = symbol
        self.shutdown_event = asyncio.Event()
        self.stats = Stats()

    async def _sleep(self, seconds: float):
        """Sleep that aborts immediately when shutdown is signalled."""
        try:
            await asyncio.wait_for(self.shutdown_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _request_with_retry(self, payload: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
        """Request wrapper with internal retry logic."""
        attempt = 0
        while not self.shutdown_event.is_set():
            try:
                return await self.client.request(payload)
            except (RuntimeError, asyncio.TimeoutError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = _backoff(attempt)
                logger.warning(f"Request failed ({e}). Retrying {attempt}/{max_retries} in {delay:.1f}s...")
                await self._sleep(delay)
        raise RuntimeError("Shutdown during request retry")

    async def _insert_batch_async(self, symbol: str, times: List[float], prices: List[float]) -> int:
        """Offload synchronous SQLite I/O to a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.store.insert_batch, symbol, times, prices
        )

    async def list_active_symbols(self):
        resp = await self._request_with_retry(
            {"active_symbols": "brief", "product_type": "basic", "landing_company": "svg"}
        )
        symbols = [s for s in resp["active_symbols"] if s["market"] == "synthetic_index"]
        print(f"\n{'Symbol':<15} | {'Display Name':<30}")
        print("-" * 50)
        for s in sorted(symbols, key=lambda x: x["symbol"]):
            print(f"{s['symbol']:<15} | {s['display_name']:<30}")
        print("-" * 50)
        print(f"Total: {len(symbols)} synthetic symbols")

    async def collect_history(
        self,
        target_count: Optional[int] = None,
        hours: Optional[float] = None,
    ):
        logger.info(f"Starting HISTORY collection for {self.symbol}")
        boundary = (time.time() - hours * 3600) if hours else None
        if boundary:
            logger.info(f"  time boundary: {datetime.fromtimestamp(boundary)}")

        current_end = "latest"
        prev_oldest: Optional[float] = None
        consecutive_empty = 0

        while not self.shutdown_event.is_set():
            try:
                resp = await self._request_with_retry({
                    "ticks_history": self.symbol,
                    "style": "ticks",
                    "end": current_end,
                    "count": BATCH_SIZE,
                })
            except Exception as e:
                logger.error(f"History collection failed after retries: {e}")
                break

            history = resp.get("history", {})
            times: List[float] = history.get("times", [])
            prices: List[float] = history.get("prices", [])

            if not times:
                logger.info("History exhausted (empty response).")
                break

            self.stats.batches += 1
            inserted = await self._insert_batch_async(self.symbol, times, prices)
            self.stats.inserted += inserted
            self.stats.duplicates += len(times) - inserted

            oldest, newest = min(times), max(times)
            fmt = lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            logger.info(
                f"[{self.symbol}] batch={self.stats.batches} ticks={len(times)} "
                f"new={inserted} range=[{fmt(oldest)} → {fmt(newest)}]"
            )

            # Termination checks
            if oldest == prev_oldest:
                logger.info("Oldest tick unchanged — reached beginning of available history.")
                break

            if prev_oldest is not None and oldest > prev_oldest:
                logger.warning(
                    f"API returned forward-jumping oldest ({oldest:.0f} > {prev_oldest:.0f}). "
                    "Possible API anomaly. Stopping."
                )
                break

            if inserted == 0:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    logger.info("Two consecutive empty batches — data already present.")
                    break
            else:
                consecutive_empty = 0

            if target_count and self.stats.inserted >= target_count:
                logger.info(f"Target count {target_count:,} reached.")
                break

            if boundary and oldest <= boundary:
                logger.info(f"Time boundary reached.")
                break

            if len(times) < int(BATCH_SIZE * 0.9):
                logger.info(f"Partial batch ({len(times)}) — likely end of history.")
                break

            prev_oldest = oldest
            current_end = str(oldest)  # Don't truncate to avoid skipping fractional ticks
            await self._sleep(RATE_LIMIT_SLEEP)

        self.stats.log(self.symbol)

    async def collect_backfill(self, target_count: Optional[int] = None):
        last_epoch = self.store.get_latest_epoch(self.symbol)
        if not last_epoch:
            logger.info("No existing data found. Falling back to history mode.")
            await self.collect_history(target_count=target_count)
            return

        logger.info(f"Starting BACKFILL collection for {self.symbol} from {datetime.fromtimestamp(last_epoch)}")
        
        start_ts = int(last_epoch)
        consecutive_empty = 0

        while not self.shutdown_event.is_set():
            now_ts = int(time.time())
            if start_ts >= now_ts:
                logger.info("Backfill reached current time.")
                break

            # Fetch in chunks of BATCH_SIZE seconds to avoid hitting the 5000 tick limit
            end_ts = min(start_ts + BATCH_SIZE, now_ts)
            
            try:
                resp = await self._request_with_retry({
                    "ticks_history": self.symbol,
                    "style": "ticks",
                    "start": start_ts,
                    "end": str(end_ts),
                    "count": BATCH_SIZE,
                })
            except Exception as e:
                logger.error(f"Backfill collection failed after retries: {e}")
                break

            history = resp.get("history", {})
            times: List[float] = history.get("times", [])
            prices: List[float] = history.get("prices", [])

            if not times:
                if end_ts >= now_ts:
                    logger.info("Backfill exhausted.")
                    break
                else:
                    start_ts = end_ts
                    await self._sleep(RATE_LIMIT_SLEEP)
                    continue

            self.stats.batches += 1
            inserted = await self._insert_batch_async(self.symbol, times, prices)
            self.stats.inserted += inserted
            self.stats.duplicates += len(times) - inserted

            oldest, newest = min(times), max(times)
            fmt = lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            logger.info(
                f"[{self.symbol}] batch={self.stats.batches} ticks={len(times)} "
                f"new={inserted} range=[{fmt(oldest)} → {fmt(newest)}]"
            )

            if len(times) == BATCH_SIZE:
                # If the batch is full, we move the pointer to the newest tick received.
                # To avoid an infinite loop if many ticks have the same timestamp,
                # we advance by at least 1 second since the API expects integer epochs.
                next_start = int(newest)
                if next_start <= int(start_ts):
                    next_start = int(start_ts) + 1
                start_ts = next_start
            else:
                start_ts = int(end_ts)

            if target_count and self.stats.inserted >= target_count:
                logger.info(f"Target count {target_count:,} reached.")
                break

            await self._sleep(RATE_LIMIT_SLEEP)

        self.stats.log(self.symbol)

    async def run_live(self):
        logger.info(f"Starting LIVE subscription for {self.symbol}")
        attempt = 0
        tick_count = 0
        last_report = time.monotonic()

        while not self.shutdown_event.is_set():
            try:
                if not self.client.is_open:
                    await self.client.connect()

                await self.client.request(
                    {"ticks": self.symbol, "subscribe": 1}
                )
                attempt = 0

                while not self.shutdown_event.is_set():
                    try:
                        data = await asyncio.wait_for(
                            self.client.message_queue.get(), timeout=30.0
                        )
                    except asyncio.TimeoutError:
                        # Check if connection is still alive to avoid "zombie" state
                        if not self.client.is_open:
                            logger.warning("Connection lost during live stream wait.")
                            break
                        continue

                    if data.get("msg_type") != "tick":
                        # If it's the subscription response (handled by req_id) it won't be here,
                        # but other unsolicited messages might.
                        continue

                    t = data.get("tick", {})
                    epoch, quote = t.get("epoch"), t.get("quote")
                    if epoch is None or quote is None:
                        continue

                    inserted = await self._insert_batch_async(self.symbol, [epoch], [quote])
                    if inserted:
                        self.stats.inserted += inserted
                        tick_count += 1

                    now = time.monotonic()
                    if now - last_report >= 60.0:
                        logger.info(
                            f"[{self.symbol}] live: {tick_count} ticks/min | "
                            f"total={self.stats.inserted:,}"
                        )
                        tick_count = 0
                        last_report = now

            except (ConnectionClosed, WebSocketException, OSError, asyncio.TimeoutError) as e:
                if self.shutdown_event.is_set():
                    break
                attempt += 1
                if attempt > MAX_RECONNECT_ATTEMPTS:
                    logger.error(f"Max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) exhausted.")
                    break
                delay = _backoff(attempt)
                self.stats.reconnects += 1
                logger.warning(
                    f"Connection lost or network error ({type(e).__name__}: {e}). Reconnect #{attempt}/{MAX_RECONNECT_ATTEMPTS} "
                    f"in {delay:.1f}s..."
                )
                await self._sleep(delay)

            except Exception as e:
                if self.shutdown_event.is_set():
                    break
                logger.error(f"Unexpected live error: {e}", exc_info=True)
                attempt += 1
                if attempt > MAX_RECONNECT_ATTEMPTS:
                    break
                delay = _backoff(attempt)
                await self._sleep(delay)

    def signal_shutdown(self):
        logger.info("Shutdown signal received.")
        self.shutdown_event.set()


# --- Entry Point ---

async def main():
    parser = argparse.ArgumentParser(
        description="Deriv Tick Collector — async service for historical backfill and live ingestion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  history   Fetch historical ticks backwards from 'latest' until --count or --hours is met.
  backfill  Fetch historical ticks forwards from the last saved epoch until now.
  live      Subscribe to real-time WebSocket stream for continuous ingestion.
  both      Historical backfill first, then seamlessly transition to live.
  list      List all available Synthetic Index symbols and exit.

Examples:
  python3 scripts/tick_collector.py --mode list
  python3 scripts/tick_collector.py --symbol 1HZ100V --mode history --hours 24
  python3 scripts/tick_collector.py --symbol R_100 --mode both --log-level DEBUG
        """,
    )
    parser.add_argument("--symbol", default="1HZ100V", help="Deriv symbol (default: 1HZ100V)")
    parser.add_argument("--db", default="data/tick_store.db", help="SQLite path (default: data/tick_store.db)")
    parser.add_argument("--mode", choices=["history", "backfill", "live", "both", "list"], default="history")
    parser.add_argument("--hours", type=float, help="History window in hours")
    parser.add_argument("--count", type=int, help="Target number of historical ticks")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    client = DerivClient(DEFAULT_APP_ID)
    store = TickStore(args.db)
    service = CollectionService(client, store, args.symbol)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, service.signal_shutdown)

    try:
        await client.connect()
        if args.mode == "list":
            await service.list_active_symbols()
        elif args.mode == "history":
            await service.collect_history(target_count=args.count, hours=args.hours)
        elif args.mode == "backfill":
            await service.collect_backfill(target_count=args.count)
        elif args.mode == "live":
            await service.run_live()
        elif args.mode == "both":
            await service.collect_backfill(target_count=args.count)
            if not service.shutdown_event.is_set():
                await service.run_live()
    finally:
        await client.disconnect()
        store.close()
        logger.info("Exited safely.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
