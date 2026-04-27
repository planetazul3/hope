#!/usr/bin/env python3
"""
Deriv Tick Collector – Professional Async Service
--------------------------------------------------
Modes: history | backfill | live | both | list
Exponential backoff, idempotent storage, automatic reconnect.
"""

import asyncio
import json
import logging
import os
import random
import signal
import sqlite3
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# ---------------------------------------------------------------------------
# Minimal .env loader
# ---------------------------------------------------------------------------
def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                v = v.split("#", 1)[0].strip().strip('"').strip("'")
                k = k.strip()
                if k not in os.environ:
                    os.environ[k] = v

load_env_file()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_APP_ID = os.environ.get("DERIV_APP_ID", "1089")
BATCH_SIZE = 5000               # API max ticks per request
WRITE_BUFFER_FLUSH_SIZE = 1000  # flush to DB every N ticks
WRITE_BUFFER_FLUSH_INTERVAL = 1.0  # seconds between forced flushes
RATE_LIMIT_SLEEP = 1.0          # politeness delay between API calls
RECONNECT_DELAY_BASE = 5.0
RECONNECT_DELAY_MAX = 60.0
MAX_RECONNECT_ATTEMPTS = 10
MAX_RETRIES_PER_REQUEST = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Collector")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _backoff(attempt: int) -> float:
    """Jittered exponential backoff."""
    delay = min(RECONNECT_DELAY_BASE * (2 ** attempt), RECONNECT_DELAY_MAX)
    return delay * (0.5 + random.random() * 0.5)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
class TickStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-65536")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT    NOT NULL,
                epoch  INTEGER NOT NULL,
                quote  REAL    NOT NULL,
                UNIQUE(symbol, epoch)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_epoch ON ticks(symbol, epoch)"
        )
        self.conn.commit()

    def insert_batch(self, symbol: str, epochs: List[float], quotes: List[float]) -> int:
        """Insert a batch of ticks, ignoring duplicates. Returns number of inserted rows."""
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
            cur = self.conn.cursor()
            cur.executemany(
                "INSERT OR IGNORE INTO ticks (symbol, epoch, quote) VALUES (?, ?, ?)",
                valid,
            )
            self.conn.commit()
            return cur.rowcount
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.conn.rollback()
            return 0

    def get_count(self, symbol: str) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM ticks WHERE symbol = ?", (symbol,)).fetchone()
        return row[0] if row else 0

    def get_latest_epoch(self, symbol: str) -> Optional[float]:
        row = self.conn.execute(
            "SELECT MAX(epoch) FROM ticks WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

# ---------------------------------------------------------------------------
# Deriv WebSocket client with request/response routing
# ---------------------------------------------------------------------------
class DerivClient:
    def __init__(self, app_id: str) -> None:
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_counter = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._listener_task: Optional[asyncio.Task] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()

    def _next_req_id(self) -> int:
        self._req_counter = (self._req_counter % 9_999) + 1
        return self._req_counter

    @property
    def is_open(self) -> bool:
        return self.ws is not None and getattr(self.ws.state, 'name', '') == 'OPEN'

    async def connect(self) -> None:
        logger.info(f"Connecting to Deriv (AppID: {self.app_id})...")
        self.ws = await websockets.connect(
            self.ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10,
        )
        self._listener_task = asyncio.create_task(self._message_listener(), name="ws-listener")
        logger.info("Connected.")

    async def disconnect(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        # Reject pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self.message_queue = asyncio.Queue()
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def _message_listener(self) -> None:
        """Routes responses by req_id, others go to the queue."""
        try:
            async for raw in self.ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                req_id = data.get("req_id")
                if req_id and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        if "error" in data:
                            fut.set_exception(
                                RuntimeError(data["error"].get("message", "Unknown API error"))
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

# ---------------------------------------------------------------------------
# Statistics tracker
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self) -> None:
        self.inserted = 0
        self.duplicates = 0
        self.batches = 0
        self.reconnects = 0
        self._t0 = time.monotonic()

    def log(self, symbol: str) -> None:
        elapsed = time.monotonic() - self._t0
        rate = self.inserted / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"[{symbol}] inserted={self.inserted:,} duplicates={self.duplicates:,} "
            f"batches={self.batches} reconnects={self.reconnects} "
            f"rate={rate:.1f}/s elapsed={elapsed:.0f}s"
        )

# ---------------------------------------------------------------------------
# Collection Service
# ---------------------------------------------------------------------------
class CollectionService:
    def __init__(self, client: DerivClient, store: TickStore, symbol: str) -> None:
        self.client = client
        self.store = store
        self.symbol = symbol
        self.shutdown_event = asyncio.Event()
        self.stats = Stats()
        self.write_buffer: List[Tuple[float, float]] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    async def _sleep(self, seconds: float) -> None:
        """Sleep that aborts immediately when shutdown is signalled."""
        try:
            await asyncio.wait_for(self.shutdown_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _request_with_retry(self, payload: Dict[str, Any], max_retries: int = MAX_RETRIES_PER_REQUEST) -> Dict[str, Any]:
        """Exponential backoff wrapper for API requests."""
        attempt = 0
        while not self.shutdown_event.is_set():
            try:
                return await self.client.request(payload)
            except (RuntimeError, asyncio.TimeoutError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = _backoff(attempt)
                logger.warning(f"Request failed ({e}). Retry {attempt}/{max_retries} in {delay:.1f}s...")
                await self._sleep(delay)
        raise RuntimeError("Shutdown during request retry")

    async def _insert_batch_async(self, symbol: str, times: List[float], prices: List[float]) -> int:
        """Offload synchronous DB write to thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.store.insert_batch, symbol, times, prices)

    async def _flush_buffer_task(self) -> None:
        """Periodically flush write buffer."""
        logger.debug("Flush task started.")
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(WRITE_BUFFER_FLUSH_INTERVAL)
                await self.flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush task: {e}")
        await self.flush_buffer()
        logger.debug("Flush task stopped.")

    async def flush_buffer(self) -> None:
        async with self._buffer_lock:
            if not self.write_buffer:
                return
            batch = self.write_buffer
            self.write_buffer = []
        epochs = [t[0] for t in batch]
        quotes = [t[1] for t in batch]
        inserted = await self._insert_batch_async(self.symbol, epochs, quotes)
        if inserted:
            self.stats.inserted += inserted
            self.stats.duplicates += len(batch) - inserted

    async def list_active_symbols(self) -> None:
        resp = await self._request_with_retry(
            {"active_symbols": "brief", "product_type": "basic", "landing_company": "svg"}
        )
        symbols = [s for s in resp["active_symbols"] if s["market"] == "synthetic_index"]
        logger.info(f"\n{'Symbol':<15} | {'Display Name':<30}")
        logger.info("-" * 50)
        for s in sorted(symbols, key=lambda x: x["symbol"]):
            logger.info(f"{s['symbol']:<15} | {s['display_name']:<30}")
        logger.info("-" * 50)
        logger.info(f"Total: {len(symbols)} synthetic symbols")

    async def collect_history(self, target_count: Optional[int] = None, hours: Optional[float] = None) -> None:
        """Fetch historical ticks backwards from 'latest'."""
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
            except Exception:
                logger.exception("History collection failed")
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
            fmt = lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"[{self.symbol}] batch={self.stats.batches} ticks={len(times)} "
                f"new={inserted} range=[{fmt(oldest)} → {fmt(newest)}]"
            )

            # Termination checks
            if oldest == prev_oldest:
                logger.info("Oldest tick unchanged – reached beginning.")
                break
            if prev_oldest is not None and oldest > prev_oldest:
                logger.warning("Forward-jumping oldest – stopping.")
                break
            if inserted == 0:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    logger.info("Two consecutive empty batches – data already present.")
                    break
            else:
                consecutive_empty = 0

            if target_count and self.stats.inserted >= target_count:
                logger.info(f"Target count {target_count:,} reached.")
                break
            if boundary and oldest <= boundary:
                logger.info("Time boundary reached.")
                break
            if len(times) < int(BATCH_SIZE * 0.9):
                logger.info("Partial batch – end of history.")
                break

            prev_oldest = oldest
            current_end = str(oldest)          # float string keeps precision
            await self._sleep(RATE_LIMIT_SLEEP)

        self.stats.log(self.symbol)

    async def collect_backfill(self, target_count: Optional[int] = None) -> None:
        """
        Fetch missing ticks from the last stored epoch up to 'latest'.
        Uses start=last_epoch+0.001 to avoid duplicates, and loops until
        we receive fewer than BATCH_SIZE ticks (meaning we've caught up).
        """
        last_epoch = self.store.get_latest_epoch(self.symbol)
        if not last_epoch:
            logger.info("No existing data, falling back to history mode.")
            await self.collect_history(target_count=target_count)
            return

        logger.info(
            f"Starting BACKFILL for {self.symbol} from "
            f"{datetime.fromtimestamp(last_epoch)}"
        )
        start = last_epoch + 0.001  # avoid re-fetching last stored tick

        while not self.shutdown_event.is_set():
            try:
                resp = await self._request_with_retry({
                    "ticks_history": self.symbol,
                    "style": "ticks",
                    "start": start,
                    "end": "latest",
                    "count": BATCH_SIZE,
                })
            except Exception:
                logger.exception("Backfill request failed")
                break

            history = resp.get("history", {})
            times: List[float] = history.get("times", [])
            prices: List[float] = history.get("prices", [])

            if not times:
                logger.info("Backfill caught up – no new ticks.")
                break

            self.stats.batches += 1
            inserted = await self._insert_batch_async(self.symbol, times, prices)
            self.stats.inserted += inserted
            self.stats.duplicates += len(times) - inserted

            newest = max(times)
            fmt = lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"[{self.symbol}] batch={self.stats.batches} ticks={len(times)} "
                f"new={inserted} range=[{fmt(min(times))} → {fmt(newest)}]"
            )

            if target_count and self.stats.inserted >= target_count:
                logger.info(f"Target count {target_count:,} reached.")
                break

            # If we received fewer than the batch size, we've reached 'latest'
            if len(times) < BATCH_SIZE:
                logger.info("Partial batch – caught up to latest.")
                break

            # Update start just after the last tick we processed
            start = newest + 0.001
            await self._sleep(RATE_LIMIT_SLEEP)

        self.stats.log(self.symbol)

    async def run_live(self) -> None:
        """Subscribe to real‑time ticks and buffer writes."""
        logger.info(f"Starting LIVE subscription for {self.symbol}")
        attempt = 0
        tick_count = 0
        last_report = time.monotonic()

        while not self.shutdown_event.is_set():
            try:
                if not self.client.is_open:
                    await self.client.connect()

                await self.client.request({"ticks": self.symbol, "subscribe": 1})
                attempt = 0

                while not self.shutdown_event.is_set():
                    try:
                        data = await asyncio.wait_for(
                            self.client.message_queue.get(), timeout=30.0
                        )
                    except asyncio.TimeoutError:
                        if not self.client.is_open:
                            logger.warning("Connection lost during live stream.")
                            break
                        continue

                    if data.get("msg_type") != "tick":
                        continue

                    t = data.get("tick", {})
                    epoch, quote = t.get("epoch"), t.get("quote")
                    if epoch is None or quote is None:
                        continue

                    async with self._buffer_lock:
                        self.write_buffer.append((epoch, quote))
                        tick_count += 1
                        if len(self.write_buffer) >= WRITE_BUFFER_FLUSH_SIZE:
                            asyncio.create_task(self.flush_buffer())

                    now = time.monotonic()
                    if now - last_report >= 60.0:
                        logger.info(
                            f"[{self.symbol}] live ticks/min: {tick_count} | "
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
                    f"Connection lost ({type(e).__name__}: {e}). "
                    f"Reconnect #{attempt}/{MAX_RECONNECT_ATTEMPTS} in {delay:.1f}s..."
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

    def signal_shutdown(self) -> None:
        logger.info("Shutdown signal received.")
        self.shutdown_event.set()

    async def start_background_flush(self) -> None:
        """Start periodic flush task and flush any leftover buffer."""
        self._flush_task = asyncio.create_task(self._flush_buffer_task())

    async def stop_background_flush(self) -> None:
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Deriv Tick Collector – professional async service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  history   Fetch backwards from 'latest' until --count or --hours.
  backfill  Fetch forward from last stored tick until now.
  live      Subscribe to real-time WebSocket stream.
  both      Backfill first, then seamlessly go live.
  list      List available synthetic index symbols.

Examples:
  %(prog)s --mode list
  %(prog)s --symbol 1HZ100V --mode history --hours 24
  %(prog)s --symbol R_100 --mode both --log-level DEBUG
        """,
    )
    parser.add_argument("--symbol", default=os.environ.get("DERIV_SYMBOL"), help="Deriv symbol")
    parser.add_argument("--db", default="data/tick_store.db", help="SQLite path")
    parser.add_argument("--mode", choices=["history", "backfill", "live", "both", "list"], default="history")
    parser.add_argument("--hours", type=float, help="History window in hours")
    parser.add_argument("--count", type=int, help="Target number of ticks")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    symbol = args.symbol or os.environ.get("DERIV_SYMBOL", "1HZ100V")
    client = DerivClient(DEFAULT_APP_ID)
    store = TickStore(args.db)
    service = CollectionService(client, store, symbol)

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
            await service.start_background_flush()
            await service.run_live()
        elif args.mode == "both":
            await service.collect_backfill(target_count=args.count)
            if not service.shutdown_event.is_set():
                logger.info("Transitioning to LIVE stream as requested.")
                await service.flush_buffer()                 # flush any leftover from backfill
                await service.start_background_flush()
                await service.run_live()
    finally:
        await client.disconnect()
        await service.stop_background_flush()
        store.close()
        logger.info("Exited safely.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass