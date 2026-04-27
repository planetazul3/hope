#!/usr/bin/env python3
"""
Professional Deriv Tick Exporter
--------------------------------
Exports ticks from SQLite to CSV/Parquet with asset‑based naming.
"""

import argparse
import gzip
import logging
import os
import sqlite3          # <-- ADD THIS
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------
def get_last_epoch_from_csv(csv_path: str) -> float | None:
    if not os.path.exists(csv_path):
        return None
    try:
        if csv_path.endswith('.gz'):
            with gzip.open(csv_path, 'rt') as f:
                last_line = None
                for line in f:
                    last_line = line
                if last_line:
                    parts = last_line.strip().split(',')
                    return float(parts[1]) if len(parts) >= 3 else float(parts[0])
        else:
            with open(csv_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0:
                    return None
                f.seek(max(0, f.tell() - 1024), os.SEEK_SET)
                lines = f.readlines()
                if not lines:
                    return None
                last_line = lines[-1].decode().strip()
                if not last_line and len(lines) > 1:
                    last_line = lines[-2].decode().strip()
                if not last_line:
                    return None
                parts = last_line.split(',')
                return float(parts[1]) if len(parts) >= 3 else float(parts[0])
    except Exception:
        logger.warning("Could not read last epoch from CSV", exc_info=True)
        return None

def validate_data_streaming(chunk: pd.DataFrame, last_states: dict, expected_interval: float = 1.0) -> None:
    if chunk.empty:
        return
    dupes = chunk.duplicated(subset=['symbol', 'epoch']).sum()
    if dupes:
        logger.info("WARNING: Found %d duplicate symbol-epochs in chunk.", dupes)
    for symbol, group in chunk.groupby('symbol'):
        prev_epoch = last_states.get(symbol)
        if prev_epoch is not None and not group.empty and group['epoch'].iloc[0] <= prev_epoch:
            logger.info("WARNING: [%s] Duplicate/out-of-order tick at epoch %s", symbol, group['epoch'].iloc[0])
        epochs = group['epoch']
        if prev_epoch is not None:
            epochs = pd.concat([pd.Series([prev_epoch]), epochs])
        diffs = epochs.diff().dropna()
        gaps = diffs[diffs > expected_interval * 2.1]
        if not gaps.empty:
            logger.info("WARNING: [%s] Found %d gaps > %.1fs.", symbol, len(gaps), expected_interval * 2.1)
            logger.info("Max gap: %.2fs", gaps.max())
        last_states[symbol] = group['epoch'].iloc[-1]

def print_stats(df: pd.DataFrame) -> None:
    if df.empty:
        return
    logger.info("\n--- Tick Statistics ---")
    logger.info(df['quote'].describe())
    try:
        counts, bins = np.histogram(df['quote'], bins=10)
        max_count = counts.max()
        logger.info("\nPrice Distribution Histogram:")
        for i in range(len(counts)):
            bar = '#' * int(counts[i] / max_count * 20)
            logger.info(f"{bins[i]:10.2f} - {bins[i+1]:10.2f} | {bar} ({counts[i]})")
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------
def export_ticks(args: argparse.Namespace) -> None:
    db_path = args.db
    if not os.path.exists(db_path):
        logger.error("Database not found at %s", db_path)
        sys.exit(1)

    # --- Output path with symbol substitution ---
    symbol = args.symbol
    csv_path = args.csv
    if '{symbol}' in csv_path and symbol:
        csv_path = csv_path.replace('{symbol}', symbol)
    elif symbol and csv_path == 'data/ticks.csv':  # default fallback
        base, ext = os.path.splitext(csv_path)
        if ext == '.gz':
            base = os.path.splitext(base)[0]
            ext = '.csv.gz'
        csv_path = f"data/{symbol}_ticks{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    compression = 'gzip' if args.compress or csv_path.endswith('.gz') else None
    if compression and not csv_path.endswith('.gz'):
        csv_path += '.gz'

    # --- Incremental logic ---
    start_epoch = args.start
    mode = 'w'
    is_incremental_active = False
    if args.incremental:
        last_epoch = get_last_epoch_from_csv(csv_path)
        if last_epoch:
            logger.info("Incremental mode: last epoch in CSV is %s", last_epoch)
            start_epoch = last_epoch
            is_incremental_active = True
            mode = 'a'

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    try:
        # Check schema
        cursor = conn.execute("PRAGMA table_info(ticks)")
        columns = [row[1] for row in cursor.fetchall()]
        has_symbol = "symbol" in columns

        select_cols = "symbol, epoch, quote" if has_symbol else "epoch, quote"
        query = f"SELECT {select_cols} FROM ticks"
        conditions = []
        params = []

        if symbol and has_symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        elif symbol:
            logger.warning("Symbol column missing – ignoring --symbol filter.")

        if args.hours:
            start_time = datetime.now() - timedelta(hours=args.hours)
            conditions.append("epoch >= ?")
            params.append(start_time.timestamp())
        elif is_incremental_active:
            conditions.append("epoch > ?")
            params.append(start_epoch)
        elif start_epoch:
            conditions.append("epoch >= ?")
            params.append(start_epoch)

        if args.end:
            conditions.append("epoch <= ?")
            params.append(args.end)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY epoch ASC"

        count_query = "SELECT COUNT(*) FROM ticks"
        if conditions:
            count_query += " WHERE " + " AND ".join(conditions)

        total_rows = conn.execute(count_query, params).fetchone()[0]
        if total_rows == 0:
            logger.info("No ticks match the criteria.")
            return

        logger.info("Exporting %d ticks...", total_rows)

        chunk_iter = pd.read_sql_query(query, conn, params=params, chunksize=args.chunk_size)
        first_chunk = True
        last_states = {}

        # --- Parquet setup ---
        parquet_path = csv_path.replace('.csv', '').replace('.gz', '') + '.parquet'
        use_parquet = args.parquet
        parquet_writer = None

        if use_parquet:
            # Try fastparquet first (supports append); fallback to pyarrow for new files only
            try:
                import importlib.util
                if importlib.util.find_spec("fastparquet") is not None:
                    parquet_engine = 'fastparquet'
                else:
                    raise ImportError
            except ImportError:
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    parquet_engine = 'pyarrow'
                except ImportError:
                    logger.warning("Parquet libraries not found. Disabling Parquet export.")
                    use_parquet = False
                    parquet_engine = None

        with tqdm(total=total_rows, desc="Exporting", unit="ticks") as pbar:
            for chunk in chunk_iter:
                # CSV
                header = first_chunk and mode == 'w'
                chunk.to_csv(csv_path, index=False, header=header, mode='a' if not first_chunk else mode, compression=compression)

                # Parquet
                if use_parquet:
                    if parquet_engine == 'fastparquet':
                        chunk.to_parquet(parquet_path, index=False, engine='fastparquet',
                                         append=os.path.exists(parquet_path))
                    else:  # pyarrow – only fresh export
                        table = pa.Table.from_pandas(chunk)
                        if parquet_writer is None:
                            if not first_chunk or mode == 'a':
                                raise RuntimeError("PyArrow fallback does not support incremental Parquet. Use a new file.")
                            parquet_writer = pq.ParquetWriter(parquet_path, table.schema)
                        parquet_writer.write_table(table)

                # Validation & stats
                if args.validate:
                    validate_data_streaming(chunk, last_states)
                if args.stats:
                    logger.info("Chunk stats:\n%s", chunk['quote'].describe())

                pbar.update(len(chunk))
                first_chunk = False

        if use_parquet and parquet_writer:
            parquet_writer.close()

        logger.info("Successfully exported to %s", csv_path)
        if use_parquet:
            logger.info("Parquet file also created at %s", parquet_path)

    except Exception:
        logger.exception("Export failed")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Deriv tick exporter")
    parser.add_argument("--db", default="data/tick_store.db")
    parser.add_argument("--csv", default="data/{symbol}_ticks.csv",
                        help="Output CSV path (use {symbol} for asset name)")
    parser.add_argument("--symbol", default=os.environ.get("DERIV_SYMBOL"), help="Filter by symbol")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--compress", action="store_true", help="Gzip output CSV")
    parser.add_argument("--parquet", action="store_true")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100000)
    args = parser.parse_args()
    export_ticks(args)