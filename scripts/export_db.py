import sqlite3
import pandas as pd
import argparse
import sys
import os
import gzip
from tqdm import tqdm
from datetime import datetime, timedelta

def validate_data_streaming(chunk, last_states, expected_interval=1.0):
    """
    Checks for gaps and duplicates in the tick data across chunks.
    last_states: Dict[symbol, float] - storing the last epoch of each symbol.
    """
    if chunk.empty:
        return

    # Check for duplicates within chunk
    dupes = chunk.duplicated(subset=['symbol', 'epoch']).sum()
    if dupes > 0:
        print(f"WARNING: Found {dupes} duplicate symbol-epochs in chunk.")

    # Check for gaps and duplicates against previous chunk state
    for symbol, group in chunk.groupby('symbol'):
        prev_epoch = last_states.get(symbol)
        
        # Check cross-chunk duplicate
        if prev_epoch is not None and not group.empty and group['epoch'].iloc[0] <= prev_epoch:
             print(f"WARNING: [{symbol}] Found duplicate or out-of-order tick at epoch {group['epoch'].iloc[0]}")

        # Check gaps within current group
        epochs = group['epoch']
        if prev_epoch is not None:
            # Prepend last epoch to check gap at chunk seam
            epochs = pd.concat([pd.Series([prev_epoch]), epochs])
        
        diffs = epochs.diff().dropna()
        gaps = diffs[diffs > expected_interval * 2.1] # Allow jitter (2.1s threshold as per standards)
        if not gaps.empty:
            print(f"WARNING: [{symbol}] Found {len(gaps)} gaps > {expected_interval * 2.1}s.")
            print(f"Max gap in chunk: {gaps.max():.2f}s")
        
        last_states[symbol] = group['epoch'].iloc[-1]

def get_last_epoch_from_csv(csv_path):
    """
    Reads the last line of a CSV (support Gzip) to find the last exported epoch.
    Expects 3 columns: symbol, epoch, quote
    """
    if not os.path.exists(csv_path):
        return None
    
    try:
        is_gzip = csv_path.endswith('.gz')
        if not is_gzip:
            with open(csv_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0: return None
                f.seek(max(0, f.tell() - 1024), os.SEEK_SET)
                lines = f.readlines()
                if not lines: return None
                last_line = lines[-1].decode().strip()
                if not last_line and len(lines) > 1:
                    last_line = lines[-2].decode().strip()
                if not last_line: return None
                parts = last_line.split(',')
                # Handle both 2-column (epoch,quote) and 3-column (symbol,epoch,quote)
                return float(parts[1]) if len(parts) >= 3 else float(parts[0])
        else:
            with gzip.open(csv_path, 'rt') as f:
                last_line = None
                for line in f:
                    last_line = line
                if last_line:
                    parts = last_line.strip().split(',')
                    return float(parts[1]) if len(parts) >= 3 else float(parts[0])
    except Exception as e:
        print(f"Warning: Could not read last epoch from CSV: {e}")
        return None
    return None

def print_stats(df):
    """
    Prints summary statistics of the exported ticks.
    """
    if df.empty: return
    print("\n--- Tick Statistics ---")
    print(df['quote'].describe())
    
    # Simple text histogram
    try:
        import numpy as np
        counts, bins = np.histogram(df['quote'], bins=10)
        max_count = counts.max()
        print("\nPrice Distribution Histogram:")
        for i in range(len(counts)):
            bar = '#' * int(counts[i] / max_count * 20)
            print(f"{bins[i]:10.2f} - {bins[i+1]:10.2f} | {bar} ({counts[i]})")
    except ImportError:
        pass

def export_ticks(args):
    db_path = args.db
    csv_path = args.csv
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    # Determine compression
    compression = None
    if args.compress or csv_path.endswith('.gz'):
        compression = 'gzip'
        if not csv_path.endswith('.gz'):
            csv_path += '.gz'

    # Handle Incremental
    start_epoch = args.start
    mode = 'w'
    is_incremental_active = False
    if args.incremental:
        last_epoch = get_last_epoch_from_csv(csv_path)
        if last_epoch:
            print(f"Incremental mode: Last epoch in CSV was {last_epoch}")
            start_epoch = last_epoch
            is_incremental_active = True
            mode = 'a'

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        
        # Check schema
        cursor = conn.execute("PRAGMA table_info(ticks)")
        columns = [row[1] for row in cursor.fetchall()]
        has_symbol = "symbol" in columns

        # Build Query
        select_cols = "symbol, epoch, quote" if has_symbol else "epoch, quote"
        query = f"SELECT {select_cols} FROM ticks"
        
        conditions = []
        params = []

        if args.symbol and has_symbol:
            conditions.append("symbol = ?")
            params.append(args.symbol)
        elif args.symbol:
            print("Warning: Symbol column not found in database. Filtering by symbol will be ignored.")

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

        # Get total count for progress bar
        count_query = f"SELECT COUNT(*) FROM ticks"
        if conditions:
            count_query += " WHERE " + " AND ".join(conditions)
        
        total_rows = conn.execute(count_query, params).fetchone()[0]
        if total_rows == 0:
            print("No new ticks found matching the criteria.")
            return

        print(f"Exporting {total_rows:,} ticks...")

        # Process in chunks
        chunk_iter = pd.read_sql_query(query, conn, params=params, chunksize=args.chunk_size)
        
        first_chunk = True
        last_states = {} # For streaming validation
        
        # Parquet handling
        parquet_path = csv_path.replace('.csv', '').replace('.gz', '') + '.parquet'
        use_parquet = args.parquet
        if use_parquet:
            try:
                import pyarrow
            except ImportError:
                try:
                    import fastparquet
                except ImportError:
                    print("Warning: Neither 'pyarrow' nor 'fastparquet' found. Parquet export will be disabled.")
                    use_parquet = False

        with tqdm(total=total_rows, desc="Exporting", unit="ticks") as pbar:
            for chunk in chunk_iter:
                # CSV Export
                header = False 
                current_mode = mode if first_chunk else 'a'
                chunk.to_csv(csv_path, index=False, header=header, mode=current_mode, compression=compression)
                
                # Incremental Parquet Export (Append)
                if use_parquet:
                    try:
                        # fastparquet supports append; pyarrow requires reading/writing or partitioned files
                        # We'll use a simpler 'concatenate and write' if file is small, 
                        # but for hardening we should try to append.
                        chunk.to_parquet(parquet_path, index=False, engine='fastparquet', append=os.path.exists(parquet_path))
                    except (ImportError, TypeError):
                        # Fallback for pyarrow or if fastparquet fails
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                        table = pa.Table.from_pandas(chunk)
                        if first_chunk and mode == 'w':
                            writer = pq.ParquetWriter(parquet_path, table.schema)
                            writer.write_table(table)
                            writer.close()
                        else:
                            print("WARNING: 'fastparquet' is required for safe incremental Parquet appends. PyArrow fallback is disabled to prevent OOM.")
                            pass

                # Streaming Validation
                if args.validate:
                    validate_data_streaming(chunk, last_states)
                
                # Stats (Incremental accumulation could be added here, for now we print per chunk if --stats)
                if args.stats:
                    print(f"\nChunk Stats:")
                    print(chunk['quote'].describe())

                pbar.update(len(chunk))
                first_chunk = False

        if use_parquet:
            print(f"Successfully exported Parquet to {parquet_path}")
        print(f"Successfully exported ticks to {csv_path}")
        
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Deriv tick exporter")
    parser.add_argument("--db", default="data/tick_store.db", help="SQLite DB path")
    parser.add_argument("--csv", default="data/ticks.csv", help="Output CSV path")
    parser.add_argument("--symbol", help="Filter by symbol")
    parser.add_argument("--hours", type=float, help="Export last N hours of data")
    parser.add_argument("--start", type=float, help="Start epoch")
    parser.add_argument("--end", type=float, help="End epoch")
    parser.add_argument("--compress", action="store_true", help="Gzip the output CSV")
    parser.add_argument("--parquet", action="store_true", help="Export to Parquet format")
    parser.add_argument("--incremental", action="store_true", help="Only export ticks newer than last epoch in CSV")
    parser.add_argument("--validate", action="store_true", help="Run gap and duplicate analysis")
    parser.add_argument("--stats", action="store_true", help="Print summary statistics and histogram")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Chunk size for processing")
    
    args = parser.parse_args()
    
    export_ticks(args)
