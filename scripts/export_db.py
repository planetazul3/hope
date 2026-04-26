import sqlite3
import pandas as pd
import argparse
import sys
import os
import gzip
from tqdm import tqdm
from datetime import datetime, timedelta

def validate_data(df, expected_interval=1.0):
    """
    Checks for gaps and duplicates in the tick data.
    """
    if len(df) < 2:
        return
    
    # Check for duplicates
    dupes = df.duplicated(subset=['epoch']).sum()
    if dupes > 0:
        print(f"WARNING: Found {dupes} duplicate epochs.")

    # Check for gaps
    diffs = df['epoch'].diff().dropna()
    gaps = diffs[diffs > expected_interval * 2] # Allow some jitter
    if not gaps.empty:
        print(f"WARNING: Found {len(gaps)} gaps larger than {expected_interval * 2}s.")
        print(f"Max gap: {gaps.max():.2f}s")

def get_last_epoch_from_csv(csv_path):
    """
    Reads the last line of a CSV (support Gzip) to find the last exported epoch.
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
                return float(last_line.split(',')[0])
        else:
            with gzip.open(csv_path, 'rt') as f:
                last_line = None
                for line in f:
                    last_line = line
                if last_line:
                    return float(last_line.strip().split(',')[0])
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
    if args.incremental:
        last_epoch = get_last_epoch_from_csv(csv_path)
        if last_epoch:
            print(f"Incremental mode: Last epoch in CSV was {last_epoch}")
            start_epoch = last_epoch + 0.0001 # Small epsilon
            mode = 'a'

    # Build Query
    query = "SELECT epoch, quote FROM ticks"
    conditions = []
    params = []

    # Symbol filter (new)
    if args.symbol:
        conditions.append("symbol = ?")
        params.append(args.symbol)

    if args.hours:
        start_time = datetime.now() - timedelta(hours=args.hours)
        conditions.append("epoch >= ?")
        params.append(start_time.timestamp())
    elif start_epoch:
        conditions.append("epoch >= ?")
        params.append(start_epoch)
    
    if args.end:
        conditions.append("epoch <= ?")
        params.append(args.end)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY epoch ASC"

    try:
        conn = sqlite3.connect(db_path)
        
        # Check if symbol column exists to support legacy DBs during transition
        cursor = conn.execute("PRAGMA table_info(ticks)")
        columns = [row[1] for row in cursor.fetchall()]
        if "symbol" not in columns and args.symbol:
            print("Warning: Symbol column not found in database. Filtering by symbol will be ignored.")
            query = query.replace("symbol = ? AND ", "").replace("WHERE symbol = ?", "")
            if args.symbol in params:
                params.remove(args.symbol)

        # Get total count for progress bar
        count_query = "SELECT COUNT(*) FROM ticks"
        if conditions:
            # Re-verify if symbol filter was removed
            final_conditions = []
            final_params = []
            for c, p in zip(conditions, params):
                if "symbol" in c and "symbol" not in columns: continue
                final_conditions.append(c)
                final_params.append(p)
            if final_conditions:
                count_query += " WHERE " + " AND ".join(final_conditions)
        else:
            final_params = []
        
        total_rows = conn.execute(count_query, final_params).fetchone()[0]
        if total_rows == 0:
            print("No new ticks found matching the criteria.")
            return

        print(f"Exporting {total_rows:,} ticks...")

        # Process in chunks
        chunk_iter = pd.read_sql_query(query, conn, params=params, chunksize=args.chunk_size)
        
        first_chunk = True
        all_chunks = [] if (args.validate or args.parquet or args.stats) else None

        with tqdm(total=total_rows, desc="Exporting", unit="ticks") as pbar:
            for chunk in chunk_iter:
                # CSV Export
                header = False # Maintain compatibility
                current_mode = mode if first_chunk else 'a'
                chunk.to_csv(csv_path, index=False, header=header, mode=current_mode, compression=compression)
                
                if all_chunks is not None:
                    all_chunks.append(chunk)
                
                pbar.update(len(chunk))
                first_chunk = False

        full_df = None
        if all_chunks:
            full_df = pd.concat(all_chunks)
        elif (args.parquet or args.validate or args.stats):
            full_df = pd.read_csv(csv_path, header=None, names=['epoch', 'quote'], compression=compression)

        if args.parquet and full_df is not None:
            try:
                parquet_path = csv_path.replace('.csv', '').replace('.gz', '') + '.parquet'
                full_df.to_parquet(parquet_path, index=False)
                print(f"Exported to {parquet_path}")
            except ImportError:
                print("Error: 'pyarrow' or 'fastparquet' is required for Parquet export.")

        if args.validate and full_df is not None:
            print("Validating data integrity...")
            validate_data(full_df)

        if args.stats and full_df is not None:
            print_stats(full_df)

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
