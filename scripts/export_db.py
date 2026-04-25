import sqlite3
import csv
import sys
import argparse

def export_ticks(db_path, csv_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT epoch, quote FROM ticks ORDER BY epoch ASC")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(['epoch', 'quote']) # No header for simpler Rust parsing
            for row in cursor:
                writer.writerow(row)
        print(f"Exported ticks to {csv_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Deriv ticks from SQLite to CSV")
    parser.add_argument("--db", default="data/tick_store.db", help="SQLite DB path")
    parser.add_argument("--csv", default="data/ticks.csv", help="Output CSV path")
    args = parser.parse_args()
    
    export_ticks(args.db, args.csv)
