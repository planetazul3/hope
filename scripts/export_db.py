import sqlite3
import csv
import sys

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
    export_ticks("data/tick_store.db", "data/ticks.csv")
