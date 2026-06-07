import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threat_logs.db")

def init_db():
    """Initializes the SQLite database and creates the threat_logs table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threat_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            source_dataset TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            execution_time REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_threat(source_dataset: str, predicted_class: str, confidence_score: float, execution_time: float):
    """Inserts a new threat detection entry into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO threat_logs (timestamp, source_dataset, predicted_class, confidence_score, execution_time)
        VALUES (?, ?, ?, ?, ?)
    """, (now_str, source_dataset, predicted_class, float(confidence_score), float(execution_time)))
    conn.commit()
    conn.close()
    print(f"[DB LOGGER] Event logged: Class={predicted_class}, Conf={confidence_score:.4f}, ExecTime={execution_time:.4f}s")

def get_logs():
    """Retrieves all logged threat rows from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM threat_logs ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Database threat_logs initialized successfully at:", DB_PATH)
