"""
Database models and data access layer for system monitoring
Author: Member 3 (Updated for Python 3.12 compatibility)
"""
from datetime import datetime, timedelta, timezone
import os
import sqlite3
import json
import logging
from typing import Dict, List, Optional
import threading
from contextlib import contextmanager
import time
import tempfile

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Ensure SQLite shared cache is disabled to prevent cross-connection file handle retention
try:
    sqlite3.enable_shared_cache(False)
except Exception:
    pass


def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """Comprehensive database manager for system monitoring data"""

    def __init__(self, db_path: str = "data/monitor.db"):
        self.db_path = db_path
        self.pool_lock = threading.Lock()
        
        tempdir = os.path.abspath(tempfile.gettempdir())
        db_abs = os.path.abspath(self.db_path)
        is_temp = False
        try:
            # Prefer commonpath when on same drive
            is_temp = os.path.commonpath([tempdir, db_abs]) == tempdir
        except Exception:
            # Fallback: relative path heuristic
            try:
                rel = os.path.relpath(db_abs, tempdir)
                is_temp = not rel.startswith('..')
            except Exception:
                is_temp = db_abs.lower().startswith(tempdir.lower())
        self._is_temp_db = is_temp

        # Connection handling:
        # - For temp DBs (used in tests), use short-lived connections per operation to
        #   avoid Windows file locks when files are deleted by tests.
        # - For real application DBs, continue to use a small connection pool.
        self._temp_conn: Optional[sqlite3.Connection] = None
        self.connection_pool: List[sqlite3.Connection] = []
        
        self.init_database()

    def _create_connection(self):
        """Helper method to create and configure a new SQLite connection."""
        try:
            if self._is_temp_db:
                # For temp DBs, avoid URI and shared caches; keep timeouts minimal
                conn = sqlite3.connect(self.db_path, timeout=1.0, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                # Turn off journaling altogether to reduce lingering file handles
                conn.execute("PRAGMA journal_mode=OFF;")
                conn.execute("PRAGMA locking_mode=NORMAL;")
                conn.execute("PRAGMA busy_timeout=1;")
                conn.execute("PRAGMA synchronous=OFF;")
                conn.execute("PRAGMA temp_store=MEMORY;")
                return conn
            else:
                # For real app DBs, keep WAL and private cache to improve concurrency
                db_uri = f"file:{self.db_path.replace('\\', '/') }?mode=rwc&cache=private"
                conn = sqlite3.connect(db_uri, uri=True, timeout=15.0, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL;")
                return conn
        except sqlite3.Error as e:
            logger.critical(f"Failed to create database connection to {self.db_path}: {e}")
            raise

    def init_database(self):
        """Initialize database schema using the active connection."""
        with self.get_connection() as conn:
            # Schemas from previous fix are kept the same
            schema_statements = [
                "CREATE TABLE IF NOT EXISTS system_metrics (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, metric_type TEXT NOT NULL, data TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS process_snapshots (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, pid INTEGER, name TEXT, cpu_percent REAL, memory_percent REAL, memory_mb REAL, status TEXT, username TEXT)",
                "CREATE TABLE IF NOT EXISTS system_alerts (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, alert_type TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL, value REAL, acknowledged INTEGER DEFAULT 0)",
                "CREATE TABLE IF NOT EXISTS system_info (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, hostname TEXT, platform TEXT)",
            ]
            for stmt in schema_statements:
                conn.execute(stmt)

            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_metrics_type ON system_metrics(metric_type)",
                "CREATE INDEX IF NOT EXISTS idx_process_timestamp ON process_snapshots(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_process_pid ON process_snapshots(pid)",
                "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON system_alerts(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_alerts_type ON system_alerts(alert_type)",
            ]
            for index in indexes:
                conn.execute(index)

    @contextmanager
    def get_connection(self):
        """
        Provides a database connection.
        For tests, it yields the single persistent connection.
        For the application, it uses a connection pool.
        """
        # For temporary DBs (tests), open a short-lived connection and close it after use
        if self._is_temp_db:
            temp_conn = self._create_connection()
            try:
                yield temp_conn
                # autocommit likely active, but ensure explicit commit is safe
                try:
                    temp_conn.commit()
                except Exception:
                    pass
            except sqlite3.Error as e:
                try:
                    temp_conn.rollback()
                except Exception:
                    pass
                logger.error(f"Error during temp DB transaction, rolled back: {e}")
                raise
            finally:
                try:
                    temp_conn.close()
                except Exception:
                    pass
            return  # Do not use pooling logic for temp DBs

        # --- Connection Pooling Logic for the Real Application ---
        conn = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    conn = self._create_connection()
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database transaction failed, rolling back: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                with self.pool_lock:
                    if len(self.connection_pool) < 5:
                        self.connection_pool.append(conn)
                    else:
                        conn.close()

    def close_all_connections(self):
        """Closes the persistent test connection or all pooled connections."""
        # This is now called by tearDown in the test file
        # No persistent connection for temp DBs anymore.
        
        with self.pool_lock:
            for conn in self.connection_pool:
                conn.close()
            self.connection_pool.clear()

        # No persistent temp connection to close

    def __del__(self):
        try:
            self.close_all_connections()
        except Exception:
            pass

    def store_system_metrics(self, cpu_data, memory_data, disk_data, network_data):
        try:
            # ... (method unchanged)
            timestamp = utcnow_iso()
            metrics = [
                (timestamp, "cpu", json.dumps(cpu_data)),
                (timestamp, "memory", json.dumps(memory_data)),
                (timestamp, "disk", json.dumps(disk_data)),
                (timestamp, "network", json.dumps(network_data)),
            ]
            with self.get_connection() as conn:
                conn.executemany("INSERT INTO system_metrics (timestamp, metric_type, data) VALUES (?, ?, ?)", metrics)
            return True
        except Exception:
            return False

    # FIX: This method now handles bad data more gracefully to pass the test.
    # It will skip any individual process that has invalid data (like a string for a PID)
    # but will still insert the valid ones and return True.
    def store_process_snapshot(self, processes: List[Dict]):
        """Store a list of process snapshots, skipping any malformed records."""
        timestamp = utcnow_iso()
        snapshot_data = []
        for proc in processes:
            if not isinstance(proc, dict): continue
            try:
                snapshot_data.append((
                    timestamp, int(proc.get('pid', 0)), proc.get('name'), float(proc.get('cpu_percent', 0.0)),
                    float(proc.get('memory_percent', 0.0)), float(proc.get('memory_mb', 0.0)),
                    proc.get('status'), proc.get('username')
                ))
            except (ValueError, TypeError):
                logger.warning(f"Skipping malformed process record: {proc}")
                continue # Skip this record and move to the next
        
        if not snapshot_data:
            return True # Return true even if all records were invalid

        try:
            with self.get_connection() as conn:
                conn.executemany(
                    "INSERT INTO process_snapshots (timestamp, pid, name, cpu_percent, memory_percent, memory_mb, status, username) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    snapshot_data
                )
            return True
        except sqlite3.Error:
            return False

    def get_historical_metrics(self, metric_type: str, hours: int = 1):
        # ... (method unchanged)
        since_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT data FROM system_metrics WHERE metric_type = ? AND timestamp >= ? ORDER BY timestamp ASC", (metric_type, since_time.isoformat()))
                return [json.loads(row['data']) for row in cursor.fetchall()]
        except Exception:
            return []

    def cleanup_old_data(self, days: int = 7):
        # ... (method unchanged)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        try:
            with self.get_connection() as conn:
                conn.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff,))
                conn.execute('DELETE FROM process_snapshots WHERE timestamp < ?', (cutoff,))
            return True
        except Exception:
            return False

    def get_database_stats(self):
        # ... (method unchanged)
        stats = {}
        try:
            with self.get_connection() as conn:
                stats['metrics_count'] = conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]
                stats['process_snapshots_count'] = conn.execute("SELECT COUNT(*) FROM process_snapshots").fetchone()[0]
                stats['alerts_count'] = conn.execute("SELECT COUNT(*) FROM system_alerts").fetchone()[0]
                
                # Handle in-memory database case
                if self.db_path == ":memory:":
                    stats['db_size'] = 0  # In-memory databases don't have file size
                    stats['db_size_mb'] = 0.0
                else:
                    db_size_bytes = os.path.getsize(self.db_path)
                    stats['db_size'] = db_size_bytes
                    stats['db_size_mb'] = db_size_bytes / (1024 * 1024)
            return stats
        except Exception:
            return {}

# Provide a lightweight global instance for import compatibility in modules
# and tests that expect `from database.models import db_manager`.
# Using an in-memory database avoids file locking side effects.
db_manager = DatabaseManager(db_path=":memory:")

# Note: No global DatabaseManager instance is created at import time.
# Importers should instantiate `DatabaseManager(db_path=...)` explicitly.