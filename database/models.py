"""
Database models and data access layer for system monitoring
Author: Member 3 
"""
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
import os
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DatabaseManager:
    """Comprehensive database manager for system monitoring data"""

    def __init__(self, db_path: str = "data/monitor.db"):
        self.db_path = db_path
        # simple connection pooling
        self.connection_pool: List[sqlite3.Connection] = []
        self.pool_lock = threading.Lock()
        self.max_connections = 5
        # ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()

    @contextmanager
    def get_connection(self):
        """
        Context manager that yields a sqlite3 connection.
        Connections are pooled up to `max_connections`.
        Uses check_same_thread=False to allow threads to reuse connections;
        we still guard pool operations with a lock.
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    # allow access from multiple threads but be careful with transactions
                    conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    # improve WAL mode for concurrency
                    try:
                        conn.execute("PRAGMA journal_mode=WAL;")
                    except Exception:
                        pass

            yield conn

            # commit after successful usage if autocommit isn't used
            try:
                conn.commit()
            except Exception:
                # If commit fails, let outer exception handling deal with it
                pass

        except Exception as e:
            logger.exception(f"Database connection error: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if conn:
                with self.pool_lock:
                    if len(self.connection_pool) < self.max_connections:
                        self.connection_pool.append(conn)
                    else:
                        try:
                            conn.close()
                        except Exception:
                            pass

    def init_database(self):
        """Initialize database with comprehensive schema and indexes"""
        with self.get_connection() as conn:
            # System metrics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_sent REAL,
                    network_recv REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Process snapshots table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS process_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_mb REAL,
                    status TEXT,
                    username TEXT,
                    cmdline TEXT,
                    parent_pid INTEGER,
                    thread_count INTEGER,
                    data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # System alerts table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL,
                    threshold REAL,
                    acknowledged INTEGER DEFAULT 0,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # System information table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    hostname TEXT,
                    platform TEXT,
                    processor TEXT,
                    architecture TEXT,
                    boot_time TEXT,
                    uptime_seconds INTEGER,
                    total_memory INTEGER,
                    cpu_cores INTEGER,
                    users TEXT,
                    data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Data export log table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS export_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    export_type TEXT NOT NULL,
                    format TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    records_count INTEGER,
                    file_size INTEGER,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_seconds REAL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Performance optimization indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_metrics_type ON system_metrics(metric_type)",
                "CREATE INDEX IF NOT EXISTS idx_metrics_cpu ON system_metrics(cpu_usage)",
                "CREATE INDEX IF NOT EXISTS idx_metrics_memory ON system_metrics(memory_usage)",
                "CREATE INDEX IF NOT EXISTS idx_process_timestamp ON process_snapshots(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_process_pid ON process_snapshots(pid)",
                "CREATE INDEX IF NOT EXISTS idx_process_name ON process_snapshots(name)",
                "CREATE INDEX IF NOT EXISTS idx_process_cpu ON process_snapshots(cpu_percent)",
                "CREATE INDEX IF NOT EXISTS idx_process_memory ON process_snapshots(memory_percent)",
                "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON system_alerts(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_alerts_type ON system_alerts(alert_type)",
                "CREATE INDEX IF NOT EXISTS idx_alerts_level ON system_alerts(level)",
                "CREATE INDEX IF NOT EXISTS idx_alerts_ack ON system_alerts(acknowledged)",
                "CREATE INDEX IF NOT EXISTS idx_sysinfo_timestamp ON system_info(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_export_type ON export_log(export_type)",
            ]

            for index_sql in indexes:
                conn.execute(index_sql)

            conn.commit()

        logger.info("Database initialized with comprehensive schema and indexes")

    def store_system_metrics(
        self, cpu_data: Dict, memory_data: Dict, disk_data: Dict, network_data: Dict
    ) -> bool:
        """Store system metrics with extracted values for indexing"""
        try:
            timestamp = datetime.utcnow().isoformat()

            with self.get_connection() as conn:
                # Extract key values for indexing
                cpu_usage = cpu_data.get("usage_percent", None)
                memory_usage = memory_data.get("usage_percent", None)
                disk_usage = disk_data.get("total_usage_percent", None)
                network_sent = network_data.get("sent_rate_kbps", None)
                network_recv = network_data.get("recv_rate_kbps", None)

                # Store individual metric records (one per metric type)
                metrics = [
                    (
                        timestamp,
                        "cpu",
                        json.dumps(cpu_data),
                        cpu_usage,
                        None,
                        None,
                        None,
                        None,
                    ),
                    (
                        timestamp,
                        "memory",
                        json.dumps(memory_data),
                        None,
                        memory_usage,
                        None,
                        None,
                        None,
                    ),
                    (
                        timestamp,
                        "disk",
                        json.dumps(disk_data),
                        None,
                        None,
                        disk_usage,
                        None,
                        None,
                    ),
                    (
                        timestamp,
                        "network",
                        json.dumps(network_data),
                        None,
                        None,
                        None,
                        network_sent,
                        network_recv,
                    ),
                ]

                conn.executemany(
                    """
                    INSERT INTO system_metrics 
                    (timestamp, metric_type, data, cpu_usage, memory_usage, disk_usage, network_sent, network_recv)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    metrics,
                )

                # commit handled by get_connection context manager
                return True

        except Exception as e:
            logger.exception(f"Error storing system metrics: {e}")
            return False

    def store_process_snapshot(self, processes: List[Dict]) -> bool:
        """Store enhanced process snapshot"""
        try:
            timestamp = datetime.utcnow().isoformat()

            with self.get_connection() as conn:
                process_records = []
                for process in processes:
                    cmdline = ""
                    if process.get("cmdline"):
                        # cmdline may be list or string
                        if isinstance(process.get("cmdline"), (list, tuple)):
                            cmdline = " ".join(process.get("cmdline"))
                        else:
                            cmdline = str(process.get("cmdline"))

                    parent_pid = process.get("ppid") or process.get("parent_pid") or None

                    record = (
                        timestamp,
                        process.get("pid"),
                        process.get("name"),
                        process.get("cpu_percent"),
                        process.get("memory_percent"),
                        process.get("memory_mb"),
                        process.get("status"),
                        process.get("username"),
                        cmdline,
                        parent_pid,
                        process.get("num_threads"),
                        json.dumps(process),
                    )
                    process_records.append(record)

                if process_records:
                    conn.executemany(
                        """
                        INSERT INTO process_snapshots 
                        (timestamp, pid, name, cpu_percent, memory_percent, memory_mb, status, 
                        username, cmdline, parent_pid, thread_count, data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        process_records,
                    )

                return True

        except Exception as e:
            logger.exception(f"Error storing process snapshot: {e}")
            return False

    def store_system_alert(
        self,
        alert_type: str,
        level: str,
        message: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """Store enhanced system alert"""
        try:
            timestamp = datetime.utcnow().isoformat()

            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO system_alerts 
                    (timestamp, alert_type, level, message, value, threshold)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, alert_type, level, message, value, threshold),
                )

                return True

        except Exception as e:
            logger.exception(f"Error storing system alert: {e}")
            return False

    def store_system_info(self, system_info: Dict) -> bool:
        """Store enhanced system information"""
        try:
            timestamp = datetime.utcnow().isoformat()

            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO system_info 
                    (timestamp, hostname, platform, processor, architecture, boot_time, 
                     uptime_seconds, total_memory, cpu_cores, users, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        system_info.get("hostname"),
                        system_info.get("platform"),
                        system_info.get("processor"),
                        system_info.get("architecture"),
                        system_info.get("boot_time"),
                        system_info.get("uptime_seconds"),
                        system_info.get("total_memory"),
                        system_info.get("cpu_cores"),
                        json.dumps(system_info.get("users", [])),
                        json.dumps(system_info),
                    ),
                )

                return True

        except Exception as e:
            logger.exception(f"Error storing system info: {e}")
            return False

    def get_historical_metrics(self, metric_type: str, hours: int = 24, limit: int = 1000) -> List[Dict]:
        """Get historical metrics with performance optimization"""
        try:
            start_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT timestamp, data FROM system_metrics 
                    WHERE metric_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (metric_type, start_time, limit),
                )

                results = []
                for row in cursor.fetchall():
                    try:
                        data = json.loads(row["data"])
                        data["timestamp"] = row["timestamp"]
                        results.append(data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON data in database: {row['data'][:100]}")
                        continue

                # Return in chronological order (oldest first)
                return list(reversed(results))

        except Exception as e:
            logger.exception(f"Error getting historical metrics: {e}")
            return []

    def get_process_history(
        self, pid: Optional[int] = None, hours: int = 24, limit: int = 1000
    ) -> List[Dict]:
        """Get enhanced process history"""
        try:
            start_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            with self.get_connection() as conn:
                if pid:
                    cursor = conn.execute(
                        """
                        SELECT * FROM process_snapshots 
                        WHERE pid = ? AND timestamp >= ?
                        ORDER BY timestamp DESC LIMIT ?
                        """,
                        (pid, start_time, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM process_snapshots 
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC LIMIT ?
                        """,
                        (start_time, limit),
                    )

                results = []
                for row in cursor.fetchall():
                    record = dict(row)
                    try:
                        if record.get("data"):
                            detailed_data = json.loads(record["data"])
                            if isinstance(detailed_data, dict):
                                record.update(detailed_data)
                    except json.JSONDecodeError:
                        pass
                    results.append(record)

                return list(reversed(results))

        except Exception as e:
            logger.exception(f"Error getting process history: {e}")
            return []

    def get_system_alerts(
        self, hours: int = 24, acknowledged: bool = False, limit: int = 100, alert_type: Optional[str] = None
    ) -> List[Dict]:
        """Get enhanced system alerts with filtering"""
        try:
            start_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            with self.get_connection() as conn:
                query = "SELECT * FROM system_alerts WHERE timestamp >= ?"
                params: List = [start_time]

                if not acknowledged:
                    query += " AND acknowledged = 0"

                if alert_type:
                    query += " AND alert_type = ?"
                    params.append(alert_type)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)

                results = [dict(row) for row in cursor.fetchall()]

                return results

        except Exception as e:
            logger.exception(f"Error getting system alerts: {e}")
            return []

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert with user tracking"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    UPDATE system_alerts 
                    SET acknowledged = 1, acknowledged_at = ?, acknowledged_by = ?
                    WHERE id = ?
                    """,
                    (datetime.utcnow().isoformat(), acknowledged_by, alert_id),
                )
                return cursor.rowcount > 0

        except Exception as e:
            logger.exception(f"Error acknowledging alert: {e}")
            return False

    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                stats: Dict = {}

                # Table counts
                tables = ["system_metrics", "process_snapshots", "system_alerts", "system_info", "export_log"]
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]

                # Date ranges
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM system_metrics")
                min_max = cursor.fetchone()
                min_ts, max_ts = min_max if min_max else (None, None)
                stats["data_range"] = {"earliest": min_ts, "latest": max_ts}

                # Recent activity
                one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
                cursor.execute("SELECT COUNT(*) FROM system_metrics WHERE timestamp >= ?", (one_hour_ago,))
                stats["recent_metrics_count"] = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM system_alerts WHERE timestamp >= ? AND acknowledged = 0", (one_hour_ago,)
                )
                stats["unacknowledged_alerts"] = cursor.fetchone()[0]

                # Database file size
                if os.path.exists(self.db_path):
                    stats["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)
                else:
                    stats["db_size_mb"] = 0

                # Index usage statistics (SQLite specific)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                indexes = [row[0] for row in cursor.fetchall()]
                stats["index_count"] = len(indexes)

                return stats

        except Exception as e:
            logger.exception(f"Error getting database stats: {e}")
            return {}

    def _batch_delete_table_by_timestamp(self, conn: sqlite3.Connection, table: str, cutoff_time: str, batch_size: int = 1000):
        """
        Helper to delete rows in batches on SQLite (which doesn't support DELETE ... LIMIT).
        We select rowids first, then delete them in a single statement.
        """
        while True:
            sel = conn.execute(
                f"SELECT rowid FROM {table} WHERE timestamp < ? LIMIT ?", (cutoff_time, batch_size)
            )
            rowids = [str(r[0]) for r in sel.fetchall()]
            if not rowids:
                break
            # protect against SQL injection by ensuring only integers in rowids
            placeholders = ",".join(["?"] * len(rowids))
            conn.execute(f"DELETE FROM {table} WHERE rowid IN ({placeholders})", rowids)
            # continue until fewer than batch_size removed
            if len(rowids) < batch_size:
                break

    def cleanup_old_data(self, days_to_keep: int = 7) -> bool:
        """Clean up old data with performance optimization"""
        try:
            cutoff_time = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()

            with self.get_connection() as conn:
                batch_size = 1000

                # Clean up old metrics in batches
                self._batch_delete_table_by_timestamp(conn, "system_metrics", cutoff_time, batch_size)

                # Clean up old process snapshots in batches
                self._batch_delete_table_by_timestamp(conn, "process_snapshots", cutoff_time, batch_size)

                # Clean up old acknowledged alerts (no batch limit necessary, but we can reuse helper with a query)
                # We'll delete acknowledged alerts older than cutoff_time
                cursor = conn.execute(
                    "DELETE FROM system_alerts WHERE timestamp < ? AND acknowledged = 1", (cutoff_time,)
                )
                alerts_deleted = cursor.rowcount if cursor is not None else 0

                # Vacuum database to reclaim space
                try:
                    conn.execute("VACUUM")
                except Exception:
                    # VACUUM can be heavy and may fail on some setups; ignore errors
                    pass

                logger.info(f"Cleaned up data older than {days_to_keep} days, deleted acknowledged alerts: {alerts_deleted}")
                return True

        except Exception as e:
            logger.exception(f"Error cleaning up old data: {e}")
            return False

    def export_data_to_file(self, table_name: str, output_file: str, format_type: str = "csv", hours: int = 24) -> bool:
        """Export data with logging. Uses pandas if available, otherwise falls back to CSV writer."""
        try:
            import time
            start_time = datetime.utcnow()
            start_timestamp = (start_time - timedelta(hours=hours)).isoformat()

            with self.get_connection() as conn:
                # Get data based on table
                if table_name == "system_metrics":
                    query = """
                        SELECT timestamp, metric_type, cpu_usage, memory_usage, 
                               disk_usage, network_sent, network_recv 
                        FROM system_metrics 
                        WHERE timestamp >= ? ORDER BY timestamp
                    """
                elif table_name == "process_snapshots":
                    query = """
                        SELECT timestamp, pid, name, cpu_percent, memory_percent, 
                               memory_mb, status, username, cmdline 
                        FROM process_snapshots 
                        WHERE timestamp >= ? ORDER BY timestamp
                    """
                elif table_name == "system_alerts":
                    query = """
                        SELECT timestamp, alert_type, level, message, value, 
                               threshold, acknowledged 
                        FROM system_alerts 
                        WHERE timestamp >= ? ORDER BY timestamp
                    """
                else:
                    logger.error("Unsupported table for export: %s", table_name)
                    return False

                # Try to use pandas for convenient export; fall back if pandas not installed
                try:
                    import pandas as pd

                    df = pd.read_sql_query(query, conn, params=[start_timestamp])
                    records_count = len(df)

                    if format_type.lower() == "csv":
                        df.to_csv(output_file, index=False)
                    elif format_type.lower() == "json":
                        df.to_json(output_file, orient="records", date_format="iso")
                    elif format_type.lower() == "excel":
                        df.to_excel(output_file, index=False)
                    else:
                        logger.error("Unsupported export format: %s", format_type)
                        return False
                except Exception as pd_exc:
                    # fallback: manual CSV using sqlite cursor
                    cursor = conn.execute(query, (start_timestamp,))
                    rows = cursor.fetchall()
                    columns = [d[0] for d in cursor.description] if cursor.description else []
                    records_count = len(rows)

                    if format_type.lower() == "csv":
                        import csv

                        with open(output_file, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if columns:
                                writer.writerow(columns)
                            for r in rows:
                                writer.writerow([r[c] for c in columns])
                    elif format_type.lower() == "json":
                        out = []
                        for r in rows:
                            obj = {}
                            for idx, col in enumerate(columns):
                                obj[col] = r[idx]
                            out.append(obj)
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(out, f, default=str)
                    else:
                        logger.error("Unsupported export format (no pandas): %s", format_type)
                        return False

                # Log export
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0

                conn.execute(
                    """
                    INSERT INTO export_log 
                    (export_type, format, file_path, records_count, file_size, 
                     start_time, end_time, duration_seconds, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (table_name, format_type, output_file, records_count, file_size, start_time.isoformat(), end_time.isoformat(), duration, "success"),
                )

                logger.info(f"Exported {records_count} records to {output_file} in {duration:.2f}s")
                return True

        except Exception as e:
            logger.exception(f"Error exporting data: {e}")
            # Log failed export attempt
            try:
                with self.get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO export_log 
                        (export_type, format, file_path, start_time, end_time, status, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            table_name,
                            format_type,
                            output_file,
                            datetime.utcnow().isoformat(),
                            datetime.utcnow().isoformat(),
                            "failed",
                            str(e),
                        ),
                    )
            except Exception:
                pass
            return False


# Global database manager instance
db_manager = DatabaseManager()
