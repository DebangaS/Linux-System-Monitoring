"""
Advanced Database Models and Architecture
Author: Member 3
"""
import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class DatabaseConfig:
    """Database configuration parameters"""
    db_path: str
    pool_size: int = 20
    timeout: int = 30
    wal_mode: bool = True
    cache_size: int = -64000  # 64MB (negative uses pages in SQLite)
    temp_store: str = "MEMORY"
    synchronous: str = "NORMAL"
    journal_mode: str = "WAL"
    foreign_keys: bool = True


class ConnectionPool:
    """Thread-safe database connection pool"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._used_connections = set()

        # Ensure DB directory exists
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize pool
        for _ in range(config.pool_size):
            conn = self._create_connection()
            self._pool.append(conn)

        logger.info(f"Connection pool initialized with {config.pool_size} connections")

    def _create_connection(self) -> sqlite3.Connection:
        """Create optimized database connection"""
        conn = sqlite3.connect(
            self.config.db_path,
            timeout=self.config.timeout,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )

        # Optimize connection settings
        conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
        conn.execute(f"PRAGMA synchronous = {self.config.synchronous}")
        conn.execute(f"PRAGMA cache_size = {self.config.cache_size}")
        conn.execute(f"PRAGMA temp_store = {self.config.temp_store}")
        fk = 'ON' if self.config.foreign_keys else 'OFF'
        conn.execute(f"PRAGMA foreign_keys = {fk}")
        # Helpful optimizations
        conn.execute("PRAGMA automatic_index = ON")
        conn.execute("PRAGMA optimize")

        return conn

    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager"""
        conn = None
        try:
            with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                    self._used_connections.add(conn)
                else:
                    # Pool exhausted, create new connection
                    conn = self._create_connection()
                    self._used_connections.add(conn)
                    logger.warning("Connection pool exhausted, created new connection")

            yield conn

        finally:
            if conn:
                try:
                    # Rollback any uncommitted transactions
                    try:
                        conn.rollback()
                    except sqlite3.OperationalError:
                        # No active transaction or DB closed
                        pass

                    with self._lock:
                        self._used_connections.discard(conn)
                        if len(self._pool) < self.config.pool_size:
                            self._pool.append(conn)
                        else:
                            conn.close()
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()
                    except Exception:
                        pass


class AdvancedDatabaseManager:
    """Advanced database manager with partitioning and optimization"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = ConnectionPool(config)

        # Partitioning settings
        self.partition_enabled = True
        self.partition_interval = 'month'  # 'day', 'week', 'month'

        # Cache settings
        self.query_cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = 300  # 5 minutes

        # Initialize schema
        self._initialize_advanced_schema()

        logger.info("Advanced database manager initialized")

    def get_database_stats(self) -> Dict:
        """Provide lightweight stats compatible with tests"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                stats: Dict[str, Any] = {}

                # Count rows from current/historical views as system_metrics equivalent
                try:
                    cursor.execute("SELECT COUNT(*) FROM v_all_metrics")
                    stats['metrics_count'] = cursor.fetchone()[0]
                except Exception:
                    stats['metrics_count'] = 0

                # We don't maintain process_snapshots here; keep zero
                stats['process_snapshots_count'] = 0

                # Alerts and others are part of advanced schema, may be absent
                try:
                    cursor.execute("SELECT COUNT(*) FROM system_events")
                    stats['alerts_count'] = cursor.fetchone()[0]
                except Exception:
                    stats['alerts_count'] = 0

                # DB size
                try:
                    size_bytes = Path(self.config.db_path).stat().st_size
                except Exception:
                    size_bytes = 0
                stats['db_size'] = size_bytes
                stats['db_size_mb'] = size_bytes / (1024 * 1024)

                return stats
        except Exception as e:
            logger.error(f"Error getting advanced database stats: {e}")
            return {}

    def _initialize_advanced_schema(self):
        """Initialize advanced database schema with partitioning"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # Create partitioned metrics tables
            self._create_partitioned_tables(cursor)

            # Create analytics tables
            self._create_analytics_tables(cursor)

            # Create indexes
            self._create_advanced_indexes(cursor)

            # Create views
            self._create_analytical_views(cursor)

            # Create triggers
            self._create_triggers(cursor)

            conn.commit()

    def _create_partitioned_tables(self, cursor: sqlite3.Cursor):
        """Create partitioned tables for better performance"""
        # Current metrics table (hot data - last 7 days)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics_current (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                metric_type VARCHAR(50) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                value REAL NOT NULL,
                metadata TEXT,
                source VARCHAR(50) DEFAULT 'system',
                tags TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT check_current_data CHECK (
                    datetime(timestamp) >= datetime('now', '-7 days')
                )
            )
        ''')

        # Historical metrics table (warm data - older than 7 days)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics_historical (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                metric_type VARCHAR(50) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                value REAL NOT NULL,
                metadata TEXT,
                source VARCHAR(50) DEFAULT 'system',
                tags TEXT,
                archived_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT check_historical_data CHECK (
                    datetime(timestamp) < datetime('now', '-7 days')
                )
            )
        ''')

        # Aggregated metrics table (summary data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics_aggregated (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_bucket DATETIME NOT NULL,
                metric_type VARCHAR(50) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                aggregation_level VARCHAR(20) NOT NULL, -- 'hour', 'day', 'week', 'month'
                min_value REAL,
                max_value REAL,
                avg_value REAL,
                sum_value REAL,
                count_value INTEGER,
                std_dev REAL,
                percentile_95 REAL,
                percentile_99 REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(time_bucket, metric_type, metric_name, aggregation_level)
            )
        ''')

        # Advanced process metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS process_metrics_advanced (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                pid INTEGER NOT NULL,
                name VARCHAR(255) NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                memory_rss INTEGER,
                memory_vms INTEGER,
                io_read_bytes INTEGER,
                io_write_bytes INTEGER,
                num_threads INTEGER,
                num_fds INTEGER,
                create_time REAL,
                status VARCHAR(50),
                username VARCHAR(100),
                command_line TEXT,
                parent_pid INTEGER,
                children_count INTEGER,
                cpu_times_user REAL,
                cpu_times_system REAL,
                memory_full_info TEXT, -- JSON with detailed memory info
                io_counters TEXT, -- JSON with detailed I/O counters
                connections_count INTEGER,
                environment_vars TEXT -- JSON with environment variables
            )
        ''')

        # System events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL, -- 'info', 'warning', 'error', 'critical'
                source VARCHAR(100) NOT NULL,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                details TEXT, -- JSON with detailed information
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at DATETIME,
                resolved_by VARCHAR(100),
                tags TEXT, -- JSON array of tags
                impact_score INTEGER DEFAULT 0, -- 0-100 impact score
                related_metrics TEXT, -- JSON array of related metric names
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        logger.info("Partitioned tables created")

    def _create_analytics_tables(self, cursor: sqlite3.Cursor):
        """Create tables for advanced analytics"""
        # Performance baselines
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name VARCHAR(100) NOT NULL,
                baseline_type VARCHAR(50) NOT NULL, -- 'hourly', 'daily', 'weekly'
                time_period VARCHAR(50) NOT NULL, -- '00:00-01:00', 'monday', etc.
                min_value REAL,
                max_value REAL,
                avg_value REAL,
                std_dev REAL,
                percentile_50 REAL,
                percentile_90 REAL,
                percentile_95 REAL,
                percentile_99 REAL,
                sample_count INTEGER,
                confidence_level REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(metric_name, baseline_type, time_period)
            )
        ''')

        # Anomaly detection results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomaly_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                value REAL NOT NULL,
                expected_value REAL,
                anomaly_score REAL NOT NULL,
                anomaly_type VARCHAR(50), -- 'statistical', 'ml', 'threshold'
                severity VARCHAR(20) NOT NULL,
                details TEXT, -- JSON with detection details
                baseline_id INTEGER,
                confirmed BOOLEAN DEFAULT NULL, -- NULL=unconfirmed, TRUE=confirmed, FALSE=unconfirmed_false
                confirmed_by VARCHAR(100),
                confirmed_at DATETIME,
                FOREIGN KEY (baseline_id) REFERENCES performance_baselines(id)
            )
        ''')

        # Capacity planning data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capacity_planning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_type VARCHAR(50) NOT NULL, -- 'cpu', 'memory', 'disk', 'network'
                current_capacity REAL NOT NULL,
                current_utilization REAL NOT NULL,
                projected_utilization REAL,
                projection_date DATETIME,
                growth_rate REAL, -- percentage per month
                time_to_exhaustion INTEGER, -- days until 100% utilization
                recommended_action TEXT,
                confidence_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Business metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                metric_category VARCHAR(50) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                value REAL NOT NULL,
                unit VARCHAR(20),
                target_value REAL,
                threshold_warning REAL,
                threshold_critical REAL,
                business_impact TEXT,
                calculation_method TEXT,
                data_sources TEXT, -- JSON array of source tables/metrics
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        logger.info("Analytics tables created")

    def _create_advanced_indexes(self, cursor: sqlite3.Cursor):
        """Create advanced indexes for optimal query performance"""
        indexes = [
            # Current metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_metrics_current_timestamp ON metrics_current(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_current_type_name ON metrics_current(metric_type, metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_current_metric_name ON metrics_current(metric_name)",

            # Historical metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_metrics_historical_timestamp ON metrics_historical(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_historical_type ON metrics_historical(metric_type)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_historical_archived ON metrics_historical(archived_at)",

            # Aggregated metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_metrics_agg_bucket ON metrics_aggregated(time_bucket)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_agg_level ON metrics_aggregated(aggregation_level)",

            # Process metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_process_adv_timestamp ON process_metrics_advanced(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_process_adv_pid ON process_metrics_advanced(pid)",
            "CREATE INDEX IF NOT EXISTS idx_process_adv_name ON process_metrics_advanced(name)",
            "CREATE INDEX IF NOT EXISTS idx_process_adv_cpu ON process_metrics_advanced(cpu_percent)",
            "CREATE INDEX IF NOT EXISTS idx_process_adv_memory ON process_metrics_advanced(memory_percent)",

            # System events indexes
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_events_type ON system_events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_events_severity ON system_events(severity)",
            "CREATE INDEX IF NOT EXISTS idx_events_resolved ON system_events(resolved)",

            # Analytics indexes
            "CREATE INDEX IF NOT EXISTS idx_baselines_metric ON performance_baselines(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomaly_detections(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_anomalies_metric ON anomaly_detections(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomaly_detections(severity)",

            # Capacity planning indexes
            "CREATE INDEX IF NOT EXISTS idx_capacity_resource ON capacity_planning(resource_type)",
            "CREATE INDEX IF NOT EXISTS idx_capacity_utilization ON capacity_planning(current_utilization)",

            # Business metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_business_timestamp ON business_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_business_category ON business_metrics(metric_category)"
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error as e:
                logger.error(f"Error creating index: {e}")

        logger.info("Advanced indexes created")

    def _create_analytical_views(self, cursor: sqlite3.Cursor):
        """Create views for complex analytical queries"""
        # View for all metrics (current + historical)
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_all_metrics AS
            SELECT
                timestamp, metric_type, metric_name, value, metadata, source, tags,
                'current' as partition_type
            FROM metrics_current
            UNION ALL
            SELECT
                timestamp, metric_type, metric_name, value, metadata, source, tags,
                'historical' as partition_type
            FROM metrics_historical
        ''')

        # View for recent system health
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_system_health_summary AS
            SELECT
                metric_type,
                COUNT(*) as sample_count,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                datetime('now') as generated_at
            FROM metrics_current
            WHERE timestamp >= datetime('now', '-1 hour')
            GROUP BY metric_type
        ''')

        # View for top resource consumers
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_top_resource_consumers AS
            SELECT
                name,
                pid,
                AVG(cpu_percent) as avg_cpu,
                AVG(memory_percent) as avg_memory,
                MAX(memory_rss) as peak_memory_rss,
                COUNT(*) as sample_count
            FROM process_metrics_advanced
            WHERE timestamp >= datetime('now', '-1 hour')
            GROUP BY name, pid
            HAVING sample_count > 5
            ORDER BY avg_cpu DESC, avg_memory DESC
        ''')

        # View for system events summary
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_events_summary AS
            SELECT
                event_type,
                severity,
                COUNT(*) as event_count,
                COUNT(CASE WHEN resolved = 1 THEN 1 END) as resolved_count,
                AVG(impact_score) as avg_impact_score,
                MAX(timestamp) as latest_event
            FROM system_events
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY event_type, severity
            ORDER BY avg_impact_score DESC, event_count DESC
        ''')

        logger.info("Analytical views created")

    def _create_triggers(self, cursor: sqlite3.Cursor):
        """Create triggers for data management"""

        # Trigger to move old data from current to historical only when a new row older than 7 days is inserted.
        # Use NEW.* to insert the single NEW row if it's old.
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS tr_archive_old_metric_row
            AFTER INSERT ON metrics_current
            WHEN datetime(NEW.timestamp) < datetime('now', '-7 days')
            BEGIN
                INSERT INTO metrics_historical (timestamp, metric_type, metric_name, value, metadata, source, tags)
                VALUES (NEW.timestamp, NEW.metric_type, NEW.metric_name, NEW.value, NEW.metadata, NEW.source, NEW.tags);

                DELETE FROM metrics_current WHERE id = NEW.id;
            END;
        ''')

        # Trigger to update business metrics - create or replace a latest business metric row.
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS tr_update_business_metrics
            AFTER INSERT ON metrics_current
            WHEN NEW.metric_type IN ('cpu', 'memory', 'disk')
            BEGIN
                INSERT INTO business_metrics (
                    timestamp, metric_category, metric_name, value, unit, created_at
                ) VALUES (
                    NEW.timestamp, 'performance',
                    NEW.metric_type || '_utilization', NEW.value, 'percent', CURRENT_TIMESTAMP
                );
            END;
        ''')

        logger.info("Triggers created")

    def store_advanced_metric(
        self,
        metric_type: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict] = None,
        source: str = 'system',
        tags: Optional[List[str]] = None
    ) -> bool:
        """Store metric with advanced features"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                timestamp = datetime.utcnow().isoformat(sep=' ')

                cursor.execute('''
                    INSERT INTO metrics_current
                    (timestamp, metric_type, metric_name, value, metadata, source, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    metric_type,
                    metric_name,
                    value,
                    json.dumps(metadata) if metadata else None,
                    source,
                    json.dumps(tags) if tags else None
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing advanced metric: {e}")
            return False

    def store_process_advanced(self, process_data: Dict) -> bool:
        """Store advanced process metrics"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO process_metrics_advanced
                    (timestamp, pid, name, cpu_percent, memory_percent, memory_rss,
                     memory_vms, io_read_bytes, io_write_bytes, num_threads, num_fds,
                     create_time, status, username, command_line, parent_pid,
                     children_count, cpu_times_user, cpu_times_system,
                     memory_full_info, io_counters, connections_count, environment_vars)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.utcnow().isoformat(sep=' '),
                    process_data.get('pid'),
                    process_data.get('name'),
                    process_data.get('cpu_percent'),
                    process_data.get('memory_percent'),
                    process_data.get('memory_rss'),
                    process_data.get('memory_vms'),
                    process_data.get('io_read_bytes'),
                    process_data.get('io_write_bytes'),
                    process_data.get('num_threads'),
                    process_data.get('num_fds'),
                    process_data.get('create_time'),
                    process_data.get('status'),
                    process_data.get('username'),
                    process_data.get('command_line'),
                    process_data.get('parent_pid'),
                    process_data.get('children_count'),
                    process_data.get('cpu_times_user'),
                    process_data.get('cpu_times_system'),
                    json.dumps(process_data.get('memory_full_info')) if process_data.get('memory_full_info') is not None else None,
                    json.dumps(process_data.get('io_counters')) if process_data.get('io_counters') is not None else None,
                    process_data.get('connections_count'),
                    json.dumps(process_data.get('environment_vars')) if process_data.get('environment_vars') is not None else None
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing advanced process data: {e}")
            return False

    def create_system_event(
        self,
        event_type: str,
        severity: str,
        source: str,
        title: str,
        description: Optional[str] = None,
        details: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        impact_score: int = 0,
        related_metrics: Optional[List[str]] = None
    ) -> int:
        """Create system event record"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO system_events
                    (timestamp, event_type, severity, source, title, description,
                     details, tags, impact_score, related_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.utcnow().isoformat(sep=' '),
                    event_type,
                    severity,
                    source,
                    title,
                    description,
                    json.dumps(details) if details else None,
                    json.dumps(tags) if tags else None,
                    impact_score,
                    json.dumps(related_metrics) if related_metrics else None
                ))

                event_id = cursor.lastrowid
                conn.commit()

                logger.info(f"System event created: {event_id} - {title}")
                return event_id

        except Exception as e:
            logger.error(f"Error creating system event: {e}")
            return 0

    def get_cached_query(self, cache_key: str, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute query with caching"""
        # Check cache
        cached = self.query_cache.get(cache_key)
        if cached:
            cached_data, cached_time = cached
            if time.time() - cached_time < self.cache_ttl:
                return cached_data

        # Execute query
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

                columns = [description[0] for description in cursor.description] if cursor.description else []
                results: List[Dict[str, Any]] = []

                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row))
                    results.append(row_dict)

                # Cache results
                self.query_cache[cache_key] = (results, time.time())

                return results

        except Exception as e:
            logger.error(f"Error executing cached query: {e}")
            return []

    def analyze_query_performance(self, query: str, params: Tuple = ()) -> Dict:
        """Analyze query performance"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                # Get query plan
                explain_query = f"EXPLAIN QUERY PLAN {query}"
                cursor.execute(explain_query, params)
                query_plan = cursor.fetchall()

                # Execute query with timing
                start_time = time.time()
                cursor.execute(query, params)
                results = cursor.fetchall()
                execution_time = time.time() - start_time

                return {
                    'execution_time_ms': execution_time * 1000,
                    'rows_returned': len(results),
                    'query_plan': [
                        {
                            'id': plan[0],
                            'parent': plan[1],
                            'notused': plan[2],
                            'detail': plan[3]
                        } for plan in query_plan
                    ]
                }

        except Exception as e:
            logger.error(f"Query performance analysis error: {e}")
            return {'error': str(e)}


# Global advanced database manager
advanced_db: Optional[AdvancedDatabaseManager] = None


def get_advanced_db_manager() -> AdvancedDatabaseManager:
    """Get global advanced database manager instance"""
    global advanced_db
    if advanced_db is None:
        config = DatabaseConfig(
            db_path='data/advanced_system_monitor.db',
            pool_size=20,
            timeout=30
        )
        advanced_db = AdvancedDatabaseManager(config)
    return advanced_db
