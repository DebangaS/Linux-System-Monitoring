"""
Performance monitoring and optimization utilities
Author: Member 5
"""

import time
import functools
import threading
import logging
from typing import Dict, List, Callable, Any
from datetime import datetime
import psutil
import sqlite3

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and analyze application performance"""

    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
        self.start_time = time.time()

    def timing_decorator(self, func_name: str = None):
        """Decorator to measure function execution time"""

        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self.record_timing(name, execution_time, success=True)
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.record_timing(name, execution_time, success=False, error=str(e))
                    raise

            return wrapper

        return decorator

    def record_timing(self, func_name: str, execution_time: float, success: bool = True, error: str = None):
        """Record timing information for a function"""
        with self.lock:
            if func_name not in self.metrics:
                self.metrics[func_name] = {
                    'call_count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'success_count': 0,
                    'error_count': 0,
                    'recent_calls': []
                }
            metric = self.metrics[func_name]
            metric['call_count'] += 1
            metric['total_time'] += execution_time
            metric['min_time'] = min(metric['min_time'], execution_time)
            metric['max_time'] = max(metric['max_time'], execution_time)

            if success:
                metric['success_count'] += 1
            else:
                metric['error_count'] += 1
                logger.error(f"Function {func_name} failed: {error}")

            metric['recent_calls'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'execution_time': execution_time,
                'success': success,
            })

            # Limit recent calls to the last 100
            if len(metric['recent_calls']) > 100:
                metric['recent_calls'] = metric['recent_calls'][-100:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            summary = {
                'monitoring_duration_seconds': time.time() - self.start_time,
                'functions': {},
                'overall_stats': {
                    'total_function_calls': 0,
                    'total_execution_time': 0.0,
                    'average_success_rate': 0.0,
                },
            }
            total_calls = 0
            total_time = 0.0
            total_success = 0
            for func_name, metric in self.metrics.items():
                if metric['call_count'] > 0:
                    avg_time = metric['total_time'] / metric['call_count']
                    success_rate = (metric['success_count'] / metric['call_count']) * 100
                    summary['functions'][func_name] = {
                        'call_count': metric['call_count'],
                        'average_time': round(avg_time, 4),
                        'min_time': round(metric['min_time'], 4),
                        'max_time': round(metric['max_time'], 4),
                        'total_time': round(metric['total_time'], 4),
                        'success_rate': round(success_rate, 2),
                        'error_count': metric['error_count'],
                    }
                    total_calls += metric['call_count']
                    total_time += metric['total_time']
                    total_success += metric['success_count']

            if total_calls > 0:
                summary['overall_stats']['total_function_calls'] = total_calls
                summary['overall_stats']['total_execution_time'] = round(total_time, 4)
                summary['overall_stats']['average_success_rate'] = round((total_success / total_calls) * 100, 2)

            return summary

    def get_slow_functions(self, threshold_seconds: float = 1.0) -> List[Dict]:
        """Get functions that are performing slowly"""
        slow_functions = []
        with self.lock:
            for func_name, metric in self.metrics.items():
                if metric['call_count'] > 0:
                    avg_time = metric['total_time'] / metric['call_count']
                    if avg_time > threshold_seconds:
                        slow_functions.append({
                            'function': func_name,
                            'average_time': round(avg_time, 4),
                            'max_time': round(metric['max_time'], 4),
                            'call_count': metric['call_count'],
                        })
        return sorted(slow_functions, key=lambda x: x['average_time'], reverse=True)

    def reset_metrics(self):
        """Reset all performance metrics"""
        with self.lock:
            self.metrics.clear()
            self.start_time = time.time()


class DatabaseOptimizer:
    """Database performance optimization utilities"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def analyze_query_performance(self, query: str, params: tuple = None) -> Dict:
        """Analyze query performance and provide recommendations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA query_planner = ON")  # Enable query plan analysis
                start_time = time.time()
                cursor = conn.execute(query, params or ())
                results = cursor.fetchall()
                execution_time = time.time() - start_time
                explain_cursor = conn.execute(f"EXPLAIN QUERY PLAN {query}", params or ())
                query_plan = explain_cursor.fetchall()
                return {
                    'execution_time': round(execution_time, 6),
                    'row_count': len(results),
                    'query_plan': [
                        {'id': row[0], 'parent': row[1], 'notused': row[2], 'detail': row[3]} for row in query_plan
                    ],
                    'recommendations': self._get_query_recommendations(query_plan),
                }
        except Exception as e:
            logger.error(f"Error analyzing query performance: {e}")
            return {'error': str(e)}

    def _get_query_recommendations(self, query_plan: List) -> List[str]:
        """Generate query optimization recommendations based on query plan"""
        recommendations = []
        for row in query_plan:
            detail = row[3].lower() if len(row) > 3 else ""
            if "scan table" in detail and "using index" not in detail:
                recommendations.append(f"Consider adding an index for table scan: {detail}")
            if "temp b-tree" in detail:
                recommendations.append("Query uses temporary B-tree, consider optimizing ORDER BY clauses")
            if "using covering index" not in detail and "index" in detail:
                recommendations.append("Consider creating a covering index to avoid table lookups")
        if not recommendations:
            recommendations.append("Query appears to be well optimized")
        return recommendations

    def get_database_statistics(self) -> Dict:
        """Get comprehensive database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                stats['tables'] = {}
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    cursor = conn.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    stats['tables'][table] = {
                        'row_count': row_count,
                        'column_count': len(columns),
                        'columns': [{'name': col[1], 'type': col[2], 'not_null': bool(col[3])} for col in columns],
                    }
                cursor = conn.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index'")
                indexes = cursor.fetchall()
                stats['indexes'] = {}
                for index_name, table_name in indexes:
                    cursor = conn.execute(f"PRAGMA index_info({index_name})")
                    index_info = cursor.fetchall()
                    stats['indexes'][index_name] = {
                        'table': table_name,
                        'columns': [col[2] for col in index_info],
                    }
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                stats['database'] = {
                    'size_bytes': page_count * page_size,
                    'size_mb': round((page_count * page_size) / (1024 * 1024), 2),
                    'page_count': page_count,
                    'page_size': page_size,
                }
                return stats
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {'error': str(e)}

    def optimize_database(self) -> Dict:
        """Perform database optimization operations"""
        try:
            optimizations = []
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("ANALYZE")
                optimizations.append("Analyzed all tables for better query planning")
                conn.execute("VACUUM")
                optimizations.append("Vacuumed database to reclaim unused space")
                conn.execute("PRAGMA optimize")
                optimizations.append("Updated database statistics")
            return {
                'success': True,
                'optimizations_performed': optimizations,
                'timestamp': datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }


class SystemResourceMonitor:
    """Monitor system resources used by the application"""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss

    def get_resource_usage(self) -> Dict:
        """Get current resource usage of the application"""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            num_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            num_threads = self.process.num_threads()
            try:
                connections = len(self.process.connections())
            except (psutil.AccessDenied, AttributeError):
                connections = 0
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory': {
                    'rss_mb': round(memory_info.rss / (1024 * 1024), 2),
                    'vms_mb': round(memory_info.vms / (1024 * 1024), 2),
                    'percent': round(memory_percent, 2),
                    'growth_mb': round((memory_info.rss - self.initial_memory) / (1024 * 1024), 2),
                },
                'file_descriptors': num_fds,
                'threads': num_threads,
                'connections': connections,
                'uptime_seconds': round(time.time() - self.start_time, 1),
                'timestamp': datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {'error': str(e)}


performance_monitor = PerformanceMonitor()


def monitor_performance(func_name: str = None):
    """Convenience decorator for performance monitoring"""
    return performance_monitor.timing_decorator(func_name)
