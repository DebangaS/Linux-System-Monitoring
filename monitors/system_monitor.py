"""
Enhanced System Resource Monitor with Database Integration
Author: Member 2 (updating Day 2 work)
"""

import psutil
import time
import threading
import os
import logging
from datetime import datetime
from typing import Dict, List
from collections import deque

# Import database manager
from database.models import db_manager
# data_analyzer is optional; provide a safe stub when not available
try:
    from database.data_analyzer import data_analyzer  # type: ignore
except Exception:
    class _AnalyzerStub:
        def get_cpu_trends(self, hours: int = 24):
            return {}
        def get_memory_trends(self, hours: int = 24):
            return {}
        def get_alert_summary(self, hours: int = 24):
            return {}
    data_analyzer = _AnalyzerStub()

logger = logging.getLogger(__name__)


class SystemResourceMonitor:
    def get_alerts(self) -> list:
        """Return current system alerts (placeholder implementation)"""
        # TODO: Implement real alert logic
        return []
    """Enhanced system resource monitor with database integration"""

    def __init__(self, max_data_points=100, enable_database=True):
        self.max_data_points = max_data_points
        self.enable_database = enable_database

        # In-memory data history (for real-time display)
        self.data_history = {
            'cpu': deque(maxlen=max_data_points),
            'memory': deque(maxlen=max_data_points),
            'disk': deque(maxlen=max_data_points),
            'network': deque(maxlen=max_data_points)
        }

        self.last_network_io = psutil.net_io_counters()
        self.monitoring = False
        self.monitor_thread = None

        # Data validation settings
        self.validation_rules = {
            'cpu_usage': {'min': 0, 'max': 100},
            'memory_usage': {'min': 0, 'max': 100},
            'disk_usage': {'min': 0, 'max': 100}
        }

    # -----------------------------
    # Data Validation and Sanitization
    # -----------------------------
    def validate_data(self, data_type: str, value: float) -> bool:
        """Validate data values before storage"""
        try:
            if data_type in self.validation_rules:
                rules = self.validation_rules[data_type]
                return rules['min'] <= value <= rules['max']
            return True
        except Exception as e:
            logger.error(f"Error validating {data_type}: {e}")
            return False

    def sanitize_data(self, data: Dict) -> Dict:
        """Sanitize data before database storage"""
        try:
            sanitized = {}
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    sanitized[key] = round(float(value), 2)
                elif isinstance(value, dict):
                    sanitized[key] = self.sanitize_data(value)
                elif isinstance(value, list):
                    sanitized[key] = [
                        round(float(v), 2) if isinstance(v, (int, float)) else v
                        for v in value
                    ]
                else:
                    sanitized[key] = value
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing data: {e}")
            return data

    # -----------------------------
    # Resource Usage Functions
    # -----------------------------
    def get_cpu_usage_enhanced(self) -> Dict:
        """Enhanced CPU usage with validation and database integration"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.0)
            cpu_count = psutil.cpu_count()
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            cpu_per_core = psutil.cpu_percent(percpu=True, interval=0.0)
            cpu_times = psutil.cpu_times()

            if not self.validate_data('cpu_usage', cpu_percent):
                logger.warning(f"Invalid CPU usage value: {cpu_percent}")
                cpu_percent = 0.0

            cpu_data = {
                'usage_percent': cpu_percent,
                'count_physical': cpu_count,
                'count_logical': cpu_count_logical,
                'per_core': [round(core, 2) for core in cpu_per_core],
                'frequency': {
                    'current': round(cpu_freq.current, 2) if cpu_freq else 0,
                    'min': round(cpu_freq.min, 2) if cpu_freq else 0,
                    'max': round(cpu_freq.max, 2) if cpu_freq else 0
                },
                'times': {
                    'user': round(cpu_times.user, 2),
                    'system': round(cpu_times.system, 2),
                    'idle': round(cpu_times.idle, 2)
                },
                'load_average': list(os.getloadavg()) if hasattr(os, 'getloadavg') else [],
                'timestamp': datetime.now().isoformat()
            }

            cpu_data = self.sanitize_data(cpu_data)
            self.data_history['cpu'].append(cpu_data)
            return cpu_data
        except Exception as e:
            logger.error(f"Error getting enhanced CPU usage: {e}")
            return {'usage_percent': 0, 'error': str(e)}

    def get_memory_usage_enhanced(self) -> Dict:
        """Enhanced memory usage with validation and database integration"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            memory_percent = memory.percent
            if not self.validate_data('memory_usage', memory_percent):
                logger.warning(f"Invalid memory usage value: {memory_percent}")
                memory_percent = 0.0

            memory_data = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'free': memory.free,
                'usage_percent': memory_percent,
                'active': getattr(memory, 'active', 0),
                'inactive': getattr(memory, 'inactive', 0),
                'buffers': getattr(memory, 'buffers', 0),
                'cached': getattr(memory, 'cached', 0),
                'shared': getattr(memory, 'shared', 0),
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'usage_percent': round(swap.percent, 2),
                    'sin': getattr(swap, 'sin', 0),
                    'sout': getattr(swap, 'sout', 0)
                },
                'timestamp': datetime.now().isoformat()
            }

            memory_data = self.sanitize_data(memory_data)
            self.data_history['memory'].append(memory_data)
            return memory_data
        except Exception as e:
            logger.error(f"Error getting enhanced memory usage: {e}")
            return {'usage_percent': 0, 'error': str(e)}

    # Placeholder for disk and network (assuming existing Day 2 functions)
    def get_disk_usage(self) -> Dict:
        try:
            disk = psutil.disk_usage('/')
            data = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'total_usage_percent': disk.percent,
                'timestamp': datetime.now().isoformat()
            }
            return self.sanitize_data(data)
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return {}

    def get_network_usage(self) -> Dict:
        try:
            new_network_io = psutil.net_io_counters()
            sent_rate = (new_network_io.bytes_sent - self.last_network_io.bytes_sent) / 1024
            recv_rate = (new_network_io.bytes_recv - self.last_network_io.bytes_recv) / 1024
            self.last_network_io = new_network_io

            data = {
                'sent_rate_kbps': round(sent_rate, 2),
                'recv_rate_kbps': round(recv_rate, 2),
                'timestamp': datetime.now().isoformat()
            }
            self.data_history['network'].append(data)
            return data
        except Exception as e:
            logger.error(f"Error getting network usage: {e}")
            return {}

    # -----------------------------
    # Aggregation Functions
    # -----------------------------
    def get_aggregated_data(self, hours: int = 1) -> Dict:
        """Get aggregated data from database for specified time period"""
        try:
            if not self.enable_database:
                return {}

            cpu_data = db_manager.get_historical_metrics('cpu', hours)
            memory_data = db_manager.get_historical_metrics('memory', hours)
            disk_data = db_manager.get_historical_metrics('disk', hours)
            network_data = db_manager.get_historical_metrics('network', hours)

            aggregated = {
                'time_period_hours': hours,
                'data_points_count': {
                    'cpu': len(cpu_data),
                    'memory': len(memory_data),
                    'disk': len(disk_data),
                    'network': len(network_data)
                },
                'cpu': self._aggregate_cpu_data(cpu_data),
                'memory': self._aggregate_memory_data(memory_data),
                'disk': self._aggregate_disk_data(disk_data),
                'network': self._aggregate_network_data(network_data),
                'timestamp': datetime.now().isoformat()
            }
            return aggregated
        except Exception as e:
            logger.error(f"Error getting aggregated data: {e}")
            return {}

    def _aggregate_cpu_data(self, cpu_data: List[Dict]) -> Dict:
        if not cpu_data:
            return {}
        try:
            usage_values = [data.get('usage_percent', 0) for data in cpu_data]
            return {
                'average_usage': round(sum(usage_values) / len(usage_values), 2),
                'min_usage': round(min(usage_values), 2),
                'max_usage': round(max(usage_values), 2),
                'samples_count': len(usage_values),
                'high_usage_periods': len([v for v in usage_values if v > 80])
            }
        except Exception as e:
            logger.error(f"Error aggregating CPU data: {e}")
            return {}

    def _aggregate_memory_data(self, memory_data: List[Dict]) -> Dict:
        if not memory_data:
            return {}
        try:
            usage_values = [data.get('usage_percent', 0) for data in memory_data]
            return {
                'average_usage': round(sum(usage_values) / len(usage_values), 2),
                'min_usage': round(min(usage_values), 2),
                'max_usage': round(max(usage_values), 2),
                'samples_count': len(usage_values),
                'high_usage_periods': len([v for v in usage_values if v > 85])
            }
        except Exception as e:
            logger.error(f"Error aggregating memory data: {e}")
            return {}

    def _aggregate_disk_data(self, disk_data: List[Dict]) -> Dict:
        if not disk_data:
            return {}
        try:
            usage_values = [data.get('total_usage_percent', 0) for data in disk_data]
            return {
                'average_usage': round(sum(usage_values) / len(usage_values), 2),
                'min_usage': round(min(usage_values), 2),
                'max_usage': round(max(usage_values), 2),
                'samples_count': len(usage_values),
                'critical_periods': len([v for v in usage_values if v > 90])
            }
        except Exception as e:
            logger.error(f"Error aggregating disk data: {e}")
            return {}

    def _aggregate_network_data(self, network_data: List[Dict]) -> Dict:
        if not network_data:
            return {}
        try:
            sent_values = [data.get('sent_rate_kbps', 0) for data in network_data]
            recv_values = [data.get('recv_rate_kbps', 0) for data in network_data]
            return {
                'average_sent_kbps': round(sum(sent_values) / len(sent_values), 2),
                'average_recv_kbps': round(sum(recv_values) / len(recv_values), 2),
                'peak_sent_kbps': round(max(sent_values), 2),
                'peak_recv_kbps': round(max(recv_values), 2),
                'samples_count': len(sent_values)
            }
        except Exception as e:
            logger.error(f"Error aggregating network data: {e}")
            return {}

    # -----------------------------
    # Monitoring Thread Control
    # -----------------------------
    def start_optimized_monitoring(self, interval: int = 2, store_interval: int = 10) -> bool:
        """Start optimized monitoring with configurable storage interval"""
        try:
            if self.monitoring:
                logger.warning("Monitoring already running")
                return False

            def monitoring_loop():
                iteration = 0
                while self.monitoring:
                    try:
                        cpu_data = self.get_cpu_usage_enhanced()
                        memory_data = self.get_memory_usage_enhanced()
                        disk_data = self.get_disk_usage()
                        network_data = self.get_network_usage()

                        if iteration % (store_interval // interval) == 0:
                            if self.enable_database:
                                db_manager.store_system_metrics(
                                    cpu_data, memory_data, disk_data, network_data
                                )
                        iteration += 1
                        time.sleep(interval)
                    except Exception as e:
                        logger.error(f"Error in optimized monitoring loop: {e}")
                        time.sleep(interval)

            self.monitoring = True
            self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info(f"Started optimized monitoring (interval: {interval}s, store: {store_interval}s)")
            return True
        except Exception as e:
            logger.error(f"Error starting optimized monitoring: {e}")
            return False

    def stop_monitoring(self) -> bool:
        """Stop the monitoring thread"""
        try:
            if not self.monitoring:
                return True
            self.monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            logger.info("Monitoring stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            return False

    # -----------------------------
    # System Summary
    # -----------------------------
    def get_all_resources(self) -> Dict:
        """Get current snapshot of all resources"""
        return {
            'cpu': self.get_cpu_usage_enhanced(),
            'memory': self.get_memory_usage_enhanced(),
            'disk': self.get_disk_usage(),
            'network': self.get_network_usage(),
        }

    def get_system_summary(self) -> Dict:
        """Get comprehensive system summary with trends"""
        try:
            current_resources = self.get_all_resources()
            hourly_aggregated = self.get_aggregated_data(hours=1)

            trends = {}
            if self.enable_database and hasattr(data_analyzer, 'get_cpu_trends'):
                try:
                    trends = {
                        'cpu_trends': data_analyzer.get_cpu_trends(hours=24),
                        'memory_trends': data_analyzer.get_memory_trends(hours=24),
                        'alert_summary': data_analyzer.get_alert_summary(hours=24)
                    }
                except Exception:
                    trends = {}

            return {
                'current': current_resources,
                'hourly_summary': hourly_aggregated,
                'trends': trends,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system summary: {e}")
            return {}


# Enhanced global monitor instance
system_monitor = SystemResourceMonitor(enable_database=True)
