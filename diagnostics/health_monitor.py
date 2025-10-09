"""
Advanced Health Monitoring and Diagnostics
Author: Member 1 
"""
import asyncio
import time
import psutil
import logging
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# NOTE: This import would be at the top level in a real application structure.
# from database.advanced_models import get_advanced_db_manager

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any]
    timestamp: str

    def to_dict(self):
        return {
            'name': self.name, 'status': self.status.value, 'score': self.score,
            'message': self.message, 'details': self.details, 'timestamp': self.timestamp
        }

@dataclass
class SystemDiagnostics:
    overall_health: HealthStatus
    overall_score: float
    checks: List[HealthCheck]
    recommendations: List[str]
    timestamp: str

    def to_dict(self):
        return {
            'overall_health': self.overall_health.value, 'overall_score': self.overall_score,
            'checks': [check.to_dict() for check in self.checks],
            'recommendations': self.recommendations, 'timestamp': self.timestamp
        }

class AdvancedHealthMonitor:
    """Advanced health monitoring and diagnostics system."""
    # FIX: Use dependency injection for the db_manager to avoid circular imports and improve testability.
    def __init__(self, db_manager: Optional[Any] = None):
        self.db_manager = db_manager
        self.health_history = []
        self.thresholds = {
            'cpu_warning': 80.0, 'cpu_critical': 95.0,
            'memory_warning': 80.0, 'memory_critical': 95.0,
            'disk_warning': 85.0, 'disk_critical': 95.0,
            'network_error_rate_warning': 0.05, 'network_error_rate_critical': 0.10,
            'database_response_time_warning': 100.0,  # ms
            'database_response_time_critical': 500.0, # ms
            # FIX: Added missing thresholds for process health.
            'process_cpu_high': 50.0, # %
            'process_memory_high': 10.0 # %
        }
        logger.info("Advanced Health Monitor initialized")

    async def initialize_baselines(self):
        """Asynchronously initialize performance baselines by running blocking calls in a thread."""
        logger.info("Initializing performance baselines...")
        # FIX: Run the entire blocking method in a thread.
        self.performance_baselines = await asyncio.to_thread(self._initialize_baselines_sync)
        logger.info("Performance baselines initialized.")

    def _initialize_baselines_sync(self) -> dict:
        """Synchronous part of baseline initialization."""
        try:
            cpu_samples = [psutil.cpu_percent(interval=0.5) for _ in range(3)]
            baselines = {
                'cpu': {
                    'mean': statistics.mean(cpu_samples),
                    # FIX: Added 'else 0' for the stdev calculation.
                    'stdev': statistics.stdev(cpu_samples) if len(cpu_samples) > 1 else 0,
                    'samples': len(cpu_samples)
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'baseline_usage': psutil.virtual_memory().percent
                }
            }
            return baselines
        except Exception as e:
            logger.error(f"Error initializing baselines: {e}")
            return {}

    async def run_comprehensive_diagnostics(self) -> SystemDiagnostics:
        """Run comprehensive system diagnostics asynchronously."""
        try:
            health_check_tasks = [
                self._check_cpu_health(), self._check_memory_health(),
                self._check_disk_health(), self._check_network_health(),
                self._check_database_health(), self._check_process_health(),
                self._check_security_health(), self._check_performance_health()
            ]
            # FIX: Corrected incomplete line.
            check_results = await asyncio.gather(*health_check_tasks, return_exceptions=True)

            checks = []
            for result in check_results:
                if isinstance(result, Exception):
                    error_msg = f"Health check failed: {str(result)}"
                    logger.error(error_msg, exc_info=True)
                    checks.append(HealthCheck("failed_check", HealthStatus.UNKNOWN, 0.0,
                                              error_msg, {'error': str(result)}, datetime.utcnow().isoformat()))
                elif isinstance(result, HealthCheck):
                    checks.append(result)
                elif isinstance(result, list):
                    checks.extend(result)

            scores = [check.score for check in checks if check.score is not None]
            overall_score = statistics.mean(scores) if scores else 0.0
            overall_health = HealthStatus.HEALTHY if overall_score >= 0.8 else \
                             HealthStatus.WARNING if overall_score >= 0.6 else HealthStatus.CRITICAL

            diagnostics = SystemDiagnostics(overall_health, overall_score, checks,
                                            self._generate_recommendations(checks), datetime.utcnow().isoformat())
            
            self.health_history.append(diagnostics)
            if len(self.health_history) > 100: self.health_history.pop(0)
            return diagnostics
        except Exception as e:
            logger.error(f"Error running comprehensive diagnostics: {e}", exc_info=True)
            return SystemDiagnostics(HealthStatus.UNKNOWN, 0.0, [], [f"Diagnostics failed: {e}"], datetime.utcnow().isoformat())

    # NOTE: The pattern below is used for all checks:
    # The async `_check_*` function calls a synchronous `_get_*_data` function in a separate thread.
    # This prevents blocking the main application loop.
    def _get_cpu_data_sync(self):
        """Synchronous helper to gather CPU data."""
        cpu_samples = [psutil.cpu_percent(interval=0.2) for _ in range(3)]
        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        cpu_freq = psutil.cpu_freq()
        # FIX: Added a default value for the ternary operator.
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        return {
            'avg_cpu': avg_cpu, 'max_cpu': max_cpu, 'samples': cpu_samples, 'cpu_freq': cpu_freq,
            'load_avg': load_avg, 'core_count': psutil.cpu_count(), 'logical_cores': psutil.cpu_count(logical=True)
        }

    async def _check_cpu_health(self) -> HealthCheck:
        """Check CPU health without blocking the event loop."""
        try:
            data = await asyncio.to_thread(self._get_cpu_data_sync)
            max_cpu = data['max_cpu']
            if max_cpu >= self.thresholds['cpu_critical']:
                status, score, message = HealthStatus.CRITICAL, 0.0, f"Critical CPU usage: {max_cpu:.1f}%"
            elif max_cpu >= self.thresholds['cpu_warning']:
                status, score, message = HealthStatus.WARNING, 0.5, f"High CPU usage: {max_cpu:.1f}%"
            else:
                status, score, message = HealthStatus.HEALTHY, 1.0 - (data['avg_cpu'] / 100.0), f"CPU usage normal: {data['avg_cpu']:.1f}%"
            
            details = {'average_usage': data['avg_cpu'], 'max_usage': max_cpu, 'samples': data['samples'],
                       'frequency_current': data['cpu_freq'].current if data['cpu_freq'] else None,
                       'frequency_max': data['cpu_freq'].max if data['cpu_freq'] else None,
                       'load_average': data['load_avg'], 'core_count': data['core_count'], 'logical_cores': data['logical_cores']}
            return HealthCheck("cpu_health", status, score, message, details, datetime.utcnow().isoformat())
        except Exception as e:
            logger.error(f"Error checking CPU health: {e}", exc_info=True)
            return HealthCheck("cpu_health", HealthStatus.UNKNOWN, 0.0, f"CPU health check failed: {e}", {'error': str(e)}, datetime.utcnow().isoformat())

    def _get_memory_data_sync(self):
        """Synchronous helper to gather memory data."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return memory, swap

    async def _check_memory_health(self) -> HealthCheck:
        """Check memory health without blocking the event loop."""
        try:
            memory, swap = await asyncio.to_thread(self._get_memory_data_sync)
            if memory.percent >= self.thresholds['memory_critical']:
                status, score, message = HealthStatus.CRITICAL, 0.0, f"Critical memory usage: {memory.percent:.1f}%"
            elif memory.percent >= self.thresholds['memory_warning']:
                status, score, message = HealthStatus.WARNING, 0.5, f"High memory usage: {memory.percent:.1f}%"
            else:
                status, score, message = HealthStatus.HEALTHY, 1.0 - (memory.percent / 100.0), f"Memory usage normal: {memory.percent:.1f}%"
            
            if swap.percent > 50:
                status = HealthStatus.WARNING
                score = min(score, 0.3)
                message += f" (High swap usage: {swap.percent:.1f}%)"
            
            details = {'total_gb': memory.total / (1024**3), 'usage_percent': memory.percent, 'swap_percent': swap.percent}
            return HealthCheck("memory_health", status, score, message, details, datetime.utcnow().isoformat())
        except Exception as e:
            logger.error(f"Error checking memory health: {e}", exc_info=True)
            return HealthCheck("memory_health", HealthStatus.UNKNOWN, 0.0, f"Memory health check failed: {e}", {'error': str(e)}, datetime.utcnow().isoformat())
    
    # ... Other checks follow the same async/sync pattern ...

    async def _check_disk_health(self) -> List[HealthCheck]:
        """Check disk health for all partitions."""
        # FIX: Fetch system-wide IO stats once to avoid redundancy.
        io_stats = await asyncio.to_thread(psutil.disk_io_counters)
        checks = []
        try:
            partitions = await asyncio.to_thread(psutil.disk_partitions)
            for partition in partitions:
                try:
                    usage = await asyncio.to_thread(psutil.disk_usage, partition.mountpoint)
                    usage_percent = usage.percent
                    if usage_percent >= self.thresholds['disk_critical']:
                        status, score, message = HealthStatus.CRITICAL, 0.0, f"Critical disk usage: {usage_percent:.1f}%"
                    elif usage_percent >= self.thresholds['disk_warning']:
                        status, score, message = HealthStatus.WARNING, 0.5, f"High disk usage: {usage_percent:.1f}%"
                    else:
                        status, score, message = HealthStatus.HEALTHY, 1.0 - (usage_percent / 100.0), f"Disk usage normal: {usage_percent:.1f}%"
                    
                    details = { 'mountpoint': partition.mountpoint, 'total_gb': usage.total / (1024**3), 'usage_percent': usage_percent,
                                # FIX: Corrected multiple incomplete lines.
                                'io_read_count': io_stats.read_count if io_stats else None, 'io_write_count': io_stats.write_count if io_stats else None,
                                'io_read_bytes': io_stats.read_bytes if io_stats else None, 'io_write_bytes': io_stats.write_bytes if io_stats else None }
                    
                    # FIX: Corrected incomplete replace() call.
                    check_name = f"disk_health_{partition.device.replace('/', '_').replace(':', '').replace('\\', '_')}"
                    checks.append(HealthCheck(check_name, status, score, f"{partition.device}: {message}", details, datetime.utcnow().isoformat()))
                except Exception as e:
                    # FIX: Completed logging message.
                    logger.warning(f"Could not check disk {partition.device}: {e}")
                    continue
            return checks
        except Exception as e:
            logger.error(f"Error checking disk health: {e}", exc_info=True)
            return [HealthCheck("disk_health", HealthStatus.UNKNOWN, 0.0, f"Disk health check failed: {e}", {'error': str(e)}, datetime.utcnow().isoformat())]

    async def _check_database_health(self) -> HealthCheck:
        """Check database health using the injected manager."""
        if not self.db_manager:
            return HealthCheck("database_health", HealthStatus.UNKNOWN, 0.5, "Database manager not configured.", {}, datetime.utcnow().isoformat())
        try:
            start_time = time.monotonic()
            # FIX: Run blocking DB call in a thread.
            db_stats = await asyncio.to_thread(self.db_manager.get_database_stats)
            response_time = (time.monotonic() - start_time) * 1000  # ms
            
            # FIX: Completed incomplete lines.
            if response_time >= self.thresholds['database_response_time_critical']:
                status, score, message = HealthStatus.CRITICAL, 0.0, f"Critical database response time: {response_time:.1f}ms"
            elif response_time >= self.thresholds['database_response_time_warning']:
                status, score, message = HealthStatus.WARNING, 0.5, f"Slow database response time: {response_time:.1f}ms"
            else:
                status, score, message = HealthStatus.HEALTHY, max(0.1, 1.0 - (response_time / 100.0)), f"Database responding well: {response_time:.1f}ms"
            
            details = {'response_time_ms': response_time, **db_stats}
            return HealthCheck("database_health", status, score, message, details, datetime.utcnow().isoformat())
        except Exception as e:
            logger.error(f"Error checking database health: {e}", exc_info=True)
            return HealthCheck("database_health", HealthStatus.UNKNOWN, 0.0, f"Database health check failed: {e}", {'error': str(e)}, datetime.utcnow().isoformat())
    
    # ... other checks would also be refactored ...

    def _generate_recommendations(self, checks: List[HealthCheck]) -> List[str]:
        """Generate recommendations based on health checks."""
        recommendations = set() # Use a set to avoid duplicates automatically
        for check in checks:
            # FIX: Completed all recommendation strings to be helpful.
            if check.status == HealthStatus.CRITICAL:
                if 'cpu' in check.name: recommendations.add("CRITICAL: Add CPU resources or optimize high-load applications immediately.")
                elif 'memory' in check.name: recommendations.add("CRITICAL: Increase system memory or identify and fix memory leaks.")
                elif 'disk' in check.name: recommendations.add("CRITICAL: Free up disk space urgently or add more storage.")
                elif 'database' in check.name: recommendations.add("CRITICAL: Optimize slow database queries and check connection pool health.")
                elif 'network' in check.name: recommendations.add("CRITICAL: Investigate network hardware and configuration for high error rates.")
            elif check.status == HealthStatus.WARNING:
                if 'cpu' in check.name: recommendations.add("WARNING: Monitor CPU usage trends and identify processes causing high load.")
                elif 'memory' in check.name: recommendations.add("WARNING: Monitor memory usage closely and consider optimizing application memory footprints.")
                elif 'disk' in check.name: recommendations.add("WARNING: Plan for additional storage capacity or archive old data.")
                elif 'process' in check.name and 'zombie' in check.message: recommendations.add("WARNING: Investigate and clean up zombie processes to free system resources.")
                elif 'security' in check.name: recommendations.add("WARNING: Review security configurations, close unnecessary ports, and investigate suspicious processes.")
        return sorted(list(recommendations))

# Global health monitor instance
# NOTE: In a real app, you would pass the db_manager instance here.
# health_monitor = AdvancedHealthMonitor(db_manager=get_advanced_db_manager())
health_monitor = AdvancedHealthMonitor()

def get_health_monitor():
    """Get global health monitor instance."""
    return health_monitor
