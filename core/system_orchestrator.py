"""
Advanced System Orchestrator for Production Deployment
Author: Member 1 
"""
import asyncio
import time
import logging
import psutil
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import gc

# NOTE: Assuming these imports are correctly configured in your project structure.
from config import get_config
from database.advanced_models import get_advanced_db_manager
from monitors.system_monitor import system_monitor
from monitors.gpu_monitor import gpu_monitor
from monitors.network_advanced import NetworkAdvancedMonitor
from intelligence.anomaly_detector import AnomalyDetector
from intelligence.predictive_monitor import PredictiveMonitor
from auth.models import User, Session
from auth.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """System health metrics"""
    cpu_health: float
    memory_health: float
    disk_health: float
    network_health: float
    database_health: float
    overall_score: float
    timestamp: str

    def to_dict(self):
        return asdict(self)

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    response_times: Dict[str, float]
    throughput: Dict[str, int]
    error_rates: Dict[str, float]
    resource_usage: Dict[str, float]
    cache_hit_rates: Dict[str, float]
    timestamp: str

class SystemOrchestrator:
    """Advanced system orchestrator for production deployment"""
    def __init__(self):
        self.config = get_config()
        self.db_manager = get_advanced_db_manager()
        
        # Component managers
        self.anomaly_detector = AnomalyDetector()
        self.predictive_monitor = PredictiveMonitor()
        self.network_monitor = NetworkAdvancedMonitor()
        
        # Performance tracking
        self.performance_metrics = {}
        self.health_history = []
        self.active_connections = weakref.WeakSet()
        
        # Threading and async
        # NOTE: Using ThreadPoolExecutor for I/O-bound or blocking library calls
        # and ProcessPoolExecutor for CPU-intensive tasks.
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        
        self.monitoring_active = False
        self.monitoring_tasks = []

        # FIX: Initialize all instance attributes in the constructor for clarity and predictability.
        self._start_time = None
        self._process_snapshot_counter = 0
        
        logger.info("System Orchestrator initialized")

    async def start_system(self):
        """Start all system components with proper orchestration"""
        try:
            logger.info("Starting system orchestration...")
            self._start_time = time.time()
            self.monitoring_active = True
            
            await self._initialize_components()
            await self._start_monitoring_services()
            await self._start_background_tasks()
            await self._start_health_monitoring()
            
            logger.info("System orchestration started successfully")
        except Exception as e:
            logger.error(f"Failed to start system orchestration: {e}", exc_info=True)
            await self.shutdown_system()
            raise

    async def _initialize_components(self):
        """Initialize all system components"""
        component_names = ["database", "monitoring", "intelligence", "authentication", "performance_tracking"]
        initialization_tasks = [
            self._init_database(),
            self._init_monitoring(),
            self._init_intelligence(),
            self._init_authentication(),
            self._init_performance_tracking()
        ]
        
        # FIX: Corrected typo 'T' to 'True' and improved error handling logic.
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                component_name = component_names[i]
                logger.error(f"Failed to initialize component '{component_name}': {result}")
                raise result

    async def _init_database(self):
        """Initialize database with optimizations"""
        try:
            # FIX: Run blocking database operations in a separate thread to avoid blocking the event loop.
            await asyncio.to_thread(self.db_manager.optimize_database)
            stats = await asyncio.to_thread(self.db_manager.get_database_stats)
            
            if stats.get('needs_maintenance', False):
                await asyncio.to_thread(self.db_manager.run_maintenance)
            logger.info("Database initialization complete")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def _init_monitoring(self):
        """Initialize monitoring components"""
        # NOTE: network_monitor.start_monitoring should be non-blocking or run in a thread if it is.
        self.network_monitor.start_monitoring(interval=1.0)
        try:
            # FIX: Offload potentially blocking file I/O to a thread.
            await asyncio.to_thread(self.anomaly_detector.load_models, 'data/models/anomaly_models.pkl')
        except Exception:
            logger.info("No pre-trained anomaly models found, will train new ones.")
        logger.info("Monitoring initialization complete")

    async def _init_intelligence(self):
        """Initialize AI/ML components"""
        # NOTE: Ensure predictive_monitor.initialize() is async or non-blocking.
        self.predictive_monitor.initialize()
        asyncio.create_task(self._train_anomaly_models())
        logger.info("Intelligence initialization complete")

    async def _init_authentication(self):
        """Initialize authentication system"""
        # FIX: Blocking DB and initialization calls moved to a thread.
        await asyncio.to_thread(Session.cleanup_expired_sessions)
        await asyncio.to_thread(rate_limiter.initialize)
        logger.info("Authentication initialization complete")

    async def _init_performance_tracking(self):
        """Initialize performance tracking"""
        self.performance_metrics = {
            'response_times': {}, 'throughput': {}, 'error_rates': {},
            'resource_usage': {}, 'cache_hit_rates': {}
        }
        logger.info("Performance tracking initialization complete")

    def _start_main_loops(self, coroutine_funcs: List):
        """Helper to create and store main monitoring loop tasks."""
        for coro_func in coroutine_funcs:
            task = asyncio.create_task(coro_func())
            self.monitoring_tasks.append(task)

    async def _start_monitoring_services(self):
        """Start all monitoring services"""
        self._start_main_loops([
            self._monitor_system_resources, self._monitor_database_health,
            self._monitor_api_performance, self._monitor_user_sessions,
            self._monitor_security_events
        ])
        logger.info("Monitoring services started")

    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self._start_main_loops([
            self._cleanup_old_data, self._update_performance_baselines,
            self._generate_health_reports, self._optimize_system_performance,
            self._backup_critical_data
        ])
        logger.info("Background tasks started")

    async def _start_health_monitoring(self):
        """Start comprehensive health monitoring"""
        self._start_main_loops([self._health_monitoring_loop])
        logger.info("Health monitoring started")

    async def _monitor_system_resources(self):
        """Monitor system resources with advanced analytics"""
        while self.monitoring_active:
            try:
                # FIX: Entire collection of blocking psutil calls is run in a thread.
                resources = await asyncio.to_thread(self._collect_system_resources_sync)
                
                if resources:
                    await self._store_system_data(resources)
                    await self._check_resource_anomalies(resources)
                    await self._update_predictive_models(resources)
                
                # NOTE: Intervals should be configurable.
                await asyncio.sleep(self.config.get('SYSTEM_MONITOR_INTERVAL', 2))
            except asyncio.CancelledError:
                logger.info("System resource monitoring task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in system resource monitoring: {e}")
                await asyncio.sleep(5)

    def _collect_system_resources_sync(self) -> Dict[str, Any]:
        """Synchronous helper to collect resource data. Runs in a thread."""
        try:
            # NOTE: Getting top processes can be resource-intensive.
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu': system_monitor.get_cpu_data(),
                'memory': system_monitor.get_memory_data(),
                'disk': system_monitor.get_disk_data(),
                'network': system_monitor.get_network_data(),
                'gpu': gpu_monitor.get_gpu_info(),
                'network_advanced': self.network_monitor.get_interface_statistics(),
                'processes': system_monitor.get_top_processes(limit=10),
                # FIX: Corrected missing parenthesis.
                'system_load': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0,0,0)
            }
        except Exception as e:
            logger.error(f"Error collecting system resources: {e}")
            return {}

    async def _store_system_data(self, resources: Dict[str, Any]):
        """Store system data with optimized batch processing"""
        try:
            # FIX: All database calls must be run in a thread.
            await asyncio.to_thread(
                self.db_manager.store_system_metrics,
                resources.get('cpu', {}), resources.get('memory', {}),
                resources.get('disk', {}), resources.get('network', {})
            )
            
            if gpu_info := resources.get('gpu'):
                # FIX: Corrected invalid index syntax `[^0]` to `[0]` and added safety checks.
                utilization = gpu_info.get('gpus', [{}])[0].get('utilization', 0.0)
                await asyncio.to_thread(
                    self.db_manager.store_advanced_metric,
                    'gpu', 'gpu_utilization', utilization, gpu_info
                )

            self._process_snapshot_counter += 1
            if self._process_snapshot_counter % 10 == 0 and resources.get('processes'):
                await asyncio.to_thread(
                    self.db_manager.store_process_snapshot, resources['processes']
                )
        except Exception as e:
            logger.error(f"Error storing system data: {e}")

    async def _check_resource_anomalies(self, resources: Dict[str, Any]):
        """Check for resource anomalies using ML"""
        try:
            if cpu_usage := resources.get('cpu', {}).get('usage_percent'):
                cpu_anomaly = self.anomaly_detector.detect_anomaly('cpu_percent', cpu_usage, resources['timestamp'])
                if cpu_anomaly and cpu_anomaly.get('is_anomaly'):
                    # FIX: Pass the actual value, not the entire dict.
                    await self._handle_anomaly('cpu', cpu_anomaly, cpu_usage)
            
            if mem_usage := resources.get('memory', {}).get('usage_percent'):
                memory_anomaly = self.anomaly_detector.detect_anomaly('memory_percent', mem_usage, resources['timestamp'])
                if memory_anomaly and memory_anomaly.get('is_anomaly'):
                    # FIX: Corrected incomplete line and passed the correct value.
                    await self._handle_anomaly('memory', memory_anomaly, mem_usage)
        except Exception as e:
            logger.error(f"Error checking resource anomalies: {e}")

    async def _handle_anomaly(self, resource_type: str, anomaly_data: Dict, value: float):
        """Handle detected anomalies"""
        severity = 'critical' if anomaly_data.get('anomaly_score', 0) > 0.9 else 'warning'
        message = f"Anomaly detected in {resource_type}: {value:.2f}%"
        logger.warning(f"{message} (Score: {anomaly_data.get('anomaly_score', 0):.2f})")
        # FIX: Run blocking DB call in a thread.
        await asyncio.to_thread(
            self.db_manager.store_system_alert,
            alert_type=resource_type, level=severity, message=message, value=value
        )

    # NOTE: Other monitoring loops would need similar `asyncio.to_thread` wrappers for any blocking calls.
    # Below is an example for the database health monitor.
    
    async def _monitor_database_health(self):
        """Monitor database health and performance"""
        while self.monitoring_active:
            try:
                db_stats = await asyncio.to_thread(self.db_manager.get_database_stats)
                health_score = self._calculate_database_health_score(db_stats)
                
                await asyncio.to_thread(
                    self.db_manager.store_advanced_metric,
                    'database', 'health_score', health_score, db_stats
                )
                
                if health_score < 0.7:
                    # FIX: Corrected incomplete function call.
                    await self._handle_database_health_issue(health_score, db_stats)
                
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                logger.info("Database health monitoring task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error monitoring database health: {e}")
                await asyncio.sleep(30)
    
    def _calculate_database_health_score(self, db_stats: Dict) -> float:
        """Calculate database health score based on stats."""
        # NOTE: This logic is synchronous and fast, so it doesn't need a thread.
        score = 1.0
        if db_stats.get('db_size_mb', 0) > 1000: score -= 0.1
        if db_stats.get('avg_query_time', 0) > 100: score -= 0.2
        if db_stats.get('connection_pool_usage', 0) > 0.8: score -= 0.1
        return max(0.0, score)

    async def _handle_database_health_issue(self, health_score: float, db_stats: Dict):
        """Handle database health issues"""
        severity = 'critical' if health_score < 0.5 else 'warning'
        message = f"Database health degraded: score={health_score:.2f}"
        logger.warning(message)

        await asyncio.to_thread(
            self.db_manager.store_system_alert,
            'database', severity, message, health_score
        )
        
        if health_score < 0.5:
            logger.warning("Attempting automatic database optimization...")
            await asyncio.to_thread(self.db_manager.optimize_database)

    # ... Other monitoring methods would be implemented similarly ...
    # Placeholder implementations for brevity
    
    async def _monitor_api_performance(self):
        while self.monitoring_active:
            await asyncio.sleep(60) # Placeholder

    async def _monitor_user_sessions(self):
        while self.monitoring_active:
            await asyncio.sleep(300) # Placeholder
            
    async def _monitor_security_events(self):
        while self.monitoring_active:
            await asyncio.sleep(60) # Placeholder
            
    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        while self.monitoring_active:
            try:
                health = await self._calculate_system_health()
                self.health_history.append(health)
                if len(self.health_history) > 100:
                    self.health_history.pop(0)
                
                await asyncio.to_thread(
                    self.db_manager.store_advanced_metric,
                    'system_health', 'overall_score', health.overall_score, health.to_dict()
                )
                
                if health.overall_score < 0.5:
                    await self._handle_critical_health_issue(health)
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(30)

    async def _calculate_system_health(self) -> SystemHealth:
        """Calculate comprehensive system health score"""
        try:
            # FIX: Run all blocking psutil calls in a thread to prevent blocking the event loop.
            cpu_percent, memory, disk = await asyncio.to_thread(lambda: (
                psutil.cpu_percent(interval=1),
                psutil.virtual_memory(),
                psutil.disk_usage('/')
            ))

            cpu_health = max(0, 1 - (cpu_percent / 100))
            memory_health = max(0, 1 - (memory.percent / 100))
            disk_health = max(0, 1 - (disk.percent / 100))
            
            # NOTE: These should be calculated from real-time data.
            network_health, database_health = 0.9, 0.95
            
            weights = {'cpu': 0.25, 'memory': 0.25, 'disk': 0.20, 'network': 0.15, 'database': 0.15}
            overall_score = (
                cpu_health * weights['cpu'] + memory_health * weights['memory'] +
                disk_health * weights['disk'] + network_health * weights['network'] +
                database_health * weights['database']
            )
            return SystemHealth(
                cpu_health, memory_health, disk_health, network_health,
                database_health, overall_score, datetime.utcnow().isoformat()
            )
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return SystemHealth(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, datetime.utcnow().isoformat())

    async def _handle_critical_health_issue(self, health: SystemHealth):
        """Handle critical system health issues"""
        message = f"System health critical: {health.overall_score:.2f}"
        logger.critical(message)
        await asyncio.to_thread(
            self.db_manager.store_system_alert,
            'system_health', 'critical', message, health.overall_score
        )
        await self._attempt_system_remediation(health)

    async def _attempt_system_remediation(self, health: SystemHealth):
        """Attempt automatic system remediation"""
        logger.info("Attempting automatic system remediation...")
        if health.memory_health < 0.3:
            # FIX: gc.collect() is blocking and should be run in a thread.
            await asyncio.to_thread(gc.collect)
            logger.info("Performed garbage collection")
        if health.database_health < 0.3:
            await asyncio.to_thread(self.db_manager.optimize_database)
            logger.info("Performed database optimization")

    # NOTE: All background tasks that perform blocking operations should follow the same pattern.
    async def _cleanup_old_data(self):
        while self.monitoring_active:
            await asyncio.sleep(21600) # 6 hours
            logger.info("Starting old data cleanup...")
            await asyncio.to_thread(self.db_manager.cleanup_old_data, days_to_keep=7)
    
    async def _optimize_system_performance(self):
        while self.monitoring_active:
            await asyncio.sleep(7200) # 2 hours
            logger.info("Optimizing system performance...")
            await asyncio.to_thread(self.db_manager.optimize_database)
            await asyncio.to_thread(gc.collect)
            logger.info("System performance optimization completed")

    async def _backup_critical_data(self):
        while self.monitoring_active:
            await asyncio.sleep(86400) # Daily
            logger.info("Backing up data (simulation)...")

    async def _update_performance_baselines(self):
        while self.monitoring_active:
            await asyncio.sleep(86400) # Daily
            logger.info("Updating performance baselines (simulation)...")

    async def _generate_health_reports(self):
        while self.monitoring_active:
            await asyncio.sleep(3600) # Hourly
            logger.info("Generating health reports (simulation)...")
            
    async def _train_anomaly_models(self):
        """Train anomaly detection models"""
        logger.info("Training anomaly detection models (simulation)...")
        await asyncio.sleep(5)
        logger.info("Anomaly detection models trained")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            current_health = self.health_history[-1] if self.health_history else None
            uptime = time.time() - self._start_time if self._start_time else 0
            return {
                'status': 'running' if self.monitoring_active else 'stopped',
                'uptime_seconds': uptime,
                # FIX: Handled case where current_health is None.
                'current_health': current_health.to_dict() if current_health else {},
                'active_tasks': len([t for t in self.monitoring_tasks if not t.done()]),
                'active_connections': len(self.active_connections),
                'performance_metrics': self.performance_metrics,
                'database_stats': self.db_manager.get_database_stats(),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'status': 'error', 'message': str(e)}

    async def shutdown_system(self):
        """Graceful system shutdown"""
        if not self.monitoring_active:
            return
        logger.info("Initiating system shutdown...")
        self.monitoring_active = False
        
        # Cancel all running background tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # FIX: Corrected typo 'Tr' to 'True'.
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        await self._shutdown_components()
        
        try:
            # FIX: Offload potentially blocking file I/O.
            await asyncio.to_thread(self.anomaly_detector.save_models, 'data/models/anomaly_models.pkl')
        except Exception as e:
            logger.warning(f"Could not save anomaly models: {e}")
        
        # Shutdown executors
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("System shutdown completed")

    async def _shutdown_components(self):
        """Shutdown individual components"""
        # NOTE: Ensure stop_monitoring is non-blocking or run it in a thread.
        self.network_monitor.stop_monitoring()
        logger.info("Components shutdown completed")

# Global system orchestrator instance (Singleton pattern)
system_orchestrator = None

def get_system_orchestrator():
    """Get global system orchestrator instance"""
    global system_orchestrator
    if system_orchestrator is None:
        system_orchestrator = SystemOrchestrator()
    return system_orchestrator
