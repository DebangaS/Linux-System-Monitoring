"""
Health monitoring and diagnostics
Author: Member 1
"""

import psutil
import time
from datetime import datetime, timedelta
from database.models import DatabaseManager
import redis
from flask import current_app


class HealthMonitor:
    """System health monitoring"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.redis_client = None
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = current_app.config.get('CACHE_REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            current_app.logger.warning(f"Redis connection failed: {e}")

    def check_database(self):
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            stats = self.db_manager.get_database_stats()
            response_time = time.time() - start_time
            return {
                'status': 'healthy',
                'response_time_ms': round(response_time * 1000, 2),
                'database_size_mb': stats.get('db_size_mb', 0),
                'metrics_count': stats.get('metrics_count', 0),
                'last_check': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }

    def check_redis(self):
        """Check Redis connectivity"""
        if not self.redis_client:
            return {
                'status': 'unavailable',
                'message': 'Redis not configured'
            }
        try:
            start_time = time.time()
            self.redis_client.ping()
            response_time = time.time() - start_time
            info = self.redis_client.info()
            return {
                'status': 'healthy',
                'response_time_ms': round(response_time * 1000, 2),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_mb': round(info.get('used_memory', 0) / 1024 / 1024, 2),
                'uptime_seconds': info.get('uptime_in_seconds', 0),
                'last_check': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }

    def check_system_resources(self):
        """Check system resource utilization"""
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_status = 'healthy'
            if cpu_percent > 90:
                cpu_status = 'critical'
            elif cpu_percent > 75:
                cpu_status = 'warning'

            # Memory check
            memory = psutil.virtual_memory()
            memory_status = 'healthy'
            if memory.percent > 90:
                memory_status = 'critical'
            elif memory.percent > 80:
                memory_status = 'warning'

            # Disk check
            disk = psutil.disk_usage('/')
            disk_status = 'healthy'
            if disk.percent > 95:
                disk_status = 'critical'
            elif disk.percent > 85:
                disk_status = 'warning'

            return {
                'status': 'healthy',
                'cpu': {
                    'usage_percent': cpu_percent,
                    'status': cpu_status
                },
                'memory': {
                    'usage_percent': memory.percent,
                    'status': memory_status,
                    'available_mb': round(memory.available / 1024 / 1024, 2)
                },
                'disk': {
                    'usage_percent': disk.percent,
                    'status': disk_status,
                    'free_mb': round(disk.free / 1024 / 1024, 2)
                },
                'last_check': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }

    def check_application_metrics(self):
        """Check application-specific metrics"""
        try:
            current_time = datetime.utcnow()
            recent_metrics = self.db_manager.get_historical_metrics('cpu', hours=1)

            if not recent_metrics:
                status = 'warning'
                message = 'No recent metrics found'
            elif len(recent_metrics) < 10:  # Less than expected for 1 hour
                status = 'warning'
                message = 'Low metric collection rate'
            else:
                status = 'healthy'
                message = 'Metrics collection normal'

            return {
                'status': status,
                'message': message,
                'recent_metrics_count': len(recent_metrics),
                'last_metric_time': recent_metrics[0].get('timestamp') if recent_metrics else None,
                'last_check': current_time.isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }

    def get_comprehensive_health(self):
        """Get comprehensive health status"""
        health_data = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {
                'database': self.check_database(),
                'redis': self.check_redis(),
                'system_resources': self.check_system_resources(),
                'application_metrics': self.check_application_metrics()
            }
        }

        # Determine overall status
        unhealthy_checks = []
        warning_checks = []
        for check_name, check_result in health_data['checks'].items():
            if check_result['status'] == 'unhealthy':
                unhealthy_checks.append(check_name)
            elif check_result['status'] in ['warning', 'critical']:
                warning_checks.append(check_name)

        if unhealthy_checks:
            health_data['overall_status'] = 'unhealthy'
        elif warning_checks:
            health_data['overall_status'] = 'warning'

        health_data['summary'] = {
            'unhealthy_checks': unhealthy_checks,
            'warning_checks': warning_checks,
            'healthy_checks': len(health_data['checks']) - len(unhealthy_checks) - len(warning_checks)
        }

        return health_data


# Global health monitor instance
health_monitor = HealthMonitor()
