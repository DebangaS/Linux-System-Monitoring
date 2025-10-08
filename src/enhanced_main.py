"""
Enhanced System Monitor - Main Application with Advanced Features
Author: Member 2
Updated for Day 4 - Advanced Monitoring & Intelligence
"""

import asyncio
import time
import argparse
from datetime import datetime
import logging
import signal
import sys
from typing import Dict, List

# Import monitoring modules
from monitors.gpu_monitor import gpu_monitor, hardware_monitor
from monitors.network_advanced import network_advanced
from intelligence.anomaly_detector import anomaly_detector, predictive_monitor

# Import streaming and alert systems
from streaming.realtime_processor import stream_manager, delta_compressor
from alerts.notification_system import alert_manager, AlertRule, EmailNotificationChannel, SlackNotificationChannel

# Import system metrics and database
from modules.cpu_memory import get_cpu_usage, get_memory_usage, get_disk_usage
from modules.processes import list_processes, get_top_processes
from database.models import DatabaseManager

logger = logging.getLogger(__name__)


class EnhancedSystemMonitor:
    """Enhanced system monitor with advanced capabilities"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.monitoring_active = False
        self.data_collection_interval = 5.0  # seconds
        self.anomaly_detection_enabled = True
        self.streaming_enabled = True
        self.alerts_enabled = True

        # Initialize components
        self._setup_alert_rules()
        self._setup_notification_channels()
        self._setup_data_streams()

        # Statistics
        self.collection_stats = {
            'samples_collected': 0,
            'anomalies_detected': 0,
            'alerts_triggered': 0,
            'started_at': None
        }

        logger.info("Enhanced system monitor initialized")

    # --------------------- ALERTS ---------------------
    def _setup_alert_rules(self):
        """Setup default alert rules"""
        alert_rules = [
            AlertRule('high_cpu', 'cpu_percent', '>', 80.0, 'warning', 5),
            AlertRule('critical_cpu', 'cpu_percent', '>', 95.0, 'critical', 2),
            AlertRule('high_memory', 'memory_percent', '>', 85.0, 'warning', 5),
            AlertRule('critical_memory', 'memory_percent', '>', 95.0, 'critical', 2),
            AlertRule('high_disk', 'disk_percent', '>', 90.0, 'warning', 10),
            AlertRule('critical_disk', 'disk_percent', '>', 98.0, 'critical', 5),
            AlertRule('high_temperature', 'temperature', '>', 75.0, 'warning', 5),
            AlertRule('critical_temperature', 'temperature', '>', 85.0, 'critical', 2)
        ]
        for rule in alert_rules:
            alert_manager.add_rule(rule)

    def _setup_notification_channels(self):
        """Setup notification channels (configure as needed)"""
        # Email notifications (requires valid credentials)
        try:
            email_channel = EmailNotificationChannel(
                name='email',
                smtp_server='smtp.gmail.com',
                smtp_port=587,
                username='your-email@gmail.com',
                password='your-app-password',
                from_email='your-email@gmail.com',
                to_emails=['admin@yourcompany.com']
            )
            # Uncomment after SMTP setup
            # alert_manager.add_notification_channel(email_channel)
        except Exception as e:
            logger.info(f"Email notifications not configured: {e}")

        # Slack notifications (requires valid webhook)
        try:
            slack_channel = SlackNotificationChannel(
                name='slack',
                webhook_url='https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
                channel='#alerts'
            )
            # Uncomment after Slack setup
            # alert_manager.add_notification_channel(slack_channel)
        except Exception as e:
            logger.info(f"Slack notifications not configured: {e}")

    # --------------------- STREAM SETUP ---------------------
    def _setup_data_streams(self):
        """Setup real-time data streams"""
        stream_manager.create_stream('system_metrics', buffer_size=500)
        stream_manager.create_stream('process_data', buffer_size=200)
        stream_manager.create_stream('gpu_data', buffer_size=100)
        stream_manager.create_stream('network_data', buffer_size=300)
        stream_manager.create_stream('alerts', buffer_size=100)
        logger.info("Data streams initialized")

    # --------------------- MONITORING LOOP ---------------------
    async def start_monitoring(self):
        """Start the enhanced monitoring system"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.collection_stats['started_at'] = datetime.utcnow().isoformat()

        if self.alerts_enabled:
            alert_manager.start()

        if self.streaming_enabled:
            network_advanced.start_monitoring(interval=self.data_collection_interval)

        logger.info("Enhanced system monitoring started")

        # Setup graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            asyncio.create_task(self.stop_monitoring())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.monitoring_active:
                await self._collect_all_metrics()
                await asyncio.sleep(self.data_collection_interval)
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
        finally:
            await self.stop_monitoring()

    # --------------------- DATA COLLECTION ---------------------
    async def _collect_all_metrics(self):
        """Collect all system metrics"""
        try:
            timestamp = datetime.utcnow().isoformat()

            # Basic system metrics
            cpu_data = self._collect_cpu_metrics(timestamp)
            memory_data = self._collect_memory_metrics(timestamp)
            disk_data = self._collect_disk_metrics(timestamp)

            # Advanced metrics
            gpu_data = self._collect_gpu_metrics(timestamp)
            network_data = self._collect_network_metrics(timestamp)
            process_data = self._collect_process_metrics(timestamp)
            hardware_data = self._collect_hardware_metrics(timestamp)

            # Store in database
            self.db_manager.store_system_metrics(cpu_data, memory_data, disk_data, network_data)

            # Anomaly detection
            if self.anomaly_detection_enabled:
                await self._run_anomaly_detection(cpu_data, memory_data, disk_data, gpu_data)

            # Stream data
            if self.streaming_enabled:
                await self._stream_data({
                    'cpu': cpu_data,
                    'memory': memory_data,
                    'disk': disk_data,
                    'gpu': gpu_data,
                    'network': network_data,
                    'processes': process_data,
                    'hardware': hardware_data
                })

            self.collection_stats['samples_collected'] += 1

        except Exception as e:
            logger.error(f"Metrics collection error: {e}")

    # --------------------- INDIVIDUAL METRIC COLLECTION ---------------------
    def _collect_cpu_metrics(self, timestamp: str) -> Dict:
        try:
            cpu_percent = get_cpu_usage(interval=0.1)
            return {'usage_percent': cpu_percent, 'timestamp': timestamp}
        except Exception as e:
            logger.error(f"CPU metrics collection error: {e}")
            return {'usage_percent': 0, 'timestamp': timestamp}

    def _collect_memory_metrics(self, timestamp: str) -> Dict:
        try:
            memory_info = get_memory_usage()
            return {**memory_info, 'timestamp': timestamp}
        except Exception as e:
            logger.error(f"Memory metrics collection error: {e}")
            return {'percent': 0, 'timestamp': timestamp}

    def _collect_disk_metrics(self, timestamp: str) -> Dict:
        try:
            disk_info = get_disk_usage()
            return {'partitions': disk_info, 'timestamp': timestamp}
        except Exception as e:
            logger.error(f"Disk metrics collection error: {e}")
            return {'partitions': [], 'timestamp': timestamp}

    def _collect_gpu_metrics(self, timestamp: str) -> Dict:
        try:
            return gpu_monitor.get_gpu_info()
        except Exception as e:
            logger.error(f"GPU metrics collection error: {e}")
            return {'gpus': [], 'error': str(e), 'timestamp': timestamp}

    def _collect_network_metrics(self, timestamp: str) -> Dict:
        try:
            interface_stats = network_advanced.get_interface_statistics()
            bandwidth_trend = network_advanced.get_bandwidth_trend(minutes=5)
            return {
                'interfaces': interface_stats.get('interfaces', {}),
                'bandwidth_trend': bandwidth_trend,
                'timestamp': timestamp
            }
        except Exception as e:
            logger.error(f"Network metrics collection error: {e}")
            return {'interfaces': {}, 'error': str(e), 'timestamp': timestamp}

    def _collect_process_metrics(self, timestamp: str) -> Dict:
        try:
            processes = list_processes(limit=20)
            top_cpu = get_top_processes(metric='cpu_percent', count=5)
            top_memory = get_top_processes(metric='memory_percent', count=5)
            return {
                'total_processes': len(processes),
                'top_cpu': top_cpu,
                'top_memory': top_memory,
                'timestamp': timestamp
            }
        except Exception as e:
            logger.error(f"Process metrics collection error: {e}")
            return {'total_processes': 0, 'error': str(e), 'timestamp': timestamp}

    def _collect_hardware_metrics(self, timestamp: str) -> Dict:
        try:
            temperature_data = hardware_monitor.get_temperature_sensors()
            fan_data = hardware_monitor.get_fan_sensors()
            battery_data = hardware_monitor.get_battery_info()
            return {
                'temperatures': temperature_data,
                'fans': fan_data,
                'battery': battery_data,
                'timestamp': timestamp
            }
        except Exception as e:
            logger.error(f"Hardware metrics collection error: {e}")
            return {'error': str(e), 'timestamp': timestamp}

    # --------------------- ANOMALY DETECTION ---------------------
    async def _run_anomaly_detection(self, cpu_data: Dict, memory_data: Dict, disk_data: Dict, gpu_data: Dict):
        try:
            if 'usage_percent' in cpu_data:
                anomaly_detector.add_sample('cpu_percent', cpu_data['usage_percent'])
                predictive_monitor.add_metric_sample('cpu_percent', cpu_data['usage_percent'])
            if 'percent' in memory_data:
                anomaly_detector.add_sample('memory_percent', memory_data['percent'])
                predictive_monitor.add_metric_sample('memory_percent', memory_data['percent'])

            # Trigger alerts if needed
            if self.alerts_enabled:
                if 'usage_percent' in cpu_data:
                    alert_manager.check_metric('cpu_percent', cpu_data['usage_percent'])
                if 'percent' in memory_data:
                    alert_manager.check_metric('memory_percent', memory_data['percent'])

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")

    # --------------------- STREAMING ---------------------
    async def _stream_data(self, metrics_data: Dict):
        try:
            compressed_data = delta_compressor.compress_data('system_metrics', {
                'cpu': metrics_data['cpu'],
                'memory': metrics_data['memory'],
                'disk': metrics_data['disk']
            })
            stream_manager.publish_message('system_metrics', 'metrics_update', compressed_data, 'enhanced_monitor', priority=2)

            if metrics_data['gpu'].get('gpus'):
                stream_manager.publish_message('gpu_data', 'gpu_update', metrics_data['gpu'], 'enhanced_monitor', priority=1)

            stream_manager.publish_message('network_data', 'network_update', metrics_data['network'], 'enhanced_monitor', priority=1)
            stream_manager.publish_message('process_data', 'process_update', metrics_data['processes'], 'enhanced_monitor', priority=1)

        except Exception as e:
            logger.error(f"Data streaming error: {e}")

    # --------------------- STOP ---------------------
    async def stop_monitoring(self):
        if not self.monitoring_active:
            return

        logger.info("Stopping enhanced system monitoring...")
        self.monitoring_active = False

        if self.alerts_enabled:
            alert_manager.stop()
        if self.streaming_enabled:
            network_advanced.stop_monitoring()

        try:
            anomaly_detector.save_models('data/anomaly_models.joblib')
        except Exception as e:
            logger.error(f"Error saving models: {e}")

        logger.info("Enhanced system monitoring stopped")

    # --------------------- STATUS ---------------------
    def get_comprehensive_status(self) -> Dict:
        try:
            return {
                'monitoring_active': self.monitoring_active,
                'collection_stats': self.collection_stats,
                'anomaly_summary': anomaly_detector.get_anomaly_summary(hours=1),
                'predictions': predictive_monitor.get_all_predictions(),
                'alert_stats': alert_manager.get_statistics(),
                'stream_stats': stream_manager.get_global_statistics(),
                'compression_stats': delta_compressor.get_compression_stats(),
                'gpu_status': gpu_monitor.get_gpu_info(),
                'network_analysis': network_advanced.get_connection_analysis(),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Status collection error: {e}")
            return {'error': str(e)}


# --------------------- MAIN ENTRY ---------------------
def main():
    parser = argparse.ArgumentParser(description="Enhanced System Monitor")
    parser.add_argument('--interval', type=float, default=5.0, help='Data collection interval in seconds')
    parser.add_argument('--no-anomaly-detection', action='store_true', help='Disable anomaly detection')
    parser.add_argument('--no-streaming', action='store_true', help='Disable real-time streaming')
    parser.add_argument('--no-alerts', action='store_true', help='Disable alert system')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    monitor = EnhancedSystemMonitor()
    monitor.data_collection_interval = args.interval
    monitor.anomaly_detection_enabled = not args.no_anomaly_detection
    monitor.streaming_enabled = not args.no_streaming
    monitor.alerts_enabled = not args.no_alerts

    try:
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
