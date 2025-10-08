"""
System Alerts and Notification System
Author: Member 2
"""
import smtplib
import json
import time
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging
import threading
from queue import Queue, Empty
import jinja2

logger = logging.getLogger(__name__)


# =====================
# Data Structures
# =====================
@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    metric_name: str
    value: float
    threshold: float
    severity: str  # info, warning, critical
    message: str
    timestamp: str
    source: str
    acknowledged: bool = False
    resolved: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    escalation_level: int = 0


# =====================
# Alert Rule Definition
# =====================
class AlertRule:
    """Alert rule definition"""

    def __init__(self, name: str, metric_name: str, condition: str,
                 threshold: float, severity: str, cooldown_minutes: int = 5):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition  # '>', '<', '>=', '<=', '==', '!='
        self.threshold = threshold
        self.severity = severity
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = None
        self.enabled = True

    def should_trigger(self, value: float) -> bool:
        """Check if the rule should trigger based on the value"""
        if not self.enabled:
            return False

        # Check cooldown period
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                return False

        # Check condition
        if self.condition == '>':
            return value > self.threshold
        elif self.condition == '<':
            return value < self.threshold
        elif self.condition == '>=':
            return value >= self.threshold
        elif self.condition == '<=':
            return value <= self.threshold
        elif self.condition == '==':
            return value == self.threshold
        elif self.condition == '!=':
            return value != self.threshold
        return False

    def trigger(self) -> None:
        """Mark rule as triggered"""
        self.last_triggered = datetime.utcnow()


# =====================
# Notification Channels
# =====================
class NotificationChannel:
    """Base class for notification channels"""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.messages_sent = 0
        self.messages_failed = 0

    async def send_notification(self, alert: Alert) -> bool:
        raise NotImplementedError

    def get_statistics(self) -> Dict:
        return {
            'name': self.name,
            'enabled': self.enabled,
            'messages_sent': self.messages_sent,
            'messages_failed': self.messages_failed
        }


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""

    def __init__(self, name: str, smtp_server: str, smtp_port: int,
                 username: str, password: str, from_email: str, to_emails: List[str]):
        super().__init__(name)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails

        self.template_env = jinja2.Environment(
            loader=jinja2.DictLoader({
                'alert_email': """
                <html>
                <body>
                <h2>System Monitor Alert</h2>
                <div>
                    <h3>{{ alert.severity.upper() }}: {{ alert.metric_name }}</h3>
                    <p><strong>Message:</strong> {{ alert.message }}</p>
                    <p><strong>Value:</strong> {{ alert.value }}</p>
                    <p><strong>Threshold:</strong> {{ alert.threshold }}</p>
                    <p><strong>Time:</strong> {{ alert.timestamp }}</p>
                    <p><strong>Source:</strong> {{ alert.source }}</p>
                    <p><strong>Alert ID:</strong> {{ alert.alert_id }}</p>
                </div>
                <p><em>This is an automated message from System Monitor.</em></p>
                </body>
                </html>
                """
            })
        )

    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[System Monitor] {alert.severity.upper()}: {alert.metric_name}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)

            # Render HTML content
            template = self.template_env.get_template('alert_email')
            html_content = template.render(alert=alert)
            msg.attach(MIMEText(html_content, 'html'))

            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            self.messages_sent += 1
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
        except Exception as e:
            self.messages_failed += 1
            logger.error(f"Email notification failed: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""

    def __init__(self, name: str, webhook_url: str, channel: str = "#alerts"):
        super().__init__(name)
        self.webhook_url = webhook_url
        self.channel = channel

    async def send_notification(self, alert: Alert) -> bool:
        try:
            color_map = {
                'info': '#36a64f',
                'warning': '#ff9800',
                'critical': '#ff5722'
            }

            slack_message = {
                'channel': self.channel,
                'username': 'System Monitor',
                'icon_emoji': ':warning:',
                'attachments': [{
                    'color': color_map.get(alert.severity, '#36a64f'),
                    'title': f"{alert.severity.upper()}: {alert.metric_name}",
                    'text': alert.message,
                    'fields': [
                        {'title': 'Value', 'value': str(alert.value), 'short': True},
                        {'title': 'Threshold', 'value': str(alert.threshold), 'short': True},
                        {'title': 'Source', 'value': alert.source, 'short': True},
                        {'title': 'Alert ID', 'value': alert.alert_id, 'short': True}
                    ],
                    'ts': int(datetime.fromisoformat(alert.timestamp).timestamp())
                }]
            }

            response = requests.post(self.webhook_url, json=slack_message, timeout=10)
            if response.status_code == 200:
                self.messages_sent += 1
                logger.info(f"Slack alert sent for {alert.alert_id}")
                return True
            else:
                self.messages_failed += 1
                logger.error(f"Slack notification failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.messages_failed += 1
            logger.error(f"Slack notification failed: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel"""

    def __init__(self, name: str, webhook_url: str, headers: Dict = None):
        super().__init__(name)
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}

    async def send_notification(self, alert: Alert) -> bool:
        try:
            response = requests.post(
                self.webhook_url,
                json=asdict(alert),
                headers=self.headers,
                timeout=10
            )
            if 200 <= response.status_code < 300:
                self.messages_sent += 1
                logger.info(f"Webhook alert sent for {alert.alert_id}")
                return True
            else:
                self.messages_failed += 1
                logger.error(f"Webhook notification failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.messages_failed += 1
            logger.error(f"Webhook notification failed: {e}")
            return False


# =====================
# Alert Manager
# =====================
class AlertManager:
    """Central alert management system"""

    def __init__(self):
        self.rules = {}
        self.channels = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_queue = Queue()
        self.notification_workers = []
        self.running = False

        self.stats = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_metric': defaultdict(int),
            'notifications_sent': 0,
            'notifications_failed': 0
        }

        logger.info("Alert manager initialized")

    def add_rule(self, rule: AlertRule):
        self.rules[rule.name] = rule
        logger.info(f"Alert rule '{rule.name}' added")

    def remove_rule(self, rule_name: str):
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Alert rule '{rule_name}' removed")

    def add_notification_channel(self, channel: NotificationChannel):
        self.channels[channel.name] = channel
        logger.info(f"Notification channel '{channel.name}' added")

    def remove_notification_channel(self, channel_name: str):
        if channel_name in self.channels:
            del self.channels[channel_name]
            logger.info(f"Notification channel '{channel_name}' removed")

    def start(self, num_workers: int = 2):
        if self.running:
            return
        self.running = True
        for i in range(num_workers):
            worker = threading.Thread(target=self._notification_worker, args=(f"worker-{i}",))
            worker.daemon = True
            worker.start()
            self.notification_workers.append(worker)
        logger.info(f"Alert manager started with {num_workers} workers")

    def stop(self):
        self.running = False
        for worker in self.notification_workers:
            worker.join(timeout=5)
        self.notification_workers.clear()
        logger.info("Alert manager stopped")

    def check_metric(self, metric_name: str, value: float, source: str = 'system'):
        for rule_name, rule in self.rules.items():
            if rule.metric_name == metric_name and rule.should_trigger(value):
                alert_id = f"{rule_name}-{int(time.time())}"
                alert = Alert(
                    alert_id=alert_id,
                    metric_name=metric_name,
                    value=value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=f"{metric_name} is {value} (threshold: {rule.condition} {rule.threshold})",
                    timestamp=datetime.utcnow().isoformat(),
                    source=source
                )
                self._trigger_alert(alert, rule)

    def _trigger_alert(self, alert: Alert, rule: AlertRule):
        rule.trigger()
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.stats['total_alerts'] += 1
        self.stats['alerts_by_severity'][alert.severity] += 1
        self.stats['alerts_by_metric'][alert.metric_name] += 1
        self.alert_queue.put(alert)
        logger.warning(f"Alert triggered: {alert.alert_id} - {alert.message}")

    def _notification_worker(self, worker_name: str):
        logger.info(f"Notification worker '{worker_name}' started")
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._send_notifications(alert)
                self.alert_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Notification worker error: {e}")
        logger.info(f"Notification worker '{worker_name}' stopped")

    def _send_notifications(self, alert: Alert):
        for channel_name, channel in self.channels.items():
            if not channel.enabled:
                continue
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(channel.send_notification(alert))
                if success:
                    self.stats['notifications_sent'] += 1
                else:
                    self.stats['notifications_failed'] += 1
                loop.close()
            except Exception as e:
                logger.error(f"Notification sending error for channel '{channel_name}': {e}")
                self.stats['notifications_failed'] += 1

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow().isoformat()
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")

    def resolve_alert(self, alert_id: str):
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow().isoformat()
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved")

    def get_active_alerts(self) -> List[Alert]:
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [a for a in self.alert_history if datetime.fromisoformat(a.timestamp) >= cutoff_time]

    def get_statistics(self) -> Dict:
        channel_stats = {n: c.get_statistics() for n, c in self.channels.items()}
        return {
            **self.stats,
            'active_alerts_count': len(self.active_alerts),
            'rules_count': len(self.rules),
            'channels_count': len(self.channels),
            'channel_statistics': channel_stats,
            'rules': {
                name: {
                    'metric_name': rule.metric_name,
                    'condition': rule.condition,
                    'threshold': rule.threshold,
                    'severity': rule.severity,
                    'enabled': rule.enabled,
                    'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
                } for name, rule in self.rules.items()
            }
        }


# Global instance
alert_manager = AlertManager()
