"""
Advanced Network Monitoring
Author: Member 2
"""

import os
import time
import socket
import subprocess
import threading
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional
import psutil
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NetworkAdvancedMonitor:
    """Advanced network monitoring and analysis"""

    def __init__(self, history_size=1000):
        self.history_size = history_size
        self.bandwidth_history = deque(maxlen=history_size)
        self.connection_history = deque(maxlen=history_size)
        self.latency_history = defaultdict(lambda: deque(maxlen=100))
        self.last_io_counters = None
        self.last_measurement_time = None
        self.monitoring_active = False
        self.monitor_thread = None

    # -----------------------------
    # MONITORING CONTROL FUNCTIONS
    # -----------------------------
    def start_monitoring(self, interval=1.0):
        """Start continuous network monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Advanced network monitoring started")

    def stop_monitoring(self):
        """Stop continuous network monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Advanced network monitoring stopped")

    # -----------------------------
    # INTERNAL DATA COLLECTION
    # -----------------------------
    def _monitor_loop(self, interval):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_bandwidth_data()
                self._collect_connection_data()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Network monitoring loop error: {e}")
                time.sleep(interval)

    def _collect_bandwidth_data(self):
        """Collect bandwidth utilization data"""
        try:
            current_io = psutil.net_io_counters()
            current_time = time.time()

            if self.last_io_counters and self.last_measurement_time:
                time_delta = current_time - self.last_measurement_time
                if time_delta > 0:
                    bytes_sent_rate = (current_io.bytes_sent - self.last_io_counters.bytes_sent) / time_delta
                    bytes_recv_rate = (current_io.bytes_recv - self.last_io_counters.bytes_recv) / time_delta
                    packets_sent_rate = (current_io.packets_sent - self.last_io_counters.packets_sent) / time_delta
                    packets_recv_rate = (current_io.packets_recv - self.last_io_counters.packets_recv) / time_delta

                    bandwidth_data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'bytes_sent_per_sec': bytes_sent_rate,
                        'bytes_recv_per_sec': bytes_recv_rate,
                        'packets_sent_per_sec': packets_sent_rate,
                        'packets_recv_per_sec': packets_recv_rate,
                        'total_bytes_sent': current_io.bytes_sent,
                        'total_bytes_recv': current_io.bytes_recv,
                        'total_packets_sent': current_io.packets_sent,
                        'total_packets_recv': current_io.packets_recv,
                        'errors_in': current_io.errin,
                        'errors_out': current_io.errout,
                        'drops_in': current_io.dropin,
                        'drops_out': current_io.dropout
                    }

                    self.bandwidth_history.append(bandwidth_data)

            self.last_io_counters = current_io
            self.last_measurement_time = current_time

        except Exception as e:
            logger.error(f"Bandwidth data collection error: {e}")

    def _collect_connection_data(self):
        """Collect network connection data"""
        try:
            connections = psutil.net_connections(kind='inet')
            connection_stats = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_connections': len(connections),
                'by_status': defaultdict(int),
                'by_family': defaultdict(int),
                'by_type': defaultdict(int),
                'by_local_port': defaultdict(int),
                'top_processes': defaultdict(int)
            }

            for conn in connections:
                # Count by status
                connection_stats['by_status'][conn.status] += 1

                # Count by family
                if conn.family.name == 'AF_INET':
                    family_name = 'IPv4'
                elif conn.family.name == 'AF_INET6':
                    family_name = 'IPv6'
                else:
                    family_name = 'Other'
                connection_stats['by_family'][family_name] += 1

                # Count by type
                if conn.type.name == 'SOCK_STREAM':
                    type_name = 'TCP'
                elif conn.type.name == 'SOCK_DGRAM':
                    type_name = 'UDP'
                else:
                    type_name = 'Other'
                connection_stats['by_type'][type_name] += 1

                # Count by local port
                if conn.laddr:
                    connection_stats['by_local_port'][conn.laddr.port] += 1

                # Count by process
                if conn.pid:
                    try:
                        proc = psutil.Process(conn.pid)
                        connection_stats['top_processes'][proc.name()] += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            # Convert defaultdicts to normal dicts for JSON compatibility
            for key in ['by_status', 'by_family', 'by_type', 'by_local_port', 'top_processes']:
                connection_stats[key] = dict(connection_stats[key])

            self.connection_history.append(connection_stats)

        except Exception as e:
            logger.error(f"Connection data collection error: {e}")

    # -----------------------------
    # PUBLIC API FUNCTIONS
    # -----------------------------
    def get_interface_statistics(self) -> Dict:
        """Get detailed statistics for all network interfaces"""
        try:
            interfaces = psutil.net_io_counters(pernic=True)
            interface_stats = {}

            for interface_name, stats in interfaces.items():
                interface_info = {
                    'name': interface_name,
                    'bytes_sent': stats.bytes_sent,
                    'bytes_recv': stats.bytes_recv,
                    'packets_sent': stats.packets_sent,
                    'packets_recv': stats.packets_recv,
                    'errors_in': stats.errin,
                    'errors_out': stats.errout,
                    'drops_in': stats.dropin,
                    'drops_out': stats.dropout
                }

                try:
                    if_addrs = psutil.net_if_addrs().get(interface_name, [])
                    if_stats = psutil.net_if_stats().get(interface_name)
                    addresses = []

                    for addr in if_addrs:
                        addr_info = {
                            'family': str(addr.family.name) if hasattr(addr.family, 'name') else str(addr.family),
                            'address': addr.address,
                            'netmask': addr.netmask,
                            'broadcast': addr.broadcast
                        }
                        addresses.append(addr_info)

                    interface_info['addresses'] = addresses

                    if if_stats:
                        interface_info['is_up'] = if_stats.isup
                        interface_info['speed_mbps'] = if_stats.speed
                        interface_info['mtu'] = if_stats.mtu
                except Exception as e:
                    logger.warning(f"Error getting interface details for {interface_name}: {e}")

                interface_stats[interface_name] = interface_info

            return {
                'interfaces': interface_stats,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Interface statistics error: {e}")
            return {'interfaces': {}, 'error': str(e)}

    def ping_latency_test(self, host: str, count: int = 4) -> Dict:
        """Test ping latency to a host"""
        try:
            if not self._is_valid_host(host):
                return {'error': 'Invalid host'}

            # Use system ping command
            ping_cmd = ['ping', '-c', str(count), host]
            if os.name == 'nt':  # Windows
                ping_cmd = ['ping', '-n', str(count), host]

            result = subprocess.run(ping_cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                latency_data = self._parse_ping_output(result.stdout)
                self.latency_history[host].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'latency_data': latency_data
                })
                return {
                    'host': host,
                    'success': True,
                    'latency_ms': latency_data,
                    'raw_output': result.stdout,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'host': host,
                    'success': False,
                    'error': result.stderr,
                    'timestamp': datetime.utcnow().isoformat()
                }

        except subprocess.TimeoutExpired:
            return {'host': host, 'success': False, 'error': 'Ping timeout'}
        except Exception as e:
            return {'host': host, 'success': False, 'error': str(e)}

    def _is_valid_host(self, host: str) -> bool:
        """Validate if host is a valid hostname or IP"""
        try:
            socket.gethostbyname(host)
            return True
        except socket.gaierror:
            return False

    def _parse_ping_output(self, output: str) -> Dict:
        """Parse ping command output to extract latency statistics"""
        lines = output.split('\n')
        latencies = []

        for line in lines:
            if 'time=' in line:
                try:
                    time_part = line.split('time=')[1].split()[0]
                    latency = float(time_part.replace('ms', ''))
                    latencies.append(latency)
                except (IndexError, ValueError):
                    continue

        if latencies:
            return {
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'avg_ms': sum(latencies) / len(latencies),
                'count': len(latencies),
                'packet_loss_percent': 0 if len(latencies) > 0 else 100
            }
        else:
            return {
                'min_ms': None,
                'max_ms': None,
                'avg_ms': None,
                'count': 0,
                'packet_loss_percent': 100
            }

    def get_bandwidth_trend(self, minutes: int = 10) -> Dict:
        """Get bandwidth utilization trend"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_data = []

        for entry in self.bandwidth_history:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if entry_time >= cutoff_time:
                recent_data.append(entry)

        if not recent_data:
            return {'error': 'No recent data available'}

        sent_rates = [d['bytes_sent_per_sec'] for d in recent_data]
        recv_rates = [d['bytes_recv_per_sec'] for d in recent_data]

        return {
            'period_minutes': minutes,
            'samples': len(recent_data),
            'bytes_sent_per_sec': {
                'avg': sum(sent_rates) / len(sent_rates),
                'max': max(sent_rates),
                'min': min(sent_rates)
            },
            'bytes_recv_per_sec': {
                'avg': sum(recv_rates) / len(recv_rates),
                'max': max(recv_rates),
                'min': min(recv_rates)
            },
            'total_data_transferred_mb': {
                'sent': sum(sent_rates) * 60 * minutes / (1024 * 1024),
                'received': sum(recv_rates) * 60 * minutes / (1024 * 1024)
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_connection_analysis(self) -> Dict:
        """Get detailed connection analysis"""
        if not self.connection_history:
            return {'error': 'No connection data available'}

        latest_data = self.connection_history[-1]
        if len(self.connection_history) >= 2:
            previous_data = self.connection_history[-2]
            connection_change = latest_data['total_connections'] - previous_data['total_connections']
        else:
            connection_change = 0

        return {
            'current_connections': latest_data['total_connections'],
            'connection_change': connection_change,
            'by_status': latest_data['by_status'],
            'by_family': latest_data['by_family'],
            'by_type': latest_data['by_type'],
            'top_ports': latest_data['by_local_port'],
            'top_processes': latest_data['top_processes'],
            'timestamp': latest_data['timestamp']
        }


# Global instance
network_advanced = NetworkAdvancedMonitor()
