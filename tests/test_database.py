"""
Comprehensive database tests for system monitoring data
Author: Member 5
"""

import unittest
import sys
import os
import tempfile
import sqlite3
import json
from datetime import datetime, timedelta

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import DatabaseManager
from database.data_analyzer import DataAnalyzer


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.db_manager = DatabaseManager(db_path=self.temp_db)

        self.sample_cpu_data = {
            'usage_percent': 45.5,
            'count_physical': 4,
            'count_logical': 8,
            'per_core': [40.1, 50.2, 45.8, 44.3],
            'frequency': {'current': 2400.0, 'min': 800.0, 'max': 3200},
            'timestamp': datetime.utcnow().isoformat()
        }
        self.sample_memory_data = {
            'total': 16777216000,
            'available': 8388608000,
            'used': 8388608000,
            'free': 4194304000,
            'usage_percent': 50.0,
            'swap': {
                'total': 2147483648,
                'used': 0,
                'free': 2147483648,
                'usage_percent': 0.0,
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        self.sample_disk_data = {
            'partitions': [
                {
                    'device': '/dev/sda1',
                    'mountpoint': '/',
                    'filesystem': 'ext4',
                    'total': 1099511627776,
                    'used': 549755813888,
                    'free': 549755813888,
                    'usage_percent': 50.0,
                }
            ],
            'total_usage_percent': 50.0,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self.sample_network_data = {
            'bytes_sent': 1048576000,
            'bytes_recv': 2097152000,
            'packets_sent': 10000,
            'packets_recv': 15000,
            'sent_rate_kbps': 1024.0,
            'recv_rate_kbps': 2048.0,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self.sample_processes = [
            {
                'pid': 1234,
                'name': 'python',
                'cpu_percent': 15.5,
                'memory_percent': 5.2,
                'memory_mb': 256.0,
                'status': 'running',
                'username': 'testuser',
            },
            {
                'pid': 5678,
                'name': 'chrome',
                'cpu_percent': 8.3,
                'memory_percent': 12.1,
                'memory_mb': 512.0,
                'status': 'sleeping',
                'username': 'testuser',
            }
        ]

    def tearDown(self):
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)

    def test_database_initialization(self):
        self.assertTrue(os.path.exists(self.temp_db))
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            expected_tables = ['system_metrics', 'process_snapshots', 'system_alerts', 'system_info']
            for table in expected_tables:
                self.assertIn(table, tables)

            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            expected_indexes = ['idx_metrics_timestamp', 'idx_metrics_type', 'idx_process_timestamp', 'idx_process_pid', 'idx_alerts_timestamp', 'idx_alerts_type']
            for index in expected_indexes:
                self.assertIn(index, indexes)

    def test_store_and_retrieve_metrics(self):
        result = self.db_manager.store_system_metrics(
            self.sample_cpu_data,
            self.sample_memory_data,
            self.sample_disk_data,
            self.sample_network_data,
        )
        self.assertTrue(result)
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM system_metrics")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 4)
            cursor.execute("SELECT data FROM system_metrics WHERE metric_type = 'cpu'")
            data = cursor.fetchone()
            if data:
                stored_data = json.loads(data[0])
                self.assertEqual(stored_data.get('usage_percent'), 45.5)

    def test_store_and_retrieve_snapshot(self):
        result = self.db_manager.store_process_snapshot(self.sample_processes)
        self.assertTrue(result)
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM process_snapshots")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 2)
            cursor.execute("SELECT name, cpu_percent FROM process_snapshots WHERE pid = 1234")
            row = cursor.fetchone()
            if row:
                name, cpu_percent = row
                self.assertEqual(name, "python")
                self.assertEqual(cpu_percent, 15.5)

    def test_store_and_retrieve_alert(self):
        result = self.db_manager.store_system_alert('cpu', 'warning', 'High CPU usage detected', 85.5)
        self.assertTrue(result)
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT alert_type, level, message, value FROM system_alerts")
            row = cursor.fetchone()
            if row:
                alert_type, level, message, value = row
                self.assertEqual(alert_type, 'cpu')
                self.assertEqual(level, 'warning')
                self.assertEqual(message, 'High CPU usage detected')
                self.assertEqual(value, 85.5)

    def test_historical_metrics_filtering(self):
        self.db_manager.store_system_metrics(
            self.sample_cpu_data,
            self.sample_memory_data,
            self.sample_disk_data,
            self.sample_network_data,
        )
        old_data = dict(self.sample_cpu_data)
        old_data['timestamp'] = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_metrics (timestamp, metric_type, data) VALUES (?, ?, ?)",
                (old_data['timestamp'], 'cpu', json.dumps(old_data))
            )
            conn.commit()
        recent_data = self.db_manager.get_historical_metrics('cpu', hours=24)
        self.assertEqual(len(recent_data), 1)

    def test_data_integrity(self):
        invalid_data = [{'pid': 'invalid', 'name': None, 'cpu_percent': 'high'}]
        result = self.db_manager.store_process_snapshot(invalid_data)
        self.assertTrue(result)

    def test_database_stats(self):
        self.db_manager.store_system_metrics(
            self.sample_cpu_data,
            self.sample_memory_data,
            self.sample_disk_data,
            self.sample_network_data,
        )
        stats = self.db_manager.get_database_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('metrics_count', stats)
        self.assertIn('db_size', stats)
        self.assertGreater(stats['db_size'], 0)

    def test_cleanup_old_data(self):
        old_time = (datetime.utcnow() - timedelta(days=10)).isoformat()
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_metrics (timestamp, metric_type, data) VALUES (?, ?, ?)",
                (old_time, 'cpu', json.dumps(self.sample_cpu_data))
            )
            cursor.execute(
                "INSERT INTO system_alerts (timestamp, alert_type, level, message, acknowledged) VALUES (?, ?, ?, ?, ?)",
                (old_time, 'test', 'info', 'test message', 1)
            )
            conn.commit()
        self.db_manager.store_system_metrics(
            self.sample_cpu_data,
            self.sample_memory_data,
            self.sample_disk_data,
            self.sample_network_data,
        )
        result = self.db_manager.cleanup_old_data(days=7)
        self.assertTrue(result)
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM system_metrics")
            metrics_count = cursor.fetchone()[0]
            self.assertEqual(metrics_count, 4)


class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.db_manager = DatabaseManager(db_path=self.temp_db)
        self.data_analyzer = DataAnalyzer(db_path=self.temp_db)
        self.generate_test_data()

    def tearDown(self):
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)

    def generate_test_data(self):
        import random
        base_time = datetime.utcnow() - timedelta(hours=24)

        for i in range(288):
            timestamp = (base_time + timedelta(minutes=5 * i)).isoformat()
            cpu_data = {
                'usage_percent': random.uniform(20, 80) + (10 * (i % 12) / 12),
                'timestamp': timestamp
            }
            memory_data = {
                'total': 16777216000,
                'usage_percent': random.uniform(40, 90),
                'swap': {
                    'usage_percent': random.uniform(0, 20),
                },
                'timestamp': timestamp
            }
            disk_data = {
                'partitions': [
                    {
                        'device': '/dev/sda1',
                        'mountpoint': '/',
                        'filesystem': 'ext4',
                        'usage_percent': random.uniform(45, 55)
                    }
                ],
                'total_usage_percent': random.uniform(45, 55),
                'timestamp': timestamp
            }
            network_data = {
                'bytes_sent': random.randint(1e7, 1e9),
                'bytes_recv': random.randint(1e7, 1e9),
                'packets_sent': random.randint(1e3, 1e5),
                'packets_recv': random.randint(1e3, 1e5),
                'sent_rate_kbps': random.uniform(100, 1000),
                'recv_rate_kbps': random.uniform(200, 2000),
                'timestamp': timestamp
            }
            self.db_manager.store_system_metrics(cpu_data, memory_data, disk_data, network_data)

    def test_cpu_trends(self):
        trends = self.data_analyzer.get_cpu_trends(hours=24)
        self.assertIsInstance(trends, dict)
        self.assertIn('average', trends)
        self.assertIn('minimum', trends)
        self.assertIn('maximum', trends)
        self.assertIn('samples', trends)
        self.assertGreaterEqual(trends['minimum'], 0)
        self.assertLessEqual(trends['maximum'], 100)
        self.assertGreaterEqual(trends['average'], trends['minimum'])
        self.assertLessEqual(trends['average'], trends['maximum'])
        self.assertGreater(trends['samples'], 0)

    def test_memory_trends(self):
        trends = self.data_analyzer.get_memory_trends(hours=24)
        self.assertIsInstance(trends, dict)
        self.assertIn('memory', trends)
        self.assertIn('swap', trends)
        self.assertIn('samples', trends)
        self.assertIn('average', trends['memory'])
        self.assertIn('maximum', trends['memory'])
        self.assertGreaterEqual(trends['memory']['minimum'], 0)
        self.assertLessEqual(trends['memory']['maximum'], 100)

    def test_report_generation(self):
        report = self.data_analyzer.generate_report(hours=24)
        self.assertIsInstance(report, dict)
        self.assertIn('report_generated', report)
        self.assertIn('cpu_analysis', report)
        self.assertIn('memory_analysis', report)
        self.assertIn('disk_analysis', report)
        self.assertIn('network_analysis', report)
        self.assertTrue(len(report['cpu_analysis']) > 0)
        self.assertTrue(len(report['memory_analysis']) > 0)


class TestDatabasePerformance(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.db_manager = DatabaseManager(db_path=self.temp_db)

    def tearDown(self):
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)

    def test_bulk_insertion(self):
        import time
        test_data = []
        for i in range(1000):
            data = {
                'usage_percent': 50.0 + i % 50,
                'timestamp': (datetime.utcnow() - timedelta(minutes=i)).isoformat()
            }
            test_data.append(data)
        start = time.time()
        for data in test_data:
            self.db_manager.store_system_metrics(data, data, data, data)
        end = time.time()
        duration = end - start
        self.assertLess(duration, 30)

        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM system_metrics")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 4000)

    def test_query_performance(self):
        import time
        for i in range(500):
            data = {
                'usage_percent': 50.0 + i % 50,
                'timestamp': (datetime.utcnow() - timedelta(minutes=i)).isoformat()
            }
            self.db_manager.store_system_metrics(data, data, data, data)
        start = time.time()
        cpu_data = self.db_manager.get_historical_metrics('cpu', hours=24)
        duration = time.time() - start
        self.assertLess(duration, 5)
        self.assertGreater(len(cpu_data), 0)

    def test_concurrent_access(self):
        import threading
        import time

        def insert_worker(worker_id):
            for _ in range(50):
                data = {
                    'usage_percent': worker_id * 10,
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.db_manager.store_system_metrics(data, data, data, data)

        def query_worker():
            for _ in range(10):
                self.db_manager.get_historical_metrics('cpu', hours=1)
                time.sleep(0.1)

        threads = []
        for i in range(3):
            t = threading.Thread(target=insert_worker, args=(i,))
            threads.append(t)
            t.start()

        for i in range(2):
            t = threading.Thread(target=query_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM system_metrics")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 600)


if __name__ == "__main__":
    unittest.main()
