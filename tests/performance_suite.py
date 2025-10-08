""" Performance Testing & Load Testing Suite
Author: Member 5
"""

import asyncio
import aiohttp
import time
import threading
import multiprocessing
import psutil
import statistics
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import websockets
import requests
import sqlite3
import random
import queue

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    base_url: str
    websocket_url: str
    concurrent_users: int
    test_duration: int  # seconds
    ramp_up_time: int   # seconds
    endpoints: List[Dict[str, str]]
    user_scenarios: List[Dict[str, Any]]

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    timestamp: float
    response_time: float
    status_code: int
    endpoint: str
    user_id: int
    error: Optional[str] = None

class LoadTestRunner:
    """Load testing and stress testing runner"""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics = []
        self.active_users = 0
        self.test_start_time = 0
        self.test_stop_flag = threading.Event()
        # Performance monitoring
        self.system_metrics = []
        self.resource_monitor_thread = None
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Results storage
        self.results = {
            'configuration': asdict(config),
            'start_time': None,
            'end_time': None,
            'duration': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'throughput': 0,
            'error_rate': 0,
            'system_metrics': [],
            'endpoints_performance': {},
            'user_scenarios_performance': {}
        }

    def run_load_test(self) -> Dict:
        """Run complete load test"""
        self.logger.info("Starting load test...")
        self.logger.info(f"Configuration: {self.config.concurrent_users} users, {self.config.test_duration} seconds")
        self.test_start_time = time.time()
        self.results['start_time'] = datetime.fromtimestamp(self.test_start_time).isoformat()
        # Start system resource monitoring
        self.start_resource_monitoring()
        try:
            # Run the load test
            self.execute_load_test()
            # Calculate final results
            self.calculate_results()
        except Exception as e:
            self.logger.error(f"Load test failed: {str(e)}")
            self.results['error'] = str(e)
        finally:
            # Stop resource monitoring
            self.stop_resource_monitoring()
            self.results['end_time'] = datetime.now().isoformat()
            self.results['duration'] = time.time() - self.test_start_time
        self.logger.info("Load test completed")
        return self.results

    def execute_load_test(self):
        """Execute the main load test"""
        # Calculate user ramp-up schedule
        users_per_second = self.config.concurrent_users / self.config.ramp_up_time if self.config.ramp_up_time > 0 else self.config.concurrent_users
        # Start users gradually
        with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
            futures = []
            for user_id in range(self.config.concurrent_users):
                # Calculate start delay for this user
                start_delay = user_id / users_per_second if users_per_second > 0 else 0
                # Submit user task
                future = executor.submit(self.simulate_user, user_id, start_delay)
                futures.append(future)
            # Wait for test duration
            time.sleep(self.config.test_duration)
            # Signal stop
            self.test_stop_flag.set()
            # Wait for all users to finish
            for future in as_completed(futures, timeout=30):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"User simulation failed: {str(e)}")

    def simulate_user(self, user_id: int, start_delay: float):
        """Simulate a single user's behavior"""
        # Wait for ramp-up delay
        time.sleep(start_delay)
        self.active_users += 1
        session = requests.Session()
        try:
            while not self.test_stop_flag.is_set():
                # Select scenario randomly or sequentially
                if self.config.user_scenarios:
                    scenario = random.choice(self.config.user_scenarios)
                    self.execute_user_scenario(session, user_id, scenario)
                else:
                    # Default behavior: random endpoint selection
                    endpoint = random.choice(self.config.endpoints)
                    self.make_request(session, user_id, endpoint)
                # Think time (random delay between requests)
                think_time = random.uniform(1.0, 3.0)
                if self.test_stop_flag.wait(timeout=think_time):
                    break
        finally:
            session.close()
            self.active_users -= 1

    def execute_user_scenario(self, session: requests.Session, user_id: int, scenario: Dict[str, Any]):
        """Execute a specific user scenario"""
        scenario_name = scenario.get('name', 'default')
        steps = scenario.get('steps', [])
        for step in steps:
            if self.test_stop_flag.is_set():
                break
            endpoint = step.get('endpoint', {})
            think_time = step.get('think_time', 1.0)
            # Execute the step
            metric = self.make_request(session, user_id, endpoint)
            # Add scenario context
            if metric:
                setattr(metric, 'scenario', scenario_name)
            # Wait before next step
            if self.test_stop_flag.wait(timeout=think_time):
                break

    def make_request(self, session: requests.Session, user_id: int, endpoint: Dict[str, str]):
        """Make a single HTTP request and record metrics"""
        url = f"{self.config.base_url}{endpoint.get('path', '/')}"
        method = endpoint.get('method', 'GET').upper()
        headers = endpoint.get('headers', {})
        data = endpoint.get('data', {})
        start_time = time.time()
        error_msg = None
        try:
            if method == 'GET':
                response = session.get(url, headers=headers, params=data, timeout=30)
            elif method == 'POST':
                response = session.post(url, headers=headers, json=data, timeout=30)
            elif method == 'PUT':
                response = session.put(url, headers=headers, json=data, timeout=30)
            elif method == 'DELETE':
                response = session.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            status_code = response.status_code
        except requests.RequestException as e:
            status_code = 0
            error_msg = str(e)
            self.logger.warning(f"Request failed for user {user_id}: {error_msg}")
        response_time = time.time() - start_time
        # Record metric
        metric = PerformanceMetric(
            timestamp=start_time,
            response_time=response_time,
            status_code=status_code,
            endpoint=endpoint.get('path', '/'),
            user_id=user_id,
            error=error_msg
        )
        self.metrics.append(metric)
        return metric

    def start_resource_monitoring(self):
        """Start system resource monitoring"""
        def monitor_resources():
            while not self.test_stop_flag.is_set():
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    # Memory usage
                    memory = psutil.virtual_memory()
                    # Disk I/O
                    disk_io = psutil.disk_io_counters()
                    # Network I/O
                    network_io = psutil.net_io_counters()
                    # Process count
                    process_count = len(psutil.pids())
                    metric = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available': memory.available,
                        'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                        'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
                        'network_sent_bytes': network_io.bytes_sent if network_io else 0,
                        'network_recv_bytes': network_io.bytes_recv if network_io else 0,
                        'process_count': process_count,
                        'active_users': self.active_users
                    }
                    self.system_metrics.append(metric)
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {str(e)}")
                if self.test_stop_flag.wait(timeout=5):
                    break
        self.resource_monitor_thread = threading.Thread(target=monitor_resources)
        self.resource_monitor_thread.daemon = True
        self.resource_monitor_thread.start()

    def stop_resource_monitoring(self):
        """Stop system resource monitoring"""
        if self.resource_monitor_thread and self.resource_monitor_thread.is_alive():
            self.test_stop_flag.set()
            self.resource_monitor_thread.join(timeout=10)

    def calculate_results(self):
        """Calculate final test results"""
        if not self.metrics:
            return
        # Basic statistics
        total_requests = len(self.metrics)
        successful_requests = len([m for m in self.metrics if 200 <= m.status_code < 400])
        failed_requests = total_requests - successful_requests
        response_times = [m.response_time for m in self.metrics]
        # Calculate percentiles
        if response_times:
            response_times.sort()
            percentiles = {
                'p50': self.calculate_percentile(response_times, 50),
                'p90': self.calculate_percentile(response_times, 90),
                'p95': self.calculate_percentile(response_times, 95),
                'p99': self.calculate_percentile(response_times, 99)
            }
        else:
            percentiles = {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
        # Throughput (requests per second)
        test_duration = max(1, self.results['duration'])
        throughput = total_requests / test_duration
        # Error rate
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        # Update results
        self.results.update({
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'response_times': {
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'avg': statistics.mean(response_times) if response_times else 0,
                'median': statistics.median(response_times) if response_times else 0,
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                **percentiles
            },
            'throughput': throughput,
            'error_rate': error_rate,
            'system_metrics': self.system_metrics
        })
        # Analyze per-endpoint performance
        self.analyze_endpoint_performance()
        # Analyze user scenario performance if applicable
        self.analyze_scenario_performance()

    def calculate_percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not sorted_values:
            return 0
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def analyze_endpoint_performance(self):
        """Analyze performance per endpoint"""
        endpoint_metrics = {}
        for metric in self.metrics:
            endpoint = metric.endpoint
            if endpoint not in endpoint_metrics:
                endpoint_metrics[endpoint] = []
            endpoint_metrics[endpoint].append(metric)
        # Calculate statistics for each endpoint
        for endpoint, metrics_list in endpoint_metrics.items():
            response_times = [m.response_time for m in metrics_list]
            successful = [m for m in metrics_list if 200 <= m.status_code < 400]
            self.results['endpoints_performance'][endpoint] = {
                'total_requests': len(metrics_list),
                'successful_requests': len(successful),
                'error_rate': ((len(metrics_list) - len(successful)) / len(metrics_list) * 100) if len(metrics_list) > 0 else 0,
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'p95_response_time': self.calculate_percentile(sorted(response_times), 95) if response_times else 0
            }

    def analyze_scenario_performance(self):
        """Analyze user scenario performance"""
        # This would analyze performance by scenario if scenarios are defined
        # For now, we'll leave this as a placeholder
        pass

class StressTestRunner(LoadTestRunner):
    """Stress testing runner (extends load testing)"""

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.stress_phases = [
            {'name': 'baseline', 'users': config.concurrent_users // 4, 'duration': 300},
            {'name': 'ramp_up', 'users': config.concurrent_users // 2, 'duration': 300},
            {'name': 'peak', 'users': config.concurrent_users, 'duration': 600},
            {'name': 'spike', 'users': config.concurrent_users * 2, 'duration': 300},
            {'name': 'recovery', 'users': config.concurrent_users // 2, 'duration': 300}
        ]

    def run_stress_test(self) -> Dict:
        """Run comprehensive stress test"""
        self.logger.info("Starting stress test...")
        stress_results = {
            'phases': {},
            'breaking_point': None,
            'recovery_time': None,
            'system_stability': {}
        }
        for phase in self.stress_phases:
            self.logger.info(f"Starting stress phase: {phase['name']} ({phase['users']} users)")
            # Update configuration for this phase
            original_users = self.config.concurrent_users
            original_duration = self.config.test_duration
            self.config.concurrent_users = phase['users']
            self.config.test_duration = phase['duration']
            # Run phase
            phase_result = self.run_load_test()
            phase_result['phase_name'] = phase['name']
            stress_results['phases'][phase['name']] = phase_result
            # Analyze if this phase caused system failure
            if self.is_breaking_point(phase_result):
                stress_results['breaking_point'] = {
                    'phase': phase['name'],
                    'user_count': phase['users'],
                    'metrics': phase_result
                }
                break
            # Restore original configuration
            self.config.concurrent_users = original_users
            self.config.test_duration = original_duration
            # Cool-down period between phases
            self.logger.info("Cool-down period...")
            time.sleep(30)
        return stress_results

    def is_breaking_point(self, results: Dict) -> bool:
        """Determine if results indicate a breaking point"""
        error_rate = results.get('error_rate', 0)
        avg_response_time = results.get('response_times', {}).get('avg', 0)
        # Consider it a breaking point if:
        # - Error rate > 50%
        # - Average response time > 30 seconds
        return error_rate > 50 or avg_response_time > 30

class DatabasePerformanceTest:
    """Database performance testing"""

    def __init__(self, db_path: str = 'test_performance.db'):
        self.db_path = db_path
        self.setup_test_database()

    def setup_test_database(self):
        """Setup test database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Create test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_test_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                metric_type TEXT,
                metric_name TEXT,
                value REAL,
                metadata TEXT
            )
        ''')
        # Insert sample data
        sample_data = []
        base_time = datetime.now()
        for i in range(10000):  # 10k records
            timestamp = base_time + timedelta(seconds=i)
            sample_data.append((
                timestamp.isoformat(),
                random.choice(['cpu', 'memory', 'disk', 'network']),
                random.choice(['usage', 'utilization', 'bandwidth']),
                random.uniform(0, 100),
                json.dumps({'test': True, 'iteration': i})
            ))
        cursor.executemany(
            'INSERT INTO performance_test_metrics (timestamp, metric_type, metric_name, value, metadata) VALUES (?, ?, ?, ?, ?)',
            sample_data
        )
        conn.commit()
        conn.close()

    def run_database_performance_tests(self) -> Dict:
        """Run database performance tests"""
        results = {
            'insert_performance': self.test_insert_performance(),
            'select_performance': self.test_select_performance(),
            'update_performance': self.test_update_performance(),
            'complex_query_performance': self.test_complex_queries(),
            'concurrent_access': self.test_concurrent_database_access()
        }
        return results

    def test_insert_performance(self) -> Dict:
        """Test insert performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        start_time = time.time()
        # Insert 1000 records
        test_data = []
        for i in range(1000):
            test_data.append((
                datetime.now().isoformat(),
                'performance_test',
                f'insert_test_{i}',
                random.uniform(0, 100),
                json.dumps({'test': 'insert_performance', 'batch': i})
            ))
        cursor.executemany(
            'INSERT INTO performance_test_metrics (timestamp, metric_type, metric_name, value, metadata) VALUES (?, ?, ?, ?, ?)',
            test_data
        )
        conn.commit()
        conn.close()
        execution_time = time.time() - start_time
        return {
            'records_inserted': 1000,
            'execution_time': execution_time,
            'inserts_per_second': 1000 / execution_time
        }

    def test_select_performance(self) -> Dict:
        """Test select performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        queries = [
            'SELECT COUNT(*) FROM performance_test_metrics',
            'SELECT * FROM performance_test_metrics ORDER BY timestamp DESC LIMIT 100',
            'SELECT metric_type, AVG(value) FROM performance_test_metrics GROUP BY metric_type',
            'SELECT * FROM performance_test_metrics WHERE value > 50 AND timestamp >= datetime("now", "-1 hour")'
        ]
        results = {}
        for i, query in enumerate(queries):
            start_time = time.time()
            cursor.execute(query)
            records = cursor.fetchall()
            execution_time = time.time() - start_time
            results[f'query_{i+1}'] = {
                'query': query,
                'execution_time': execution_time,
                'records_returned': len(records)
            }
        conn.close()
        return results

    def test_update_performance(self) -> Dict:
        """Test update performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        start_time = time.time()
        # Update 500 records
        cursor.execute(
            'UPDATE performance_test_metrics SET value = value * 1.1 WHERE id <= 500'
        )
        conn.commit()
        conn.close()
        execution_time = time.time() - start_time
        return {
            'records_updated': 500,
            'execution_time': execution_time,
            'updates_per_second': 500 / execution_time if execution_time > 0 else 0
        }

    def test_complex_queries(self) -> Dict:
        """Test complex query performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        complex_query = '''
        SELECT
            metric_type,
            metric_name,
            COUNT(*) as sample_count,
            AVG(value) as avg_value,
            MIN(value) as min_value,
            MAX(value) as max_value,
            datetime(timestamp) as hour_bucket
        FROM performance_test_metrics
        WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY metric_type, metric_name, strftime('%H', timestamp)
        ORDER BY hour_bucket DESC, avg_value DESC
        '''
        start_time = time.time()
        cursor.execute(complex_query)
        results = cursor.fetchall()
        execution_time = time.time() - start_time
        conn.close()
        return {
            'query': 'Complex aggregation query',
            'execution_time': execution_time,
            'records_returned': len(results),
            'complexity_score': 'high'
        }

    def test_concurrent_database_access(self) -> Dict:
        """Test concurrent database access"""

        def worker(worker_id: int, results_queue: queue.Queue):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                start_time = time.time()
                # Perform mixed operations
                for i in range(10):
                    # Select
                    cursor.execute('SELECT COUNT(*) FROM performance_test_metrics WHERE metric_type = ?', (f'type_{worker_id}',))
                    cursor.fetchone()
                    # Insert
                    cursor.execute(
                        'INSERT INTO performance_test_metrics (timestamp, metric_type, metric_name, value, metadata) VALUES (?, ?, ?, ?, ?)',
                        (datetime.now().isoformat(), f'concurrent_test_{worker_id}', f'metric_{i}', random.uniform(0, 100), '{}')
                    )
                conn.commit()
                conn.close()
                execution_time = time.time() - start_time
                results_queue.put({
                    'worker_id': worker_id,
                    'execution_time': execution_time,
                    'operations_completed': 20  # 10 selects + 10 inserts
                })
            except Exception as e:
                results_queue.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'execution_time': float('inf')
                })

        # Test with 5 concurrent workers
        num_workers = 5
        results_queue = queue.Queue()
        threads = []
        start_time = time.time()
        for worker_id in range(num_workers):
            thread = threading.Thread(target=worker, args=(worker_id, results_queue))
            thread.start()
            threads.append(thread)
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        total_execution_time = time.time() - start_time
        # Collect results
        worker_results = []
        while not results_queue.empty():
            worker_results.append(results_queue.get())
        successful_workers = [r for r in worker_results if 'error' not in r]
        failed_workers = [r for r in worker_results if 'error' in r]
        return {
            'num_workers': num_workers,
            'successful_workers': len(successful_workers),
            'failed_workers': len(failed_workers),
            'total_execution_time': total_execution_time,
            'avg_worker_time': statistics.mean([r['execution_time'] for r in successful_workers]) if successful_workers else 0,
            'operations_per_second': sum(r.get('operations_completed', 0) for r in successful_workers) / total_execution_time if total_execution_time > 0 else 0
        }

def generate_performance_report(results: Dict) -> str:
    """Generate comprehensive performance test report"""
    report_lines = [
        "# Performance Test Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Load Test Results",
        ""
    ]
    if 'total_requests' in results:
        # Standard load test results
        report_lines.extend([
            f"**Duration**: {results.get('duration', 0):.2f} seconds",
            f"**Total Requests**: {results.get('total_requests', 0):,}",
            f"**Successful Requests**: {results.get('successful_requests', 0):,}",
            f"**Failed Requests**: {results.get('failed_requests', 0):,}",
            f"**Error Rate**: {results.get('error_rate', 0):.2f}%",
            f"**Throughput**: {results.get('throughput', 0):.2f} requests/second",
            "",
            "### Response Times",
            ""
        ])
        response_times = results.get('response_times', {})
        for metric, value in response_times.items():
            report_lines.append(f"**{metric.upper()}**: {value:.3f}s")
        report_lines.extend([
            "",
            "### Endpoint Performance",
            ""
        ])
        for endpoint, perf in results.get('endpoints_performance', {}).items():
            report_lines.extend([
                f"#### {endpoint}",
                f"- Requests: {perf.get('total_requests', 0):,}",
                f"- Error Rate: {perf.get('error_rate', 0):.2f}%",
                f"- Avg Response Time: {perf.get('avg_response_time', 0):.3f}s",
                f"- P95 Response Time: {perf.get('p95_response_time', 0):.3f}s",
                ""
            ])
    return "\n".join(report_lines)

def main():
    """Main entry point for performance testing"""
    import argparse
    parser = argparse.ArgumentParser(description='Performance Testing Suite')
    parser.add_argument('--test-type', choices=['load', 'stress', 'database'], default='load', help='Type of test to run')
    parser.add_argument('--base-url', default='http://localhost:5000', help='Base URL for API testing')
    parser.add_argument('--users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--ramp-up', type=int, default=10, help='Ramp-up time in seconds')
    parser.add_argument('--output', default='performance_results.json', help='Output file for results')
    args = parser.parse_args()

    if args.test_type == 'database':
        # Run database performance tests
        db_test = DatabasePerformanceTest()
        results = db_test.run_database_performance_tests()
        print("Database performance test completed")
    else:
        # Configure load test
        config = LoadTestConfig(
            base_url=args.base_url,
            websocket_url=args.base_url.replace('http', 'ws'),
            concurrent_users=args.users,
            test_duration=args.duration,
            ramp_up_time=args.ramp_up,
            endpoints=[
                {'path': '/api/v1/system/stats', 'method': 'GET'},
                {'path': '/api/v1/processes', 'method': 'GET'},
                {'path': '/api/v1/analytics/system/health', 'method': 'GET'},
                {'path': '/api/v1/dashboard/executive', 'method': 'GET'}
            ],
            user_scenarios=[]
        )
        if args.test_type == 'load':
            # Run load test
            runner = LoadTestRunner(config)
            results = runner.run_load_test()
        elif args.test_type == 'stress':
            # Run stress test
            runner = StressTestRunner(config)
            results = runner.run_stress_test()

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate report
    report = generate_performance_report(results)
    with open(args.output.replace('.json', '_report.md'), 'w') as f:
        f.write(report)

    print(f"Performance test completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
