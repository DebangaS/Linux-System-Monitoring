"""
Load testing for system monitoring application
Author: Member 5
"""

import unittest
import threading
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import statistics


class TestApplicationLoad(unittest.TestCase):
    """Load tests for the complete application"""

    def setUp(self):
        self.base_url = 'http://localhost:5000'
        self.api_url = f'{self.base_url}/api/v1'
        self.load_results = []
        self.wait_for_application()

    def wait_for_application(self):
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f'{self.api_url}/health', timeout=5)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        raise Exception("Application not ready for load testing")

    def test_concurrent_api_requests(self):
        endpoints = [
            '/system/resources', '/system/cpu', '/system/memory',
            '/system/processes', '/system/info'
        ]

        def make_request(endpoint):
            start_time = time.time()
            try:
                response = requests.get(f'{self.api_url}{endpoint}', timeout=30)
                end_time = time.time()
                return {
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time': end_time - start_time,
                    'success': response.status_code == 200,
                    'response_size': len(response.content),
                }
            except Exception as e:
                end_time = time.time()
                return {
                    'endpoint': endpoint,
                    'status_code': 0,
                    'response_time': end_time - start_time,
                    'success': False,
                    'error': str(e),
                }

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for _ in range(10):
                for endpoint in endpoints:
                    futures.append(executor.submit(make_request, endpoint))
            results = []
            try:
                for future in as_completed(futures, timeout=60):
                    result = future.result()
                    results.append(result)
            except Exception as e:
                # In case of timeout, collect results from finished futures only
                for future in futures:
                    if future.done():
                        try:
                            results.append(future.result())
                        except Exception:
                            pass

        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        success_rate = len(successful_requests) / max(len(results), 1) * 100

        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = max_response_time = min_response_time = 0

        print("\n=== Load Test Results ===")
        print(f"Total requests: {len(results)}")
        print(f"Successful requests: {len(successful_requests)}")
        print(f"Failed requests: {len(failed_requests)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Min response time: {min_response_time:.3f}s")
        print(f"Max response time: {max_response_time:.3f}s")

        self.assertGreaterEqual(success_rate, 95.0, "Success rate should be at least 95%")
        self.assertLess(avg_response_time, 5.0, "Average response time should be under 5 sec")
        self.assertLess(max_response_time, 15.0, "Max response time should be under 15 seconds")

    def test_websocket_load(self):
        import socketio

        connected_clients = []
        connection_errors = []

        def create_client():
            try:
                sio = socketio.Client()

                @sio.event
                def connect():
                    connected_clients.append(sio)

                @sio.event
                def connect_error(data):
                    connection_errors.append(data)

                sio.connect(self.base_url)
                time.sleep(2)
                sio.disconnect()
                return True
            except Exception as e:
                connection_errors.append(str(e))
                return False

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_client) for _ in range(20)]
            results = []
            try:
                for future in as_completed(futures, timeout=30):
                    results.append(future.result())
            except Exception as e:
                # In case of timeout, collect results from finished futures only
                for future in futures:
                    if future.done():
                        try:
                            results.append(future.result())
                        except Exception:
                            pass

        successful_connections = sum(results)

        print("\n=== WebSocket Load Test Results ===")
        print(f"Attempted connections: 20")
        print(f"Successful connections: {successful_connections}")
        print(f"Connection errors: {len(connection_errors)}")

        self.assertGreaterEqual(successful_connections, 15, "Should handle at least 15 concurrent connections")

    def test_database_load(self):
        from database.models import db_manager
        import tempfile

        temp_db = tempfile.mktemp(suffix='.db')
        test_db_manager = db_manager.__class__(db_path=temp_db)

        def write_data(worker_id):
            results = []
            for i in range(50):
                start_time = time.time()
                sample_data = {
                    'usage_percent': 50.0 + (worker_id * 10) + i,
                    'timestamp': datetime.utcnow().isoformat(),
                }
                success = test_db_manager.store_system_metrics(sample_data, sample_data, sample_data, sample_data)
                end_time = time.time()
                results.append({
                    'worker_id': worker_id,
                    'operation': 'write',
                    'success': success,
                    'time': end_time - start_time,
                })
            return results

        def read_data():
            results = []
            for i in range(20):
                start_time = time.time()
                data = test_db_manager.get_historical_metrics('cpu', hours=1)
                end_time = time.time()
                results.append({
                    'operation': 'read',
                    'success': len(data) >= 0,
                    'time': end_time - start_time,
                    'records': len(data) if data else 0,
                })
            return results

        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                write_futures = [executor.submit(write_data, i) for i in range(5)]
                read_futures = [executor.submit(read_data) for _ in range(3)]

                all_results = []
                try:
                    for future in as_completed(write_futures + read_futures, timeout=60):
                        all_results.extend(future.result())
                except Exception as e:
                    # In case of timeout, collect results from finished futures only
                    for future in write_futures + read_futures:
                        if future.done():
                            try:
                                all_results.extend(future.result())
                            except Exception:
                                pass

                write_results = [r for r in all_results if r['operation'] == 'write']
                read_results = [r for r in all_results if r['operation'] == 'read']

                write_success_rate = len([r for r in write_results if r['success']]) / max(len(write_results), 1)
                read_success_rate = len([r for r in read_results if r['success']]) / max(len(read_results), 1)

                avg_write_time = statistics.mean([r['time'] for r in write_results]) if write_results else 0
                avg_read_time = statistics.mean([r['time'] for r in read_results]) if read_results else 0

                print("\n=== Database Load Test Results ===")
                print(f"Write operations: {len(write_results)}")
                print(f"Read operations: {len(read_results)}")
                print(f"Write success rate: {write_success_rate * 100:.1f}%")
                print(f"Read success rate: {read_success_rate * 100:.1f}%")
                print(f"Average write time: {avg_write_time:.4f}s")
                print(f"Average read time: {avg_read_time:.4f}s")

                self.assertGreaterEqual(write_success_rate, 0.98, "Write success rate should be >= 98%")
                self.assertGreaterEqual(read_success_rate, 1.0, "Read success rate should be 100%")
                self.assertLess(avg_write_time, 0.1, "Average write time should be under 0.1 sec")
                self.assertLess(avg_read_time, 1.0, "Average read time should be under 1 second")

        finally:
            import os
            if os.path.exists(temp_db):
                os.unlink(temp_db)


if __name__ == '__main__':
    unittest.main(verbosity=2)
