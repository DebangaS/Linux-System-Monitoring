"""
Performance tests for Day 2 implementations
Author: Member 5
"""
import pytest
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modules.cpu_memory import get_cpu_usage, get_memory_usage, get_disk_usage
from modules.processes import list_processes, get_top_processes, filter_processes_by_cpu_usage


class TestDay2Performance:
    """Performance tests for Day 2 enhanced functionality"""

    def test_cpu_monitoring_performance(self):
        """Test CPU monitoring performance"""
        # Test single call performance
        start_time = time.time()
        cpu_usage = get_cpu_usage(interval=0.1)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"CPU monitoring too slow: {execution_time:.3f}s"
        assert isinstance(cpu_usage, (int, float))
        assert 0 <= cpu_usage <= 100

    def test_memory_monitoring_performance(self):
        """Test memory monitoring performance"""
        # Test single call performance
        start_time = time.time()
        memory_usage = get_memory_usage()
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 0.5, f"Memory monitoring too slow: {execution_time:.3f}s"
        assert isinstance(memory_usage, dict)

    def test_process_listing_performance(self):
        """Test process listing performance"""
        # Test with different limits
        limits_to_test = [10, 50, 100, 200]
        for limit in limits_to_test:
            start_time = time.time()
            processes = list_processes(limit=limit)
            end_time = time.time()
            execution_time = end_time - start_time
            # Performance should scale reasonably
            max_time = 0.5 + (limit / 1000)  # Base time + scaling factor
            assert execution_time < max_time, f"Process listing too slow for limit {limit}: {execution_time:.3f}s"
            assert len(processes) <= limit

    def test_process_filtering_performance(self):
        """Test process filtering performance"""
        start_time = time.time()
        high_cpu_processes = filter_processes_by_cpu_usage(min_cpu=0.1)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 2.0, f"Process filtering too slow: {execution_time:.3f}s"
        assert isinstance(high_cpu_processes, list)

    def test_top_processes_performance(self):
        """Test top processes performance"""
        counts_to_test = [5, 10, 20, 50]
        for count in counts_to_test:
            start_time = time.time()
            top_processes = get_top_processes(metric='cpu_percent', count=count)
            end_time = time.time()
            execution_time = end_time - start_time
            max_time = 1.0 + (count / 100)  # Base time + scaling factor
            assert execution_time < max_time, f"Top processes too slow for count {count}: {execution_time:.3f}s"
            assert len(top_processes) <= count

    def test_disk_usage_performance(self):
        """Test disk usage performance"""
        start_time = time.time()
        disk_usage = get_disk_usage()
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 2.0, f"Disk usage monitoring too slow: {execution_time:.3f}s"
        assert isinstance(disk_usage, list)

    def test_concurrent_monitoring_performance(self):
        """Test performance under concurrent access"""

        def monitor_task():
            results = []
            # CPU monitoring
            start = time.time()
            cpu = get_cpu_usage(interval=0.1)
            results.append(('cpu', time.time() - start))
            # Memory monitoring
            start = time.time()
            memory = get_memory_usage()
            results.append(('memory', time.time() - start))
            # Process listing
            start = time.time()
            processes = list_processes(limit=20)
            results.append(('processes', time.time() - start))
            return results

        # Run concurrent tasks
        num_threads = 5
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.time()
            futures = [executor.submit(monitor_task) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
            total_time = time.time() - start_time

        # Analyze results
        assert len(results) == num_threads
        assert total_time < 5.0, f"Concurrent monitoring too slow: {total_time:.3f}s"

        # Check individual operation times
        for task_results in results:
            for operation, op_time in task_results:
                if operation == 'cpu':
                    assert op_time < 1.0, f"Concurrent CPU monitoring too slow: {op_time:.3f}s"
                elif operation == 'memory':
                    assert op_time < 0.5, f"Concurrent memory monitoring too slow: {op_time:.3f}s"
                elif operation == 'processes':
                    assert op_time < 2.0, f"Concurrent process listing too slow: {op_time:.3f}s"

    def test_repeated_calls_performance(self):
        """Test performance of repeated calls (simulating monitoring loop)"""
        num_iterations = 20
        start_time = time.time()

        for i in range(num_iterations):
            cpu_usage = get_cpu_usage(interval=0.1)
            memory_usage = get_memory_usage()
            # Only get processes every 5th iteration (more realistic)
            if i % 5 == 0:
                processes = list_processes(limit=10)

        total_time = time.time() - start_time
        avg_time_per_iteration = total_time / num_iterations
        assert total_time < 15.0, f"Repeated calls too slow total: {total_time:.3f}s"
        assert avg_time_per_iteration < 1.0, f"Average iteration too slow: {avg_time_per_iteration:.3f}s"

    def test_memory_efficiency(self):
        """Test memory efficiency of monitoring operations"""
        import psutil
        import gc

        process = psutil.Process()
        # Force garbage collection and get initial memory
        gc.collect()
        initial_memory = process.memory_info().rss

        # Perform monitoring operations
        for i in range(50):
            cpu_usage = get_cpu_usage(interval=0.1)
            memory_usage = get_memory_usage()
            processes = list_processes(limit=10)
            # Collect garbage every 10 iterations
            if i % 10 == 0:
                gc.collect()

        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / (1024 * 1024)
        assert memory_growth_mb < 50, f"Excessive memory growth: {memory_growth_mb:.1f}MB"

    def test_cpu_usage_accuracy_vs_performance(self):
        """Test CPU usage accuracy vs performance trade-off"""
        intervals = [0.1, 0.2, 0.5, 1.0]
        for interval in intervals:
            measurements = []
            times = []
            # Take multiple measurements
            for _ in range(5):
                start_time = time.time()
                cpu_usage = get_cpu_usage(interval=interval)
                end_time = time.time()
                measurements.append(cpu_usage)
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            # Verify measurements are reasonable
            for measurement in measurements:
                assert 0 <= measurement <= 100
            # Verify timing expectations
            expected_min_time = interval
            expected_max_time = interval + 0.5  # Allow some overhead
            assert avg_time >= expected_min_time, f"CPU measurement too fast for interval {interval}: {avg_time:.3f}s"
            assert avg_time <= expected_max_time, f"CPU measurement too slow for interval {interval}: {avg_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
