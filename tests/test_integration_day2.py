"""
Integration tests for Day 2 implementations
Author: Member 5
"""
import pytest
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modules.cpu_memory import get_cpu_usage, get_memory_usage, get_disk_usage
from modules.processes import list_processes, get_top_processes
from modules.logger import setup_logger, log_system_stats, log_process_info, log_alert
from modules.exporter import export_to_csv, export_to_json, create_directories
from modules.scheduler import start_scheduler


class TestDay2Integration:
    """Integration tests for all Day 2 enhanced modules"""

    def test_system_monitoring_integration(self):
        """Test integration between CPU/Memory and Process monitoring"""
        # Get system stats
        cpu_usage = get_cpu_usage(interval=0.1)
        memory_usage = get_memory_usage()
        disk_usage = get_disk_usage()

        # Get process information
        processes = list_processes(limit=10)
        top_cpu_processes = get_top_processes(metric='cpu_percent', count=5)

        # Verify all data is collected
        assert isinstance(cpu_usage, (int, float))
        assert isinstance(memory_usage, dict)
        assert isinstance(disk_usage, list)
        assert isinstance(processes, list)
        assert isinstance(top_cpu_processes, list)

        # Test data consistency
        if top_cpu_processes:
            # Top CPU processes should be sorted
            for i in range(len(top_cpu_processes) - 1):
                assert top_cpu_processes[i]['cpu_percent'] >= top_cpu_processes[i + 1]['cpu_percent']

    def test_logging_integration(self):
        """Test integration with logging system"""
        # Setup logger
        logger = setup_logger(log_level="DEBUG", log_file="data/logs/test_integration.log")

        # Get system data
        cpu_usage = get_cpu_usage(interval=0.1)
        memory_usage = get_memory_usage()
        processes = list_processes(limit=5)

        # Test logging integration
        log_system_stats(cpu_usage, memory_usage)
        log_process_info(processes)
        log_alert("Integration test alert", "INFO", {"test": True})

        # Verify log file exists
        log_file = Path("data/logs/test_integration.log")
        assert log_file.exists()
        assert log_file.stat().st_size > 0

    def test_export_integration(self):
        """Test integration with export functionality"""
        # Create necessary directories
        create_directories()

        # Get system data
        cpu_usage = get_cpu_usage(interval=0.1)
        memory_usage = get_memory_usage()
        processes = list_processes(limit=10)

        # Test CSV export integration
        process_data = []
        for proc in processes[:5]:  # Export top 5 processes
            process_data.append({
                'pid': proc['pid'],
                'name': proc['name'],
                'cpu_percent': proc['cpu_percent'],
                'memory_percent': proc['memory_percent'],
                'status': proc['status']
            })
        export_to_csv(process_data, "test_integration_processes")

        # Test JSON export integration
        system_data = {
            'timestamp': time.time(),
            'cpu_usage': cpu_usage,
            'memory_usage': {
                'percent': memory_usage.get('percent', 0),
                'total_gb': memory_usage.get('total', 0) / (1024 ** 3),
                'available_gb': memory_usage.get('available', 0) / (1024 ** 3)
            },
            'process_count': len(processes)
        }
        export_to_json(system_data, "test_integration_system")

        # Verify export files exist
        export_dir = Path("data/exports")
        csv_files = list(export_dir.glob("test_integration_processes_*.csv"))
        json_files = list(export_dir.glob("test_integration_system_*.json"))
        assert len(csv_files) > 0
        assert len(json_files) > 0

    def test_scheduler_integration(self):
        """Test basic scheduler functionality"""
        # Test scheduler startup
        scheduler = start_scheduler()
        assert scheduler is not None

        # Give scheduler a moment to start
        time.sleep(1)

        # Test that scheduler is running
        assert scheduler.running

        # Shutdown scheduler
        scheduler.shutdown()

    def test_data_consistency(self):
        """Test data consistency across modules"""
        # Get data from different modules
        processes_list = list_processes(limit=20)
        top_cpu_processes = get_top_processes(metric='cpu_percent', count=10)
        top_memory_processes = get_top_processes(metric='memory_percent', count=10)

        # Create sets of PIDs for comparison
        all_pids = {proc['pid'] for proc in processes_list}
        top_cpu_pids = {proc['pid'] for proc in top_cpu_processes}
        top_memory_pids = {proc['pid'] for proc in top_memory_processes}

        # Top processes should be subset of all processes
        assert top_cpu_pids.issubset(all_pids)
        assert top_memory_pids.issubset(all_pids)

        # Verify CPU sorting
        if len(top_cpu_processes) >= 2:
            for i in range(len(top_cpu_processes) - 1):
                assert top_cpu_processes[i]['cpu_percent'] >= top_cpu_processes[i + 1]['cpu_percent']

        # Verify memory sorting
        if len(top_memory_processes) >= 2:
            for i in range(len(top_memory_processes) - 1):
                assert top_memory_processes[i]['memory_percent'] >= top_memory_processes[i + 1]['memory_percent']

    def test_error_handling_integration(self):
        """Test error handling across integrated modules"""
        # Test with limited permissions or simulated errors
        try:
            # Get system data
            cpu_usage = get_cpu_usage(interval=0.1)
            memory_usage = get_memory_usage()
            processes = list_processes(limit=5)

            # Log the data (should handle any logging errors gracefully)
            logger = setup_logger(log_file="data/logs/test_error_handling.log")
            log_system_stats(cpu_usage, memory_usage)
            log_process_info(processes)

            # Export the data (should handle export errors gracefully)
            if processes:
                export_to_json({
                    'cpu': cpu_usage,
                    'memory_percent': memory_usage.get('percent', 0),
                    'process_count': len(processes)
                }, "test_error_handling")

        except Exception as e:
            pytest.fail(f"Integration test failed with error: {e}")

    def test_performance_integration(self):
        """Test performance of integrated operations"""
        start_time = time.time()

        # Perform integrated operations
        cpu_usage = get_cpu_usage(interval=0.1)
        memory_usage = get_memory_usage()
        processes = list_processes(limit=50)
        top_processes = get_top_processes(metric='cpu_percent', count=10)

        # Setup logging and log data
        logger = setup_logger(log_file="data/logs/test_performance.log")
        log_system_stats(cpu_usage, memory_usage)
        log_process_info(processes)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete within reasonable time
        assert total_time < 10.0, f"Integration operations took too long: {total_time:.2f}s"

    def test_memory_usage_integration(self):
        """Test memory usage of integrated operations"""
        import psutil
        import gc

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Perform multiple integrated operations
        for i in range(10):
            cpu_usage = get_cpu_usage(interval=0.1)
            memory_usage = get_memory_usage()
            processes = list_processes(limit=100)

            # Force garbage collection
            gc.collect()

        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth < 100 * 1024 * 1024, f"Excessive memory growth: {memory_growth}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
