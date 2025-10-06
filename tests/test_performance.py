"""
Performance tests for system monitoring
Author: Member 5
"""
import unittest
import time
from concurrent.futures import ThreadPoolExecutor
from src.modules.cpu_memory import get_cpu_usage, get_memory_usage
from src.modules.processes import list_processes

class TestPerformance(unittest.TestCase):
    """Performance and load tests"""
    
    def test_monitoring_function_performance(self):
        """Test monitoring functions complete within reasonable time"""
        start_time = time.time()
        
        # Test CPU monitoring performance
        cpu_usage = get_cpu_usage(interval=0.1)
        cpu_time = time.time() - start_time
        self.assertLess(cpu_time, 1.0)  # Should complete within 1 second
        
        # Test memory monitoring performance
        start_time = time.time()
        memory_usage = get_memory_usage()
        memory_time = time.time() - start_time
        self.assertLess(memory_time, 0.5)  # Should complete within 0.5 seconds
        
        # Test process listing performance
        start_time = time.time()
        processes = list_processes(limit=50)
        process_time = time.time() - start_time
        self.assertLess(process_time, 2.0)  # Should complete within 2 seconds
    
    def test_concurrent_monitoring_calls(self):
        """Test system handles concurrent monitoring requests"""
        def monitor_task():
            return {
                'cpu': get_cpu_usage(interval=0.1),
                'memory': get_memory_usage(),
                'processes': len(list_processes(limit=20))
            }
        
        # Run 5 concurrent monitoring tasks
        with ThreadPoolExecutor(max_workers=5) as executor:
            start_time = time.time()
            futures = [executor.submit(monitor_task) for _ in range(5)]
            results = [future.result() for future in futures]
            execution_time = time.time() - start_time
        
        # All tasks should complete
        self.assertEqual(len(results), 5)
        # Should complete within reasonable time
        self.assertLess(execution_time, 5.0)
        
        # All results should be valid
        for result in results:
            self.assertIn('cpu', result)
            self.assertIn('memory', result)
            self.assertIn('processes', result)

if __name__ == '__main__':
    unittest.main()
