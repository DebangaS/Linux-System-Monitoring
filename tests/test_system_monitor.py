"""
Comprehensive tests for system monitoring functionality
Author: Member 5
"""
import unittest
import sys
import os
import time
from unittest.mock import patch, MagicMock

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.cpu_memory import get_cpu_usage, get_memory_usage, get_cpu_info, get_cpu_per_core
from src.modules.processes import list_processes, filter_processes_by_cpu_usage

class TestSystemMonitoring(unittest.TestCase):
    """Test cases for system monitoring functions"""
    
    def test_cpu_usage_returns_valid_percentage(self):
        """Test CPU usage returns valid percentage"""
        cpu_usage = get_cpu_usage(interval=0.1)
        self.assertIsInstance(cpu_usage, (int, float))
        self.assertGreaterEqual(cpu_usage, 0)
        self.assertLessEqual(cpu_usage, 100)
    
    def test_memory_usage_returns_complete_dict(self):
        """Test memory usage returns complete dictionary"""
        memory = get_memory_usage()
        self.assertIsInstance(memory, dict)
        
        required_keys = ['total', 'available', 'percent', 'used', 'free']
        for key in required_keys:
            self.assertIn(key, memory)
            self.assertIsInstance(memory[key], (int, float))
        
        # Test logical relationships
        self.assertGreater(memory['total'], 0)
        self.assertLessEqual(memory['used'], memory['total'])
        self.assertLessEqual(memory['percent'], 100)
    
    def test_cpu_per_core_returns_list(self):
        """Test per-core CPU usage returns list"""
        per_core = get_cpu_per_core()
        self.assertIsInstance(per_core, list)
        self.assertGreater(len(per_core), 0)
        
        for usage in per_core:
            self.assertIsInstance(usage, (int, float))
            self.assertGreaterEqual(usage, 0)
            self.assertLessEqual(usage, 100)
    
    def test_cpu_info_completeness(self):
        """Test CPU info contains all required fields"""
        cpu_info = get_cpu_info()
        self.assertIsInstance(cpu_info, dict)
        
        required_keys = ['physical_cores', 'total_cores', 'architecture']
        for key in required_keys:
            self.assertIn(key, cpu_info)
        
        self.assertGreater(cpu_info['physical_cores'], 0)
        self.assertGreater(cpu_info['total_cores'], 0)

class TestProcessManagement(unittest.TestCase):
    """Test cases for process management"""
    
    def test_list_processes_returns_valid_data(self):
        """Test process listing returns valid data"""
        processes = list_processes(limit=10)
        self.assertIsInstance(processes, list)
        self.assertLessEqual(len(processes), 10)
        
        if processes:
            process = processes[0]
            required_keys = ['pid', 'name', 'cpu_percent', 'memory_percent', 'status']
            for key in required_keys:
                self.assertIn(key, process)
    
    def test_filter_processes_by_cpu(self):
        """Test CPU-based process filtering"""
        high_cpu_processes = filter_processes_by_cpu_usage(0.1)
        self.assertIsInstance(high_cpu_processes, list)
        
        for process in high_cpu_processes:
            self.assertGreaterEqual(process.get('cpu_percent', 0), 0.1)

if __name__ == '__main__':
    unittest.main()
