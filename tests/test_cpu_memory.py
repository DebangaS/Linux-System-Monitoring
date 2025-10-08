"""
Tests for CPU/Memory monitoring module
Author: Member 5
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modules.cpu_memory import get_cpu_usage, get_memory_usage, get_cpu_per_core, get_cpu_info

class TestCPUMemory:
    """Test cases for CPU and Memory monitoring"""

    def test_get_cpu_usage_returns_float(self):
        """Test that CPU usage returns a float"""
        # TODO: Implement actual test - Day 1 placeholder
        cpu_usage = get_cpu_usage(interval=0.1)
        assert isinstance(cpu_usage, (int, float))
        assert 0 <= cpu_usage <= 100

    def test_get_memory_usage_returns_dict(self):
        """Test that memory usage returns a dictionary"""
        # TODO: Implement actual test - Day 1 placeholder
        memory_usage = get_memory_usage()
        assert isinstance(memory_usage, dict)
        required_keys = ['total', 'available', 'percent', 'used', 'free']
        for key in required_keys:
            assert key in memory_usage

    def test_get_cpu_per_core_returns_list(self):
        """Test that per-core CPU usage returns a list"""
        # TODO: Implement actual test - Day 1 placeholder
        cpu_per_core = get_cpu_per_core()
        assert isinstance(cpu_per_core, list)
        assert len(cpu_per_core) > 0

    def test_get_cpu_info_returns_dict(self):
        """Test that CPU info returns a dictionary"""
        # TODO: Implement actual test - Day 1 placeholder
        cpu_info = get_cpu_info()
        assert isinstance(cpu_info, dict)
        required_keys = ['physical_cores', 'total_cores']
        for key in required_keys:
            assert key in cpu_info

if __name__ == "__main__":
    pytest.main([__file__])
