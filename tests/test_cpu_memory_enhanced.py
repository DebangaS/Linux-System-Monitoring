"""
Enhanced tests for CPU/Memory monitoring module - Day 2
Author: Member 5
"""
import pytest
import sys
import os
System Monitor Project - Day 2
Member 5 (Documentation & Testing Lead) - Complete Implementation
🎯 Day 2 Objectives for Member 5
Create comprehensive tests for all Day 2 enhanced modules
Update documentation to reflect new functionality
Implement integration testing between modules
Set up performance testing framework
Create quality assurance validation scripts
Update README with new features and usage examples
🛠 Step-by-Step Implementation
Tasks:
1. Write comprehensive tests for enhanced CPU/Memory monitoring
2. Create tests for advanced process management features
3. Test integration between all enhanced modules
4. Update documentation with Day 2 functionality
5. Create performance testing framework
6. Set up automated quality checks
Files to Update/Create:
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from modules.cpu_memory import (
get_cpu_usage, get_memory_usage, get_cpu_per_core, get_cpu_info,
get_cpu_times, get_memory_info_detailed, get_disk_usage,
monitor_cpu_continuously, format_bytes
)
class TestEnhancedCPUMemory:
"""Enhanced test cases for CPU and Memory monitoring - Day 2"""
def test_get_cpu_usage_with_interval(self):
"""Test CPU usage with different intervals"""
# Test with short interval for faster testing
cpu_usage_short = get_cpu_usage(interval=0.1)
assert isinstance(cpu_usage_short, (int, float))
assert 0 &lt;= cpu_usage_short &lt;= 100
# Test with slightly longer interval
cpu_usage_long = get_cpu_usage(interval=0.5)
assert isinstance(cpu_usage_long, (int, float))
assert 0 &lt;= cpu_usage_long &lt;= 100
def test_get_memory_usage_comprehensive(self):
"""Test comprehensive memory usage statistics"""
memory = get_memory_usage()
assert isinstance(memory, dict)
# Test all required fields are present
required_fields = [
'total', 'available', 'percent', 'used', 'free',
'swap_total', 'swap_used', 'swap_free', 'swap_percent'
]
for field in required_fields:
assert field in memory, f"Missing field: {field}"
assert isinstance(memory[field], (int, float))
# Test logical relationships
assert memory['total'] &gt; 0
assert memory['used'] + memory['available'] &gt;= memory['total'] * 0.95 # Allow fo
assert 0 &lt;= memory['percent'] &lt;= 100
assert 0 &lt;= memory['swap_percent'] &lt;= 100
def test_get_cpu_per_core(self):
"""Test per-core CPU usage"""
per_core_usage = get_cpu_per_core()
assert isinstance(per_core_usage, list)
assert len(per_core_usage) &gt; 0
for usage in per_core_usage:
assert isinstance(usage, (int, float))
assert 0 &lt;= usage &lt;= 100
def test_get_cpu_info_detailed(self):
"""Test detailed CPU information"""
cpu_info = get_cpu_info()
assert isinstance(cpu_info, dict)
required_fields = [
'physical_cores', 'total_cores', 'architecture', 'machine'
]
for field in required_fields:
assert field in cpu_info, f"Missing field: {field}"
# Test logical relationships
assert cpu_info['physical_cores'] &gt; 0
assert cpu_info['total_cores'] &gt;= cpu_info['physical_cores']
# Test CPU stats if available
if 'cpu_stats' in cpu_info:
assert isinstance(cpu_info['cpu_stats'], dict)
def test_get_cpu_times(self):
"""Test CPU times breakdown"""
cpu_times = get_cpu_times()
assert isinstance(cpu_times, dict)
# Test that we have at least basic time measurements
basic_times = ['user', 'system', 'idle']
for time_type in basic_times:
if time_type in cpu_times:
assert isinstance(cpu_times[time_type], (int, float))
assert cpu_times[time_type] &gt;= 0
def test_get_memory_info_detailed(self):
"""Test detailed memory information including top processes"""
detailed_memory = get_memory_info_detailed()
assert isinstance(detailed_memory, dict)
# Should contain all basic memory info
assert 'percent' in detailed_memory
assert 'total' in detailed_memory
# Should contain top processes
if 'top_processes' in detailed_memory:
assert isinstance(detailed_memory['top_processes'], list)
if detailed_memory['top_processes']:
process = detailed_memory['top_processes'][0]
assert 'pid' in process
assert 'name' in process
assert 'memory_percent' in process
def test_get_disk_usage(self):
"""Test disk usage information"""
disk_usage = get_disk_usage()
assert isinstance(disk_usage, list)
if disk_usage: # If there are disks
disk = disk_usage[0]
required_fields = ['device', 'mountpoint', 'fstype', 'total', 'used', 'free', 'p
for field in required_fields:
assert field in disk, f"Missing field: {field}"
# Test logical relationships
assert disk['total'] &gt; 0
assert disk['used'] &gt;= 0
assert disk['free'] &gt;= 0
assert 0 &lt;= disk['percent'] &lt;= 100
assert disk['used'] + disk['free'] &lt;= disk['total'] * 1.01 # Allow small ove
def test_monitor_cpu_continuously(self):
"""Test continuous CPU monitoring"""
# Test short monitoring period
measurements = monitor_cpu_continuously(duration=3, interval=1)
assert isinstance(measurements, list)
if measurements: # If monitoring completed successfully
assert len(measurements) &gt;= 2 # Should have at least 2 measurements in 3 sec
for measurement in measurements:
assert isinstance(measurement, dict)
assert 'timestamp' in measurement
assert 'cpu_percent' in measurement
assert 'cpu_per_core' in measurement
assert 'cpu_times' in measurement
assert isinstance(measurement['timestamp'], float)
assert isinstance(measurement['cpu_percent'], (int, float))
assert isinstance(measurement['cpu_per_core'], list)
assert isinstance(measurement['cpu_times'], dict)
def test_format_bytes(self):
"""Test byte formatting utility"""
# Test different byte sizes
test_cases = [
(0, "0.0 B"),
(500, "500.0 B"),
(1024, "1.0 KB"),
(1024 * 1024, "1.0 MB"),
(1024 * 1024 * 1024, "1.0 GB"),
(1024 * 1024 * 1024 * 1024, "1.0 TB"),
]
for bytes_value, expected_format in test_cases:
result = format_bytes(bytes_value)
assert isinstance(result, str)
# Check that the result contains the expected unit
assert expected_format.split()[-1] in result
def test_error_handling(self):
"""Test error handling in CPU/Memory functions"""
# Test with mock to simulate psutil errors
with patch('modules.cpu_memory.psutil') as mock_psutil:
mock_psutil.cpu_percent.side_effect = Exception("Test error")
# Should handle error gracefully
result = get_cpu_usage()
assert result == 0.0
with patch('modules.cpu_memory.psutil') as mock_psutil:
mock_psutil.virtual_memory.side_effect = Exception("Test error")
# Should handle error gracefully
result = get_memory_usage()
assert isinstance(result, dict)
assert len(result) == 0 # Empty dict on error
def test_performance_requirements(self):
"""Test that functions meet performance requirements"""
# Test CPU usage function performance
start_time = time.time()
cpu_usage = get_cpu_usage(interval=0.1)
end_time = time.time()
# Should complete within reasonable time
assert end_time - start_time &lt; 2.0 # Less than 2 seconds
# Test memory usage function performance
start_time = time.time()
memory_usage = get_memory_usage()
end_time = time.time()
# Should complete quickly
assert end_time - start_time &lt; 1.0 # Less than 1 second
if __name__ == "__main__":
pytest.main([__file__, "-v"])
