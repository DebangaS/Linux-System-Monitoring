"""
Enhanced tests for process management module - Day 2
Author: Member 5
"""
import pytest
import sys
import os
import psutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modules.processes import (
    list_processes, filter_processes_by_name, filter_processes_by_user,
    filter_processes_by_cpu_usage, filter_processes_by_memory_usage,
    filter_processes_by_status, get_process_details, get_top_processes,
    kill_process, get_process_tree, monitor_process, format_process_table
)

class TestEnhancedProcesses:
    """Enhanced test cases for process management - Day 2"""

    def test_list_processes_basic(self):
        """Test basic process listing functionality"""
        processes = list_processes(limit=10)
        assert isinstance(processes, list)
        assert len(processes) <= 10
        if processes:
            process = processes[0]
            required_fields = ['pid', 'name', 'username', 'status', 'cpu_percent', 'memory_percent']
            for field in required_fields:
                assert field in process, f"Missing field: {field}"

    def test_list_processes_sorting(self):
        """Test process listing with different sorting options"""
        # Test sorting by CPU
        cpu_processes = list_processes(limit=5, sort_by='cpu_percent')
        assert isinstance(cpu_processes, list)
        if len(cpu_processes) >= 2:
            # Should be sorted by CPU percentage (descending)
            assert cpu_processes[0]['cpu_percent'] >= cpu_processes[1]['cpu_percent']
        # Test sorting by memory
        mem_processes = list_processes(limit=5, sort_by='memory_percent')
        assert isinstance(mem_processes, list)
        if len(mem_processes) >= 2:
            # Should be sorted by memory percentage (descending)
            assert mem_processes[0]['memory_percent'] >= mem_processes[1]['memory_percent']

    def test_filter_processes_by_name(self):
        """Test process filtering by name"""
        # Test with exact match
        python_processes = filter_processes_by_name('python', exact_match=True)
        assert isinstance(python_processes, list)
        for process in python_processes:
            assert process['name'] == 'python'
        # Test with substring match
        py_processes = filter_processes_by_name('py', exact_match=False)
        assert isinstance(py_processes, list)
        for process in py_processes:
            assert 'py' in process['name'].lower()

    def test_filter_processes_by_user(self):
        """Test process filtering by user"""
        # Get current user
        current_user = psutil.Process().username()
        user_processes = filter_processes_by_user(current_user)
        assert isinstance(user_processes, list)
        for process in user_processes:
            assert process['username'].lower() == current_user.lower()

    def test_filter_processes_by_cpu_usage(self):
        """Test process filtering by CPU usage"""
        high_cpu_processes = filter_processes_by_cpu_usage(min_cpu=0.1)
        assert isinstance(high_cpu_processes, list)
        for process in high_cpu_processes:
            assert process['cpu_percent'] >= 0.1

    def test_filter_processes_by_memory_usage(self):
        """Test process filtering by memory usage"""
        high_mem_processes = filter_processes_by_memory_usage(min_memory=0.1)
        assert isinstance(high_mem_processes, list)
        for process in high_mem_processes:
            assert process['memory_percent'] >= 0.1

    def test_filter_processes_by_status(self):
        """Test process filtering by status"""
        running_processes = filter_processes_by_status('running')
        assert isinstance(running_processes, list)
        for process in running_processes:
            assert process['status'].lower() == 'running'

    def test_get_process_details(self):
        """Test detailed process information"""
        # Get current process PID
        current_pid = os.getpid()
        details = get_process_details(current_pid)
        assert details is not None
        assert isinstance(details, dict)
        required_fields = ['pid', 'ppid', 'name', 'username', 'status']
        for field in required_fields:
            assert field in details, f"Missing field: {field}"
        assert details['pid'] == current_pid

    def test_get_process_details_invalid_pid(self):
        """Test process details with invalid PID"""
        # Use a PID that doesn't exist
        invalid_pid = 999999
        details = get_process_details(invalid_pid)
        assert details is None

    def test_get_top_processes(self):
        """Test getting top processes by different metrics"""
        # Test top by CPU
        top_cpu = get_top_processes(metric='cpu_percent', count=5)
        assert isinstance(top_cpu, list)
        assert len(top_cpu) <= 5
        if len(top_cpu) >= 2:
            assert top_cpu[0]['cpu_percent'] >= top_cpu[1]['cpu_percent']
        # Test top by memory
        top_memory = get_top_processes(metric='memory_percent', count=5)
        assert isinstance(top_memory, list)
        assert len(top_memory) <= 5
        if len(top_memory) >= 2:
            assert top_memory[0]['memory_percent'] >= top_memory[1]['memory_percent']

    def test_get_process_tree(self):
        """Test process tree functionality"""
        # Get current process
        current_pid = os.getpid()
        tree = get_process_tree(current_pid)
        assert isinstance(tree, dict)
        if tree:  # If tree is not empty
            assert 'pid' in tree
            assert 'name' in tree
            assert 'children' in tree
            assert tree['pid'] == current_pid
            assert isinstance(tree['children'], list)

    def test_monitor_process_short(self):
        """Test short-term process monitoring"""
        # Monitor current process for a short time
        current_pid = os.getpid()
        measurements = monitor_process(current_pid, duration=2, interval=1)
        assert isinstance(measurements, list)
        if measurements:
            measurement = measurements[0]
            required_fields = ['timestamp', 'pid', 'name', 'cpu_percent', 'memory_percent']
            for field in required_fields:
                assert field in measurement, f"Missing field: {field}"
            assert measurement['pid'] == current_pid

    def test_format_process_table(self):
        """Test process table formatting"""
        # Get a few processes
        processes = list_processes(limit=5)
        table = format_process_table(processes)
        assert isinstance(table, str)
        if processes:
            # Table should contain process information
            assert 'PID' in table
            assert 'Name' in table
            assert 'CPU%' in table
            assert 'Mem%' in table

    def test_format_process_table_empty(self):
        """Test process table formatting with empty list"""
        table = format_process_table([])
        assert isinstance(table, str)
        assert "No processes found" in table

    def test_kill_process_invalid(self):
        """Test killing invalid process (should fail safely)"""
        # Try to kill a non-existent process
        invalid_pid = 999999
        result = kill_process(invalid_pid, force=False)
        assert isinstance(result, bool)
        assert result is False  # Should fail for non-existent process

    def test_error_handling_processes(self):
        """Test error handling in process functions"""
        # Test with mock to simulate psutil errors
        with patch('modules.processes.psutil.process_iter') as mock_iter:
            mock_iter.side_effect = Exception("Test error")
            # Should handle error gracefully
            result = list_processes()
            assert isinstance(result, list)
            assert len(result) == 0

    def test_process_filtering_edge_cases(self):
        """Test edge cases in process filtering"""
        # Test filtering with non-existent user
        result = filter_processes_by_user("nonexistentuser123")
        assert isinstance(result, list)
        assert len(result) == 0
        # Test filtering with very high CPU threshold
        result = filter_processes_by_cpu_usage(min_cpu=200.0)
        assert isinstance(result, list)
        assert len(result) == 0
        # Test filtering with very high memory threshold
        result = filter_processes_by_memory_usage(min_memory=200.0)
        assert isinstance(result, list)
        assert len(result) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
