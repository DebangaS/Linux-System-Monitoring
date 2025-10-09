"""
Performance & Load Tests for System Monitor
Author: Member 5
"""
import pytest
import time
from modules.cpu_memory import get_cpu_usage

def test_cpu_stress():
    times = []
    for _ in range(20):
        start = time.time()
        val = get_cpu_usage()
        end = time.time()
        times.append(end - start)
        assert 0 <= val <= 100
    avg_time = sum(times)/len(times)
    assert avg_time < 0.5  # Fast enough for real-time
