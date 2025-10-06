"""
CPU and Memory Monitoring Module - Day 2 Implementation
Author: Member 2
"""
import psutil
import time
import platform
from typing import Dict, List, Optional
def get_cpu_usage(interval: float = 1.0) -&gt; float:
"""
Get current CPU usage percentage with specified interval
Args:
interval: Time interval for CPU measurement
Returns:
CPU usage percentage
"""
try:
return psutil.cpu_percent(interval=interval)
except Exception as e:
print(f"Error getting CPU usage: {e}")
return 0.0
def get_memory_usage() -&gt; Dict[str, float]:
"""
Get comprehensive memory usage statistics
Returns:
Dictionary containing memory statistics
"""
try:
memory = psutil.virtual_memory()
swap = psutil.swap_memory()
return {
# Virtual Memory
'total': memory.total,
'available': memory.available,
'percent': memory.percent,
'used': memory.used,
'free': memory.free,
'active': getattr(memory, 'active', 0),
'inactive': getattr(memory, 'inactive', 0),
'buffers': getattr(memory, 'buffers', 0),
'cached': getattr(memory, 'cached', 0),
'shared': getattr(memory, 'shared', 0),
# Swap Memory
'swap_total': swap.total,
'swap_used': swap.used,
'swap_free': swap.free,
'swap_percent': swap.percent,

'swap_sin': getattr(swap, 'sin', 0),
'swap_sout': getattr(swap, 'sout', 0)
}
except Exception as e:
print(f"Error getting memory usage: {e}")
return {}
def get_cpu_per_core() -&gt; List[float]:
"""
Get CPU usage per core
Returns:
List of CPU usage percentages for each core
"""
try:
return psutil.cpu_percent(percpu=True, interval=1)
except Exception as e:
print(f"Error getting per-core CPU usage: {e}")
return []
def get_cpu_info() -&gt; Dict[str, any]:
"""
Get comprehensive CPU information
Returns:
Dictionary containing CPU information
"""
try:
freq = psutil.cpu_freq()
cpu_stats = psutil.cpu_stats()
return {
'physical_cores': psutil.cpu_count(logical=False),
'total_cores': psutil.cpu_count(logical=True),
'max_frequency': freq.max if freq else 'N/A',
'min_frequency': freq.min if freq else 'N/A',
'current_frequency': freq.current if freq else 'N/A',
'cpu_stats': {
'ctx_switches': cpu_stats.ctx_switches,
'interrupts': cpu_stats.interrupts,
'soft_interrupts': cpu_stats.soft_interrupts,
'syscalls': getattr(cpu_stats, 'syscalls', 0)
},
'architecture': platform.architecture()[0],
'machine': platform.machine(),
'processor': platform.processor()
}
except Exception as e:
print(f"Error getting CPU info: {e}")
return {}
def get_cpu_times() -&gt; Dict[str, float]:
"""
Get CPU times breakdown
Returns:
Dictionary containing CPU times for different activities
"""
try:

cpu_times = psutil.cpu_times()
return {
'user': cpu_times.user,
'system': cpu_times.system,
'idle': cpu_times.idle,
'nice': getattr(cpu_times, 'nice', 0),
'iowait': getattr(cpu_times, 'iowait', 0),
'irq': getattr(cpu_times, 'irq', 0),
'softirq': getattr(cpu_times, 'softirq', 0),
'steal': getattr(cpu_times, 'steal', 0),
'guest': getattr(cpu_times, 'guest', 0),
'guest_nice': getattr(cpu_times, 'guest_nice', 0)
}
except Exception as e:
print(f"Error getting CPU times: {e}")
return {}
def get_memory_info_detailed() -&gt; Dict[str, any]:
"""
Get detailed memory information including per-process breakdown
Returns:
Dictionary containing detailed memory information
"""
try:
# Get top memory consuming processes
processes = []
for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'memory_info']):
try:
processes.append({
'pid': proc.info['pid'],
'name': proc.info['name'],
'memory_percent': proc.info['memory_percent'],
'memory_rss': proc.info['memory_info'].rss if proc.info['memory_info'] e
'memory_vms': proc.info['memory_info'].vms if proc.info['memory_info'] e
})
except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
pass
# Sort by memory usage and get top 10
top_processes = sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:
memory_usage = get_memory_usage()
memory_usage['top_processes'] = top_processes
return memory_usage
except Exception as e:
print(f"Error getting detailed memory info: {e}")
return get_memory_usage()
def get_disk_usage() -&gt; List[Dict[str, any]]:
"""
Get disk usage information for all mounted disks
Returns:
List of disk usage information
"""
try:

disk_usage = []
for partition in psutil.disk_partitions():
try:
usage = psutil.disk_usage(partition.mountpoint)
disk_usage.append({
'device': partition.device,
'mountpoint': partition.mountpoint,
'fstype': partition.fstype,
'total': usage.total,
'used': usage.used,
'free': usage.free,
'percent': (usage.used / usage.total) * 100
})
except PermissionError:
# This can happen on Windows for system partitions
continue
return disk_usage
except Exception as e:
print(f"Error getting disk usage: {e}")
return []
def monitor_cpu_continuously(duration: int = 60, interval: int = 5) -&gt; List[Dict[str, any
"""
Monitor CPU usage continuously for specified duration
Args:
duration: Total monitoring duration in seconds
interval: Measurement interval in seconds
Returns:
List of CPU measurements with timestamps
"""
measurements = []
start_time = time.time()
try:
while time.time() - start_time &lt; duration:
measurement = {
'timestamp': time.time(),
'cpu_percent': get_cpu_usage(interval=0.1),
'cpu_per_core': get_cpu_per_core(),
'cpu_times': get_cpu_times()
}
measurements.append(measurement)
time.sleep(interval)
except KeyboardInterrupt:
print("Monitoring stopped by user")
return measurements
def format_bytes(bytes_value: int) -&gt; str:
"""
Format bytes to human readable format
Args:
bytes_value: Size in bytes
Returns:
Formatted string with appropriate unit
"""

for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
if bytes_value &lt; 1024.0:
return f"{bytes_value:.1f} {unit}"
bytes_value /= 1024.0
return f"{bytes_value:.1f} PB"
if __name__ == "__main__":
print("Testing Enhanced CPU/Memory module - Day 2")
print("=" * 50)
# Test CPU monitoring
print("CPU Usage:", f"{get_cpu_usage()}%")
print("CPU Info:", get_cpu_info())
print("CPU per core:", get_cpu_per_core())
# Test memory monitoring
memory = get_memory_usage()
print(f"Memory Usage: {memory['percent']:.1f}%")
print(f"Total Memory: {format_bytes(memory['total'])}")
print(f"Available Memory: {format_bytes(memory['available'])}")
# Test disk usage
print("Disk Usage:")
for disk in get_disk_usage():
print(f" {disk['device']}: {disk['percent']:.1f}% used")
