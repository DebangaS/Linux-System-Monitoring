 """
 Process Management Module - Day 2 Implementation
 Author: Member 3
 """
 import psutil
 import time
 from typing import List, Dict, Optional, Callable
 from datetime import datetime
 def list_processes(limit: int = 50, sort_by: str = 'cpu_percent') -&gt; List[Dict[str, any]]
 """
 List running processes with detailed information
    Args:
        limit: Maximum number of processes to return
        sort_by: Field to sort by (cpu_percent, memory_percent, pid, name)
    Returns:
        List of process information dictionaries
    """
    processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'username', 'status', 'cpu_percent',
                                        'memory_info', 'create_time', 'cmdline', 'ppid', 'nu
            try:
                process_info = {
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'username': proc.info['username'] or 'N/A',
                    'status': proc.info['status'],
                    'cpu_percent': proc.info['cpu_percent'] or 0.0,
                    'memory_percent': proc.info['memory_percent'] or 0.0,
                    'memory_rss': proc.info['memory_info'].rss if proc.info['memory_info'] e
                    'memory_vms': proc.info['memory_info'].vms if proc.info['memory_info'] e
                    'create_time': proc.info['create_time'],
                    'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else 
                    'ppid': proc.info['ppid'],
                    'num_threads': proc.info['num_threads'] or 0,
                    'running_time': time.time() - proc.info['create_time'] if proc.info['cre
                }
                processes.append(process_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"Error listing processes: {e}")
        return []
    
    # Sort processes
    try:
        if sort_by in ['cpu_percent', 'memory_percent', 'pid', 'memory_rss', 'memory_vms', 
            processes.sort(key=lambda x: x[sort_by], reverse=True)
        elif sort_by in ['name', 'username', 'status']:
            processes.sort(key=lambda x: x[sort_by])
    except KeyError:
        print(f"Invalid sort field: {sort_by}")
    
    return processes[:limit]
 def filter_processes_by_name(name_pattern: str, exact_match: bool = False) -&gt; List[Dict[s
    """
    Filter processes by name pattern
    Args:
        name_pattern: Pattern to match process names
        exact_match: If True, match exactly; if False, match substring
    Returns:
        List of matching processes
    """
    all_processes = list_processes(limit=1000)  # Get more processes for filtering
    
    if exact_match:
        return [proc for proc in all_processes if proc['name'] == name_pattern]
    else:
        return [proc for proc in all_processes if name_pattern.lower() in proc['name'].lower
 def filter_processes_by_user(username: str) -&gt; List[Dict[str, any]]:
    """
    Filter processes by username
    Args:
        username: Username to filter by
    Returns:
        List of processes owned by the specified user
    """
    all_processes = list_processes(limit=1000)
    return [proc for proc in all_processes if proc['username'].lower() == username.lower()]
 def filter_processes_by_cpu_usage(min_cpu: float = 1.0) -&gt; List[Dict[str, any]]:
    """
    Filter processes by minimum CPU usage
    Args:
        min_cpu: Minimum CPU usage percentage
    Returns:
        List of processes with CPU usage &gt;= min_cpu
    """
    all_processes = list_processes(limit=1000, sort_by='cpu_percent')
    return [proc for proc in all_processes if proc['cpu_percent'] &gt;= min_cpu]
 def filter_processes_by_memory_usage(min_memory: float = 1.0) -&gt; List[Dict[str, any]]:
    """
    Filter processes by minimum memory usage
    Args:
        min_memory: Minimum memory usage percentage
    Returns:
        List of processes with memory usage &gt;= min_memory
    """
    all_processes = list_processes(limit=1000, sort_by='memory_percent')
    return [proc for proc in all_processes if proc['memory_percent'] &gt;= min_memory]
 def filter_processes_by_status(status: str) -&gt; List[Dict[str, any]]:
    """
    Filter processes by status
    Args:
        status: Process status (running, sleeping, zombie, etc.)
    Returns:
        List of processes with matching status
    """
    all_processes = list_processes(limit=1000)
    return [proc for proc in all_processes if proc['status'].lower() == status.lower()]
 def get_process_details(pid: int) -&gt; Optional[Dict[str, any]]:
    """
    Get comprehensive information about a specific process
    Args:
        pid: Process ID
    Returns:
        Detailed process information or None if not found
    """
    try:
        proc = psutil.Process(pid)
        
        # Get connections (if accessible)
        connections = []
        try:
            for conn in proc.connections():
                connections.append({
                    'fd': conn.fd,
                    'family': conn.family.name if hasattr(conn.family, 'name') else str(conn
                    'type': conn.type.name if hasattr(conn.type, 'name') else str(conn.type)
                    'laddr': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else '',
                    'raddr': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else '',
                    'status': conn.status
                })
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        
        # Get open files (if accessible)
        open_files = []
        try:
            for file in proc.open_files():
                open_files.append({
                    'path': file.path,
                    'fd': file.fd,
                    'position': getattr(file, 'position', 'N/A'),
                    'mode': getattr(file, 'mode', 'N/A'),
                    'flags': getattr(file, 'flags', 'N/A')
                })
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        
        return {
            'pid': proc.pid,
            'ppid': proc.ppid(),
            'name': proc.name(),
            'exe': proc.exe() if hasattr(proc, 'exe') else 'N/A',
            'cmdline': proc.cmdline(),
            'username': proc.username(),
            'create_time': proc.create_time(),
            'status': proc.status(),
            'cpu_percent': proc.cpu_percent(),
            'memory_percent': proc.memory_percent(),
            'memory_info': proc.memory_info()._asdict(),
            'io_counters': proc.io_counters()._asdict() if hasattr(proc, 'io_counters') else
            'num_threads': proc.num_threads(),
            'num_fds': proc.num_fds() if hasattr(proc, 'num_fds') else 'N/A',
            'connections': connections,
            'open_files': open_files,
            'children': [child.pid for child in proc.children()],
            'cpu_times': proc.cpu_times()._asdict(),
            'cpu_affinity': proc.cpu_affinity() if hasattr(proc, 'cpu_affinity') else [],
            'nice': proc.nice(),
            'ionice': proc.ionice()._asdict() if hasattr(proc, 'ionice') else {},
            'rlimit': {}  # Will be populated below
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
        print(f"Error getting process details for PID {pid}: {e}")
        return None
 def get_top_processes(metric: str = 'cpu_percent', count: int = 10) -&gt; List[Dict[str, any
    """
    Get top processes by specified metric
    Args:
        metric: Metric to sort by (cpu_percent, memory_percent, memory_rss)
        count: Number of top processes to return
    Returns:
        List of top processes
    """
    return list_processes(limit=count, sort_by=metric)
 def kill_process(pid: int, force: bool = False) -&gt; bool:
    """
    Kill a process by PID
    Args:
        pid: Process ID to kill
        force: If True, use SIGKILL; if False, use SIGTERM
    Returns:
        True if successful, False otherwise
    """
    try:
        proc = psutil.Process(pid)
        if force:
            proc.kill()  # SIGKILL
        else:
            proc.terminate()  # SIGTERM
        
        # Wait for process to terminate
        proc.wait(timeout=5)
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
        print(f"Error killing process {pid}: {e}")
        return False
 def get_process_tree(pid: int) -&gt; Dict[str, any]:
    """
    Get process tree starting from specified PID
    Args:
        pid: Root process ID
    Returns:
        Dictionary representing process tree
    """
    try:
        root_proc = psutil.Process(pid)
        tree = {
            'pid': root_proc.pid,
            'name': root_proc.name(),
            'status': root_proc.status(),
            'children': []
        }
        
        def build_tree(proc, tree_node):
            for child in proc.children():
                try:
                    child_node = {
                        'pid': child.pid,
                        'name': child.name(),
                        'status': child.status(),
                        'children': []
                    }
                    tree_node['children'].append(child_node)
                    build_tree(child, child_node)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        build_tree(root_proc, tree)
        return tree
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"Error building process tree for PID {pid}: {e}")
        return {}
 def monitor_process(pid: int, duration: int = 60, interval: int = 5) -&gt; List[Dict[str, an
    """
    Monitor a specific process over time
    Args:
        pid: Process ID to monitor
        duration: Monitoring duration in seconds
        interval: Measurement interval in seconds
    Returns:
        List of process measurements over time
    """
    measurements = []
    start_time = time.time()
    
    try:
        proc = psutil.Process(pid)
        
        while time.time() - start_time &lt; duration:
            try:
                measurement = {
                    'timestamp': time.time(),
                    'pid': proc.pid,
                    'name': proc.name(),
                    'cpu_percent': proc.cpu_percent(),
                    'memory_percent': proc.memory_percent(),
                    'memory_rss': proc.memory_info().rss,
                    'memory_vms': proc.memory_info().vms,
                    'status': proc.status(),
                    'num_threads': proc.num_threads(),
                    'num_fds': proc.num_fds() if hasattr(proc, 'num_fds') else 0
                }
                measurements.append(measurement)
                time.sleep(interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"Process {pid} no longer accessible")
                break
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"Cannot monitor process {pid}: {e}")
    
    return measurements
 def format_process_table(processes: List[Dict[str, any]], max_width: int = 120) -&gt; str:
    """
    Format process list as a table
    Args:
        processes: List of process dictionaries
        max_width: Maximum table width
    Returns:
        Formatted table string
    """
    if not processes:
        return "No processes found."
    
    # Table headers
    headers = ['PID', 'Name', 'User', 'CPU%', 'Mem%', 'Status', 'Threads']
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for proc in processes[:10]:  # Check first 10 for width calculation
        col_widths[0] = max(col_widths[0], len(str(proc['pid'])))
        col_widths[1] = max(col_widths[1], len(proc['name'][:15]))  # Truncate long names
        col_widths[2] = max(col_widths[2], len(proc['username'][:10]))  # Truncate long user
        col_widths[3] = max(col_widths[3], len(f"{proc['cpu_percent']:.1f}"))
        col_widths[4] = max(col_widths[4], len(f"{proc['memory_percent']:.1f}"))
        col_widths[5] = max(col_widths[5], len(proc['status']))
        col_widths[6] = max(col_widths[6], len(str(proc['num_threads'])))
    
    # Create table
    table_lines = []
    
    # Header
    header_line = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    table_lines.append(header_line)
    table_lines.append("-" * len(header_line))
    
    # Rows
    for proc in processes:
        row = [
            str(proc['pid']).ljust(col_widths[0]),
            proc['name'][:15].ljust(col_widths[1]),  # Truncate name
            proc['username'][:10].ljust(col_widths[2]),  # Truncate username
            f"{proc['cpu_percent']:.1f}".rjust(col_widths[3]),
            f"{proc['memory_percent']:.1f}".rjust(col_widths[4]),
            proc['status'].ljust(col_widths[5]),
            str(proc['num_threads']).rjust(col_widths[6])
        ]
        table_lines.append(" | ".join(row))
    
    return "\n".join(table_lines)
 if __name__ == "__main__":
print("Testing Enhanced Process Management - Day 2")
 print("=" * 60)
 # Test process listing
 processes = list_processes(10)
 print(f"Found {len(processes)} processes")
 print(format_process_table(processes))
 # Test filtering
 print(f"\nHigh CPU processes (&gt;1%):")
 high_cpu = filter_processes_by_cpu_usage(1.0)
 print(format_process_table(high_cpu[:5]))
 # Test process details for current process
 current_pid = psutil.Process().pid
 details = get_process_details(current_pid)
 if details:
 print(f"\nDetails for current process (PID {current_pid}):")
 print(f"Name: {details['name']}")
 print(f"CPU: {details['cpu_percent']:.1f}%")
 print(f"Memory: {details['memory_percent']:.1f}%")
