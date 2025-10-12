import psutil

class ProcessMonitor:
    def get_all_processes(self, limit=None):
        """Return all running processes, optionally limited in count."""
        processes = self.get_process_snapshot()
        if limit is not None:
            return processes[:limit]
        return processes
    """Basic process monitor for system analytics."""
    @staticmethod
    def get_process_snapshot():
        """Return a snapshot of current running processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return processes

process_monitor = ProcessMonitor()
