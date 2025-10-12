import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Any

class data_analyzer:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def fetch_all(self, table: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table}")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    def get_cpu_usage_stats(self) -> Dict[str, float]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT AVG(usage), MAX(usage), MIN(usage) FROM cpu_usage")
            avg, max_, min_ = cursor.fetchone()
            return {"average": avg, "max": max_, "min": min_}

    def get_memory_usage_stats(self) -> Dict[str, float]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT AVG(usage), MAX(usage), MIN(usage) FROM memory_usage")
            avg, max_, min_ = cursor.fetchone()
            return {"average": avg, "max": max_, "min": min_}

    def get_disk_usage_stats(self) -> Dict[str, float]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT AVG(usage), MAX(usage), MIN(usage) FROM disk_usage")
            avg, max_, min_ = cursor.fetchone()
            return {"average": avg, "max": max_, "min": min_}

    def get_top_processes(self, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM processes ORDER BY cpu DESC LIMIT ?", (limit,)
            )
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    def store_system_metrics(
        self, cpu: Dict[str, float], memory: Dict[str, float], disk: Dict[str, float], network: Dict[str, float]
    ) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_metrics (cpu_usage, memory_usage, disk_usage, network_usage) VALUES (?, ?, ?, ?)",
                (cpu['usage'], memory['usage'], disk['usage'], network['usage']),
            )
            conn.commit()

    # Added methods required by tests
    def get_cpu_trends(self, hours: int = 24) -> Dict[str, Any]:
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT timestamp, data FROM system_metrics WHERE metric_type = 'cpu' AND timestamp >= ? ORDER BY timestamp",
                (cutoff,),
            )
            rows = cur.fetchall()
            values = []
            for r in rows:
                try:
                    obj = json.loads(r[1]) if isinstance(r[1], str) else r[1]
                    values.append(float(obj.get('usage_percent', 0)))
                except Exception:
                    continue
            if not values:
                return {'average': 0, 'minimum': 0, 'maximum': 0, 'samples': 0}
            return {
                'average': sum(values) / len(values),
                'minimum': min(values),
                'maximum': max(values),
                'samples': len(values)
            }

    def get_memory_trends(self, hours: int = 24) -> Dict[str, Any]:
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT data FROM system_metrics WHERE metric_type = 'memory' AND timestamp >= ? ORDER BY timestamp",
                (cutoff,),
            )
            rows = cur.fetchall()
            mem_values, swap_values = [], []
            for r in rows:
                try:
                    obj = json.loads(r[0]) if isinstance(r[0], str) else r[0]
                    mem_values.append(float(obj.get('usage_percent', 0)))
                    swap = obj.get('swap', {}) or {}
                    swap_values.append(float(swap.get('usage_percent', 0)))
                except Exception:
                    continue
            result = {
                'memory': {
                    'average': sum(mem_values) / len(mem_values) if mem_values else 0,
                    'minimum': min(mem_values) if mem_values else 0,
                    'maximum': max(mem_values) if mem_values else 0,
                },
                'swap': {
                    'average': sum(swap_values) / len(swap_values) if swap_values else 0,
                    'minimum': min(swap_values) if swap_values else 0,
                    'maximum': max(swap_values) if swap_values else 0,
                },
                'samples': len(mem_values)
            }
            return result

    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        return {
            'report_generated': True,
            'cpu_analysis': self.get_cpu_trends(hours),
            'memory_analysis': self.get_memory_trends(hours),
            'disk_analysis': self._metric_basic_stats('disk', hours),
            'network_analysis': self._metric_basic_stats('network', hours),
        }

    def _metric_basic_stats(self, metric_type: str, hours: int) -> Dict[str, Any]:
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT data FROM system_metrics WHERE metric_type = ? AND timestamp >= ? ORDER BY timestamp",
                (metric_type, cutoff),
            )
            rows = cur.fetchall()
            values = []
            key = 'total_usage_percent' if metric_type == 'disk' else 'sent_rate_kbps'
            alt_key = 'recv_rate_kbps' if metric_type == 'network' else 'usage_percent'
            for r in rows:
                try:
                    obj = json.loads(r[0]) if isinstance(r[0], str) else r[0]
                    if key in obj:
                        values.append(float(obj.get(key, 0)))
                    elif alt_key in obj:
                        values.append(float(obj.get(alt_key, 0)))
                except Exception:
                    continue
            if not values:
                return {'average': 0, 'minimum': 0, 'maximum': 0, 'samples': 0}
            return {
                'average': sum(values) / len(values),
                'minimum': min(values),
                'maximum': max(values),
                'samples': len(values),
            }

# Example usage:
# analyzer = data_analyzer('system_monitor.db')
# print(analyzer.get_cpu_usage_stats())
# print(analyzer.get_top_processes())
# resources = {
#     'cpu': {'usage': 23.5},
#     'memory': {'usage': 67.8},
#     'disk': {'usage': 45.2},
#     'network': {'usage': 12.3},
# }
# analyzer.store_system_metrics(
#     resources.get('cpu'),
#     resources.get('memory'),
#     resources.get('disk'),
#     resources.get('network'),
# )