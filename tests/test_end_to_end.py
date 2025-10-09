"""
End-to-End Integration &amp; Safety Tests
Author: Member 5
"""
import pytest
from modules.cpu_memory import get_cpu_usage, get_memory_usage
from modules.processes import list_processes
from database.advanced_models import get_advanced_db_manager
from dashboards.main import launch_dashboard
from flask.testing import FlaskClient
def test_dashboard_smoke(client: FlaskClient):
response = client.get("/dashboard")
assert b"System Monitor Dashboard" in response.data
def test_cpu_monitoring_pipeline():
cpu = get_cpu_usage()
assert 0 &lt;= cpu &lt;= 100
memory = get_memory_usage()
assert isinstance(memory, dict)
assert 'percent' in memory
def test_process_list_appears():
processes = list_processes(limit=5)
assert isinstance(processes, list)
assert len(processes) &gt; 0
def test_db_health():
db = get_advanced_db_manager()
stats = db.get_database_stats()
assert 'metrics_count' in stats
