
"""
Data Export Module
Author: Member 4
"""
import csv
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

def export_to_csv(data: List[Dict[str, Any]], filename: str, directory: str = "data/exports"):
	"""
	Export data to CSV file
	Args:
		data: List of dictionaries to export
		filename: Name of the CSV file
		directory: Directory to save the file
	"""
	print(f"CSV export placeholder - Day 1: {filename}")
	Path(directory).mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	full_filename = f"{filename}_{timestamp}.csv"
	filepath = Path(directory) / full_filename
	if data:
		df = pd.DataFrame(data)
		df.to_csv(filepath, index=False)
		print(f"Data exported to {filepath}")
	else:
		print("No data to export")

def export_to_json(data: Dict[str, Any], filename: str, directory: str = "data/exports"):
	"""
	Export data to JSON file
	Args:
		data: Dictionary to export
		filename: Name of the JSON file
		directory: Directory to save the file
	"""
	print(f"JSON export placeholder - Day 1: {filename}")
	Path(directory).mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	full_filename = f"{filename}_{timestamp}.json"
	filepath = Path(directory) / full_filename
	with open(filepath, 'w') as f:
		json.dump(data, f, indent=2, default=str)
	print(f"Data exported to {filepath}")

def create_directories():
	"""Create necessary directories for data storage"""
	directories = [
		"data/logs",
		"data/exports",
		"data/snapshots",
		"data/logs/archive"
	]
	for directory in directories:
		Path(directory).mkdir(parents=True, exist_ok=True)
		print(f"Created directory: {directory}")

if __name__ == "__main__":
	create_directories()
	# Test CSV export
	sample_data = [
		{"pid": 1234, "name": "python", "cpu": 5.2},
		{"pid": 5678, "name": "chrome", "cpu": 15.8}
	]
	export_to_csv(sample_data, "test_processes")
	# Test JSON export
	sample_json = {
		"timestamp": datetime.now().isoformat(),
		"cpu_usage": 25.5,
		"memory_usage": 67.3
	}
	export_to_json(sample_json, "test_stats")
