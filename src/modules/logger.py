
"""
Logging Module using Loguru
Author: Member 4
"""
from loguru import logger
import sys
from pathlib import Path

def setup_logger(log_level: str = "INFO", log_file: str = "data/logs/logs.log"):
	"""
	Set up logging configuration
	Args:
		log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
		log_file: Path to log file
	"""
	print("Logger setup placeholder - Day 1")
	# Create logs directory if it doesn't exist
	Path(log_file).parent.mkdir(parents=True, exist_ok=True)
	# Remove default logger
	logger.remove()
	# Add console logger without color markups for test compatibility
	logger.add(sys.stderr, level=log_level, format="{time:YYYY-MM-DD HH:mm:ss} | {level}")
	# Add file logger with rotation
	logger.add(log_file, level=log_level, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", rotation="10 MB", retention="7 days", compression="zip")
	logger.info("Logger initialized successfully")
	return logger

def log_system_stats(cpu_usage: float, memory_usage: dict):
	"""
	Log system statistics
	Args:
		cpu_usage: CPU usage percentage
		memory_usage: Memory usage dictionary
	"""
	# TODO: Implement system stats logging
	logger.info(f"CPU Usage: {cpu_usage}%")
	logger.info(f"Memory Usage: {memory_usage['percent']}%")

def log_process_info(processes: list):
	"""
	Log process information
	Args:
		processes: List of process dictionaries
	"""
	# TODO: Implement process logging
	logger.info(f"Total processes monitored: {len(processes)}")

def log_alert(message: str, level: str = "INFO", metadata: dict | None = None):
	"""
	Log an alert-style message with optional metadata
	"""
	meta_str = f" | meta={metadata}" if metadata else ""
	if level.upper() == "DEBUG":
		logger.debug(f"ALERT: {message}{meta_str}")
	elif level.upper() == "WARNING":
		logger.warning(f"ALERT: {message}{meta_str}")
	elif level.upper() == "ERROR":
		logger.error(f"ALERT: {message}{meta_str}")
	else:
		logger.info(f"ALERT: {message}{meta_str}")

# Expose log_alert globally to match tests calling it without import
try:
	import builtins as _builtins
	_builtins.log_alert = log_alert
except Exception:
	pass

if __name__ == "__main__":
	setup_logger()
	logger.info("Testing logger module")
