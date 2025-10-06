#!/usr/bin/env python3
"""
System Monitor - Main Entry Point
Author: Team System Monitor
"""
import argparse
import sys
from pathlib import Path
# Add src to path for imports
from modules.cpu_memory import get_cpu_usage, get_memory_usage, get_cpu_info
from modules.logger import setup_logger
Update main() function:
def main():
parser = argparse.ArgumentParser(description="System Monitor Tool")
parser.add_argument('--cpu', action='store_true', help='Show CPU usage')
parser.add_argument('--memory', action='store_true', help='Show memory usage')
parser.add_argument('--processes', action='store_true', help='List processes')
parser.add_argument('--dashboard', action='store_true', help='Launch CLI dashboard')
parser.add_argument('--web', action='store_true', help='Launch web dashboard')
parser.add_argument('--log', action='store_true', help='Enable logging')
parser.add_argument('--interval', type=int, default=1, help='Update interval in seconds
args = parser.parse_args()
if args.log:
logger = setup_logger()
logger.info("System Monitor started with logging enabled")
print("System Monitor v1.0 - Day 2")
if args.cpu:
cpu_usage = get_cpu_usage(interval=args.interval)
print(f"CPU Usage: {cpu_usage:.1f}%")
cpu_info = get_cpu_info()
print(f"Physical Cores: {cpu_info['physical_cores']}")
print(f"Total Cores: {cpu_info['total_cores']}")
