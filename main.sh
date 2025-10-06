#!/bin/bash
# System Monitor - Main Shell Script
# Author: Member 1
echo "System Monitor - Main Runner"
echo "============================"
# Check if virtual environment exists
if [ ! -d "venv" ]; then
echo "Creating virtual environment..."
python3 -m venv venv
fi
# Activate virtual environment
source venv/bin/activate
# Install requirements if not installed
pip install -r requirements.txt
# Create data directories
mkdir -p data/logs data/exports data/snapshots data/logs/archive
# Run the main application
python src/main.py "$@"
