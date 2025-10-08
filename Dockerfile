# ==============================================
# System Monitor Docker Configuration
# Author: Member 1
# ==============================================

# Base Image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    FLASK_APP=src/main.py

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data/logs data/exports data/backups

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash sysmonitor && \
    chown -R sysmonitor:sysmonitor /app

# Switch to non-root user
USER sysmonitor

# Expose Flask/Gunicorn port
EXPOSE 5000

# Health check to verify container health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/v1/health')" || exit 1

# Run application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "src.main:app"]
