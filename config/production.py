"""
Production configuration for System Monitor
Author: Member 1
"""

import os
from datetime import timedelta


class ProductionConfig:
    """Production environment configuration"""

    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-super-secret-key-change-in-production')
    DEBUG = False
    TESTING = False

    # Database Configuration
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///data/production.db')
    DATABASE_POOL_SIZE = int(os.environ.get('DATABASE_POOL_SIZE', '20'))
    DATABASE_POOL_TIMEOUT = int(os.environ.get('DATABASE_POOL_TIMEOUT', '30'))
    DATABASE_POOL_RECYCLE = int(os.environ.get('DATABASE_POOL_RECYCLE', '3600'))

    # Security Configuration
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)

    # Rate Limiting Configuration
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')
    RATELIMIT_DEFAULT = "1000/hour"
    RATELIMIT_HEADERS_ENABLED = True

    # Monitoring Configuration
    MONITORING_ENABLED = True
    HEALTH_CHECK_INTERVAL = 30
    PERFORMANCE_MONITORING = True

    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'WARNING')
    LOG_FILE = os.environ.get('LOG_FILE', 'data/logs/production.log')
    LOG_MAX_SIZE = int(os.environ.get('LOG_MAX_SIZE', '10485760'))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', '5'))

    # WebSocket Configuration
    WEBSOCKET_PING_INTERVAL = 25
    WEBSOCKET_PING_TIMEOUT = 60
    MAX_WEBSOCKET_CONNECTIONS = int(os.environ.get('MAX_WEBSOCKET_CONNECTIONS', '100'))

    # Cache Configuration
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'redis')
    CACHE_REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', '300'))

    # CORS Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',')

    # API Configuration
    API_VERSION = '1.0'
    API_RATE_LIMIT = "100/minute"
    API_DOCUMENTATION_URL = "/api/docs"

    @staticmethod
    def init_app(app):
        """Initialize production-specific configurations"""
        import logging
        from logging.handlers import SysLogHandler

        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.WARNING)
        app.logger.addHandler(syslog_handler)
