"""
Development configuration for System Monitor
Author: Member 1
"""

from datetime import timedelta


class DevelopmentConfig:
    """Development environment configuration"""

    # Flask Configuration
    SECRET_KEY = 'dev-secret-key-not-for-production'
    DEBUG = True
    TESTING = False

    # Database Configuration
    DATABASE_URL = 'sqlite:///data/development.db'
    DATABASE_POOL_SIZE = 5
    DATABASE_POOL_TIMEOUT = 10
    DATABASE_POOL_RECYCLE = 1800

    # Security Configuration (Relaxed for development)
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)

    # Rate Limiting Configuration (More lenient)
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_DEFAULT = "10000/hour"
    RATELIMIT_HEADERS_ENABLED = True

    # Monitoring Configuration
    MONITORING_ENABLED = True
    HEALTH_CHECK_INTERVAL = 10
    PERFORMANCE_MONITORING = True

    # Logging Configuration
    LOG_LEVEL = 'DEBUG'
    LOG_FILE = 'data/logs/development.log'
    LOG_MAX_SIZE = 5242880  # 5MB
    LOG_BACKUP_COUNT = 3

    # WebSocket Configuration
    WEBSOCKET_PING_INTERVAL = 25
    WEBSOCKET_PING_TIMEOUT = 60
    MAX_WEBSOCKET_CONNECTIONS = 50

    # Cache Configuration
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 60

    # CORS Configuration (Allow all in development)
    CORS_ORIGINS = ["*"]

    # API Configuration
    API_VERSION = '1.0-dev'
    API_RATE_LIMIT = "1000/minute"
    API_DOCUMENTATION_URL = "/api/docs"

    @staticmethod
    def init_app(app):
        """Initialize development-specific configurations"""
        pass
