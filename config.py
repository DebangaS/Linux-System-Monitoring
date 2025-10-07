"""
Enhanced configuration with database settings
Author: Member 1
"""

import os
from datetime import timedelta


class Config:
    """Base configuration class with database settings"""

    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = True

    # Application Settings
    APP_NAME = "Resource Monitor Dashboard"
    APP_VERSION = "1.3.0"  # Updated for Day 3

    # Monitoring Settings
    MONITORING_INTERVAL = 2  # seconds
    DATA_RETENTION_DAYS = 7  # days
    MAX_DATA_POINTS = 100  # maximum data points to store in memory

    # API Settings
    API_PREFIX = '/api/v1'
    CORS_ORIGINS = ['http://localhost:5000', 'http://127.0.0.1:5000']

    # Database Settings - NEW for Day 3
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///data/monitor.db')
    DATABASE_PATH = 'data/monitor.db'
    DATABASE_BACKUP_INTERVAL_HOURS = 24
    DATABASE_CLEANUP_INTERVAL_HOURS = 6

    # Data Export Settings
    EXPORT_FORMATS = ['csv', 'json', 'excel']
    EXPORT_DIRECTORY = 'data/exports'
    BACKUP_DIRECTORY = 'data/backups'

    # Performance Settings
    DATABASE_POOL_SIZE = 5
    DATABASE_POOL_TIMEOUT = 30
    DATABASE_POOL_RECYCLE = 3600

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'data/logs/app.log'
    DATABASE_LOG_FILE = 'data/logs/database.log'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    MONITORING_INTERVAL = 1  # More frequent updates in development
    DATA_RETENTION_DAYS = 3  # Keep less data in development


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    MONITORING_INTERVAL = 5  # Less frequent in production
    DATA_RETENTION_DAYS = 30  # Keep more data in production

    # Production database settings
    DATABASE_POOL_SIZE = 10
    DATABASE_CLEANUP_INTERVAL_HOURS = 12


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_PATH = 'data/test_monitor.db'
    DATA_RETENTION_DAYS = 1


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
