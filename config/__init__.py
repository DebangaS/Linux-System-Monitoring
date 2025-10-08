"""
Configuration management for System Monitor
Author: Member 1
"""

import os
from .development import DevelopmentConfig
from .production import ProductionConfig


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}


def get_config():
    """Get configuration based on environment"""
    config_name = os.environ.get('FLASK_ENV', 'development')
    return config.get(config_name, config['default'])
