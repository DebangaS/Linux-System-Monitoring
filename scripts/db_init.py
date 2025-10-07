"""
Database initialization script
Author: Member 1
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from database.models import db_manager  # Ensure this exists
from config import config


def initialize_database(config_name: str = 'development'):
    """Initialize the database with proper configuration"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting database initialization...")

        # Load configuration
        app_config = config.get(config_name)
        if not app_config:
            logger.warning(f"Unknown config '{config_name}', using default.")
            app_config = config['default']

        # Initialize the database
        db_manager.init_database()
        logger.info("Database initialized successfully.")

        # Get database stats
        stats = db_manager.get_database_stats()
        logger.info(f"Database size: {stats.get('db_size_mb', 0):.2f} MB")

        # Test database operations
        logger.info("Testing database operations...")

        # Example test data
        test_system_info = {
            'hostname': 'test-host',
            'platform': 'Test Platform',
            'processor': 'Test Processor',
            'boot_time': '2025-01-01T00:00:00',
            'uptime_seconds': 3600,
            'users': ['test_user']
        }

        # Test storing system info
        success = db_manager.store_system_info(test_system_info)
        if success:
            logger.info("✅ System info storage test passed.")
        else:
            logger.error("❌ System info storage test failed.")

        logger.info("Database initialization completed successfully!")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Determine configuration name from command line (default: 'development')
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'development'
    initialize_database(config_name)
