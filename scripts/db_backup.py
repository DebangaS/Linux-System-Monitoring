"""
Database backup script
Author: Member 1
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config


def backup_database(config_name='development'):
    """Create a backup of the database"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        app_config = config[config_name]()
        db_path = app_config.DATABASE_PATH
        backup_dir = Path(app_config.BACKUP_DIRECTORY)

        # Create backup directory if it doesn't exist
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"monitor_backup_{timestamp}.db"
        backup_path = backup_dir / backup_filename

        # Copy database file
        if os.path.exists(db_path):
            shutil.copy2(db_path, backup_path)

            # Get file sizes
            original_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
            backup_size = os.path.getsize(backup_path) / (1024 * 1024)  # MB

            logger.info("✅ Database backup created successfully")
            logger.info(f"Original: {db_path} ({original_size:.2f} MB)")
            logger.info(f"Backup: {backup_path} ({backup_size:.2f} MB)")

            # Clean up old backups (keep last 5)
            cleanup_old_backups(backup_dir, keep_count=5)
        else:
            logger.error(f"❌ Database file not found: {db_path}")

    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise


def cleanup_old_backups(backup_dir, keep_count=5):
    """Clean up old backup files"""
    try:
        # Get all backup files
        backup_files = list(backup_dir.glob("monitor_backup_*.db"))

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old backups
        for backup_file in backup_files[keep_count:]:
            backup_file.unlink()
            logging.info(f"Removed old backup: {backup_file.name}")

    except Exception as e:
        logging.error(f"Error cleaning up old backups: {e}")


if __name__ == '__main__':
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'development'
    backup_database(config_name)
