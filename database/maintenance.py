"""
Database Maintenance & Automation
Author: Member 3
"""

import threading
import time
from database.advanced_models import get_advanced_db_manager
import logging

logger = logging.getLogger(__name__)

class DBMaintenanceManager:
    """
    Manages automated database maintenance tasks like index rebuilding and partition management.
    """

    def __init__(self):
        """Initializes the DBMaintenanceManager with a database connection."""
        self.db = get_advanced_db_manager()
        logger.info("DBMaintenanceManager initialized.")
        # Store a reference to threads to prevent them from being garbage collected prematurely
        self._threads = {}

    def schedule_index_rebuild(self, hours: int = 24):
        """
        Schedules a background thread to rebuild database indexes at a specified interval.

        Args:
            hours (int): The interval in hours between each index rebuild.
        """
        if 'index_rebuild' in self._threads and self._threads['index_rebuild'].is_alive():
            logger.warning("Index rebuild thread is already running. Skipping new schedule.")
            return

        def loop():
            logger.info(f"Index rebuild thread started, scheduled to run every {hours} hours.")
            while True:
                try:
                    self.db.rebuild_indexes()
                    logger.info('Indexes rebuilt successfully.')
                except Exception as e:
                    logger.error(f'Index rebuild failed: {e}')
                time.sleep(hours * 3600)

        t = threading.Thread(target=loop, daemon=True, name='IndexRebuildThread')
        self._threads['index_rebuild'] = t
        t.start()
        
    def schedule_partition_check(self, hours: int = 6):
        """
        Schedules a background thread to maintain database partitions at a specified interval.

        Args:
            hours (int): The interval in hours between each partition maintenance.
        """
        if 'partition_check' in self._threads and self._threads['partition_check'].is_alive():
            logger.warning("Partition check thread is already running. Skipping new schedule.")
            return

        def loop():
            logger.info(f"Partition check thread started, scheduled to run every {hours} hours.")
            while True:
                try:
                    self.db.maintain_partitions()
                    logger.info('Partitions maintained successfully.')
                except Exception as e:
                    logger.error(f'Partition maintenance failed: {e}')
                time.sleep(hours * 3600)

        t = threading.Thread(target=loop, daemon=True, name='PartitionCheckThread')
        self._threads['partition_check'] = t
        t.start()
