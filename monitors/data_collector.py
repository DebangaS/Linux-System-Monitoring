"""
Optimized data collection and batch processing
Author: Member 2
"""

import time
import threading
from datetime import datetime
from typing import Dict
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

from monitors.system_monitor import system_monitor
from monitors.process_monitor import process_monitor
from database.models import db_manager

logger = logging.getLogger(__name__)


class DataCollector:
    """Optimized data collector with batch processing"""

    def __init__(self, batch_size=10, max_queue_size=100):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        # Queues for batch data
        self.metrics_queue = Queue(maxsize=max_queue_size)
        self.process_queue = Queue(maxsize=max_queue_size)
        self.alert_queue = Queue(maxsize=max_queue_size)

        # States
        self.collecting = False
        self.processing = False

        # Threads
        self.collection_thread = None
        self.processing_thread = None

        # Statistics
        self.stats = {
            'collections': 0,
            'batch_stores': 0,
            'errors': 0,
            'queue_size': 0,
            'last_collection': None,
            'last_store': None
        }

    def start_collection(self, collection_interval=2, store_interval=10):
        """Start optimized data collection"""
        try:
            if self.collecting:
                logger.warning("Data collection already running")
                return False

            self.collecting = True
            self.processing = True

            # Start collection thread
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                args=(collection_interval,),
                daemon=True
            )
            self.collection_thread.start()

            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                args=(store_interval,),
                daemon=True
            )
            self.processing_thread.start()

            logger.info(f"Started data collection (collect: {collection_interval}s, store: {store_interval}s)")
            return True

        except Exception as e:
            logger.error(f"Error starting data collection: {e}")
            return False

    def _collection_loop(self, interval):
        """Main data collection loop"""
        while self.collecting:
            try:
                timestamp = datetime.now().isoformat()

                # Collect system metrics using thread pool
                with ThreadPoolExecutor(max_workers=3) as executor:
                    cpu_future = executor.submit(system_monitor.get_cpu_usage_enhanced)
                    memory_future = executor.submit(system_monitor.get_memory_usage_enhanced)
                    network_future = executor.submit(system_monitor.get_network_usage)

                    cpu_data = cpu_future.result(timeout=5)
                    memory_data = memory_future.result(timeout=5)
                    network_data = network_future.result(timeout=5)

                disk_data = system_monitor.get_disk_usage()

                metrics_data = {
                    'timestamp': timestamp,
                    'cpu': cpu_data,
                    'memory': memory_data,
                    'disk': disk_data,
                    'network': network_data
                }

                if not self.metrics_queue.full():
                    self.metrics_queue.put(metrics_data)
                    self.stats['collections'] += 1
                    self.stats['last_collection'] = timestamp
                else:
                    logger.warning("Metrics queue is full, dropping data")

                # Collect process data less frequently
                if self.stats['collections'] % 5 == 0:
                    try:
                        processes = process_monitor.get_all_processes(limit=30)
                        process_data = {
                            'timestamp': timestamp,
                            'processes': processes
                        }
                        if not self.process_queue.full():
                            self.process_queue.put(process_data)
                    except Exception as e:
                        logger.error(f"Error collecting process data: {e}")

                # Collect alerts
                alerts = system_monitor.get_alerts()
                if alerts:
                    alert_data = {
                        'timestamp': timestamp,
                        'alerts': alerts
                    }
                    if not self.alert_queue.full():
                        self.alert_queue.put(alert_data)

                # Update queue size stats
                self.stats['queue_size'] = (
                    self.metrics_queue.qsize()
                    + self.process_queue.qsize()
                    + self.alert_queue.qsize()
                )

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                self.stats['errors'] += 1
                time.sleep(interval)

    def _processing_loop(self, store_interval):
        """Process and store data in batches"""
        while self.processing:
            try:
                time.sleep(store_interval)
                self._process_metrics_batch()
                self._process_process_batch()
                self._process_alert_batch()
                self.stats['last_store'] = datetime.now().isoformat()
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.stats['errors'] += 1

    def _process_metrics_batch(self):
        """Process a batch of metrics data"""
        try:
            batch = []
            while len(batch) < self.batch_size:
                try:
                    data = self.metrics_queue.get_nowait()
                    batch.append(data)
                except Empty:
                    break

            if not batch:
                return

            for data in batch:
                success = db_manager.store_system_metrics(
                    data['cpu'], data['memory'], data['disk'], data['network']
                )
                if not success:
                    logger.error("Failed to store metrics data")
                    self.stats['errors'] += 1

            self.stats['batch_stores'] += 1
            logger.debug(f"Processed metrics batch of {len(batch)} items")

        except Exception as e:
            logger.error(f"Error processing metrics batch: {e}")
            self.stats['errors'] += 1

    def _process_process_batch(self):
        """Process a batch of process data"""
        try:
            batch = []
            while len(batch) < self.batch_size:
                try:
                    data = self.process_queue.get_nowait()
                    batch.append(data)
                except Empty:
                    break

            if not batch:
                return

            for data in batch:
                success = db_manager.store_process_snapshot(data['processes'])
                if not success:
                    logger.error("Failed to store process data")
                    self.stats['errors'] += 1

            logger.debug(f"Processed process batch of {len(batch)} items")

        except Exception as e:
            logger.error(f"Error processing process batch: {e}")
            self.stats['errors'] += 1

    def _process_alert_batch(self):
        """Process a batch of alert data"""
        try:
            batch = []
            while len(batch) < self.batch_size:
                try:
                    data = self.alert_queue.get_nowait()
                    batch.append(data)
                except Empty:
                    break

            if not batch:
                return

            for data in batch:
                for alert in data['alerts']:
                    success = db_manager.store_system_alert(
                        alert['type'],
                        alert['level'],
                        alert['message'],
                        alert.get('value')
                    )
                    if not success:
                        logger.error("Failed to store alert data")
                        self.stats['errors'] += 1

            logger.debug(f"Processed alert batch of {len(batch)} items")

        except Exception as e:
            logger.error(f"Error processing alert batch: {e}")
            self.stats['errors'] += 1

    def stop_collection(self):
        """Stop data collection"""
        try:
            logger.info("Stopping data collection...")
            self.collecting = False
            self.processing = False

            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=10)
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=10)

            # Process remaining queue items
            self._process_metrics_batch()
            self._process_process_batch()
            self._process_alert_batch()

            logger.info("Data collection stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping data collection: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            'collecting': self.collecting,
            'processing': self.processing,
            'stats': self.stats.copy(),
            'queue_sizes': {
                'metrics': self.metrics_queue.qsize(),
                'processes': self.process_queue.qsize(),
                'alerts': self.alert_queue.qsize()
            },
            'timestamp': datetime.now().isoformat()
        }


# Global instance
data_collector = DataCollector()
