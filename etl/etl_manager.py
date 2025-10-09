"""
ETL Manager for Incremental Batch and CDC
Author: Member 3
"""

from database.advanced_models import get_advanced_db_manager
from datetime import datetime
import threading
import queue
import logging
import time

logger = logging.getLogger(__name__)

class ETLManager:
    """
    Manages and automates ETL processes using a multi-threaded, queue-based approach.
    Supports both incremental batch loads and Change Data Capture (CDC).
    """

    def __init__(self):
        """Initializes the ETLManager and its components."""
        self.db = get_advanced_db_manager()
        self.etl_queue = queue.Queue(maxsize=1000)
        self.active = False
        self._etl_thread = None
        logger.info("ETLManager initialized with a queue of size 1000.")

    def start(self):
        """
        Starts the ETL processing loop in a background thread.
        Prevents starting multiple threads if one is already active.
        """
        if self.active:
            logger.warning("ETL loop is already active. Ignoring start request.")
            return

        self.active = True
        self._etl_thread = threading.Thread(target=self.etl_loop, daemon=True, name='ETL_Loop')
        self._etl_thread.start()
        logger.info("ETL processing loop started.")

    def stop(self):
        """
        Signals the ETL loop to stop gracefully.
        The thread will finish its current task and then exit.
        """
        if self.active:
            self.active = False
            logger.info("ETL loop stop signal sent. Waiting for it to finish.")
            # Optional: Add a timeout to join if you need to ensure it's stopped before exiting the main program.
            # self._etl_thread.join(timeout=5)
            # if self._etl_thread.is_alive():
            #     logger.warning("ETL thread did not stop gracefully within the timeout.")
        else:
            logger.warning("ETL loop is not active. Ignoring stop request.")
            
    def etl_loop(self):
        """
        The main loop for processing ETL jobs from the queue.
        It runs continuously until the 'active' flag is set to False.
        """
        while self.active:
            try:
                # Use a short timeout to prevent the thread from blocking indefinitely,
                # which allows it to check the 'self.active' flag and exit gracefully.
                job = self.etl_queue.get(timeout=1)  # Using a 1-second timeout
                self.process_job(job)
                self.etl_queue.task_done()  # Signal that the task is complete
            except queue.Empty:
                # This is a normal event when no jobs are in the queue.
                # The loop continues to check the `self.active` flag.
                continue
            except Exception as e:
                # Catching general exceptions to prevent the thread from crashing.
                logger.error(f"An error occurred during job processing: {e}", exc_info=True)
                # Important: If a job fails, you may or may not want to mark it as done.
                # A robust system would have a retry mechanism.
                self.etl_queue.task_done()

    def process_job(self, job: dict):
        """
        Executes the appropriate ETL function based on the job type.
        
        Args:
            job (dict): A dictionary containing job details, e.g., {'type': 'incremental', ...}.
        """
        job_type = job.get('type')
        if not job_type:
            logger.error("Job received without a 'type' field. Ignoring.")
            return

        try:
            if job_type == 'incremental':
                logger.info(f"Processing incremental ETL job.")
                self.db.run_etl_incremental(job)
            elif job_type == 'cdc':
                logger.info(f"Processing CDC ETL job.")
                self.db.run_etl_cdc(job)
            else:
                logger.warning(f"Unknown job type '{job_type}'. Ignoring job.")
        except Exception as e:
            logger.error(f"Error processing job of type '{job_type}': {e}", exc_info=True)

    def add_job(self, job: dict):
        """
        Adds a new ETL job to the queue.
        
        Args:
            job (dict): The job to be added to the queue.
        """
        if not self.active:
            logger.warning("ETL loop is not active. Job will not be processed. Use .start() first.")
            return

        try:
            self.etl_queue.put(job, block=False)
            logger.info(f"Job added to the queue.")
        except queue.Full:
            logger.error("ETL queue is full. Job could not be added.")
