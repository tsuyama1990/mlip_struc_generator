import os
import uuid
import logging
import threading
import concurrent.futures
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List
from pydantic import BaseModel, Field

from nnp_gen.core.config import AppConfig
from nnp_gen.pipeline.runner import PipelineRunner
from nnp_gen.web_ui.utils import log_capture_to_file

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    log_file_path: str
    output_dir: str
    error_message: Optional[str] = None

class JobManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(JobManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Using ThreadPoolExecutor as MDExplorer uses ProcessPoolExecutor internally.
        # Daemon processes cannot spawn children, so we must use threads here.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.jobs: Dict[str, JobInfo] = {}
        self._jobs_lock = threading.RLock() # Use RLock for internal safety
        self._initialized = True

        # Determine logs directory relative to where running
        self.logs_root = Path("logs")
        self.logs_root.mkdir(exist_ok=True)

    def submit_job(self, config: AppConfig) -> str:
        """
        Submit a job with the given configuration.
        Returns the job_id.
        """
        job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Isolate output directory
        base_output = Path(config.output_dir)
        effective_output_dir = base_output / job_id

        # Clone config (Pydantic models are mutable, but we want a copy for safety)
        # Assuming config.model_copy(deep=True) works in Pydantic V2
        job_config = config.model_copy(deep=True)
        job_config.output_dir = str(effective_output_dir)

        log_file = self.logs_root / f"{job_id}.log"

        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            start_time=datetime.now(),
            log_file_path=str(log_file),
            output_dir=str(effective_output_dir)
        )

        with self._jobs_lock:
            self.jobs[job_id] = job_info

        # Submit to executor
        self.executor.submit(self._run_job_wrapper, job_id, job_config)

        return job_id

    def _run_job_wrapper(self, job_id: str, config: AppConfig):
        """
        Wrapper to run the pipeline, capturing logs and handling status.
        """
        # Acquire job info safely
        with self._jobs_lock:
            job = self.jobs.get(job_id)

        if not job:
            logger.error(f"Job {job_id} not found in wrapper")
            return

        job.status = JobStatus.RUNNING

        log_path = job.log_file_path

        # Capture logs from 'nnp_gen' logger to the file
        # We assume nnp_gen is the parent logger for all core modules
        try:
            # Note: log_capture_to_file uses ThreadLogFilter to isolate logs
            with log_capture_to_file("nnp_gen", log_path):
                try:
                    # We also need to log the start here so it appears in the file
                    # Ensure level is INFO or lower so it gets written
                    job_logger = logging.getLogger("nnp_gen.job_manager")
                    if job_logger.getEffectiveLevel() > logging.INFO:
                        job_logger.setLevel(logging.INFO)

                    job_logger.info(f"Starting Job {job_id}")
                    job_logger.info(f"Output Directory: {config.output_dir}")

                    runner = PipelineRunner(config)
                    runner.run()

                    with self._jobs_lock:
                        job = self.jobs[job_id]  # Re-fetch to be safe
                        job.status = JobStatus.COMPLETED
                        job.end_time = datetime.now()

                    job_logger.info(f"Job {job_id} Completed Successfully")
                except Exception as inner_e:
                    # Log before updating status to avoid race conditions in monitoring
                    logging.getLogger("nnp_gen").error(f"Job Execution Failed: {inner_e}", exc_info=True)

                    with self._jobs_lock:
                        job = self.jobs[job_id]
                        job.status = JobStatus.FAILED
                        job.end_time = datetime.now()
                        job.error_message = str(inner_e)

                    raise inner_e

        except Exception as e:
            logger.error(f"Job {job_id} failed (caught in wrapper): {e}")

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        with self._jobs_lock:
            return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[JobInfo]:
        # Return sorted by start time desc
        with self._jobs_lock:
            return sorted(list(self.jobs.values()), key=lambda x: x.start_time, reverse=True)

    def get_log_content(self, job_id: str) -> str:
        # We don't need to lock to read the file, but let's check job existence safely
        with self._jobs_lock:
            job = self.jobs.get(job_id)

        if not job or not os.path.exists(job.log_file_path):
            return ""

        try:
            with open(job.log_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return "Error reading log file."

    def get_status(self, job_id: str) -> Optional[JobStatus]:
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            return job.status if job else None
