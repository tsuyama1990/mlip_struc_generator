import os
import uuid
import logging
import threading
import subprocess
import shutil
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List
from pydantic import BaseModel
from omegaconf import OmegaConf

from nnp_gen.core.config import AppConfig

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
    pid: Optional[int] = None

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

        self.jobs: Dict[str, JobInfo] = {}
        self._jobs_lock = threading.RLock()
        self._initialized = True

        # Determine logs directory
        self.logs_root = Path("logs")
        self.logs_root.mkdir(exist_ok=True)
        
        # Start a background thread to monitor processes? 
        # For simplicity, we might just poll status when requested, or use a waiter thread per job.
        # Since we want real-time logs, we can just let the subprocess write to file 
        # and we read that file. We do need to know when it finishes.
        # We can use a thread per job to wait().

    def submit_job(self, config: AppConfig) -> str:
        """
        Submit a job with the given configuration.
        Returns the job_id.
        """
        job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Isolate output directory
        base_output = Path(config.output_dir)
        effective_output_dir = base_output / job_id
        effective_output_dir.mkdir(parents=True, exist_ok=True)

        # Clone config (Pydantic models are mutable, but we want a copy for safety)
        job_config = config.model_copy(deep=True)
        job_config.output_dir = str(effective_output_dir)

        # Write config.yaml
        config_path = effective_output_dir / "config.yaml"
        # Use OmegaConf to dump Pydantic model to YAML
        # Convert to container first
        cfg_container = OmegaConf.create(job_config.model_dump(mode='json'))
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg_container, f)

        log_file_path = self.logs_root / f"{job_id}.log"

        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            start_time=datetime.now(),
            log_file_path=str(log_file_path),
            output_dir=str(effective_output_dir)
        )

        with self._jobs_lock:
            self.jobs[job_id] = job_info

        # Launch in background thread to avoid blocking UI
        t = threading.Thread(target=self._launch_subprocess, args=(job_id, effective_output_dir, log_file_path))
        t.start()

        return job_id

    def _launch_subprocess(self, job_id: str, config_dir: Path, log_file_path: Path):
        """
        Launches the subprocess and waits for it.
        """
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.RUNNING

        # Command: python main.py --config-path {config_dir} --config-name config
        # We assume main.py is in the current working directory or accessible.
        # Better to find absolute path of main.py relative to this file?
        # Assuming run from root of repo:
        cmd = [
            sys.executable, "main.py", 
            "--config-path", str(config_dir.absolute()), 
            "--config-name", "config",
            f"hydra.run.dir={config_dir.absolute()}",
            "hydra.job.chdir=False"
        ]

        try:
            with open(log_file_path, 'w') as log_file:
                # We can also log the command itself
                log_file.write(f"Executing: {' '.join(cmd)}\n")
                log_file.write(f"Start Time: {datetime.now()}\n")
                log_file.flush()

                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT, # Merge stderr into stdout
                    cwd=os.getcwd(), # Run from current dir
                    text=True
                )
            
            with self._jobs_lock:
                job.pid = process.pid
            
            # Wait for completion
            return_code = process.wait()

            with self._jobs_lock:
                job.end_time = datetime.now()
                if return_code == 0:
                    job.status = JobStatus.COMPLETED
                    with open(log_file_path, 'a') as f:
                        f.write(f"\nJob Completed Successfully at {job.end_time}\n")
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = f"Process exited with code {return_code}"
                    with open(log_file_path, 'a') as f:
                        f.write(f"\nJob Failed with exit code {return_code} at {job.end_time}\n")

        except Exception as e:
            logger.error(f"Failed to launch subprocess for job {job_id}: {e}")
            with self._jobs_lock:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.end_time = datetime.now()
            
            with open(log_file_path, 'a') as f:
                f.write(f"\nSystem Error: {e}\n")

    def stop_job(self, job_id: str) -> bool:
        """
        Stops a running job.
        """
        import psutil
        import signal

        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if not job or job.status != JobStatus.RUNNING or not job.pid:
                return False
        
        try:
            parent = psutil.Process(job.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            
            # Record stop
            with self._jobs_lock:
                 job.status = JobStatus.FAILED
                 job.error_message = "Stopped by user"
                 job.end_time = datetime.now()

            with open(job.log_file_path, 'a') as f:
                 f.write(f"\nSTOPPED BY USER at {datetime.now()}\n")
            
            return True
        except Exception as e:
            logger.error(f"Failed to stop job {job_id}: {e}")
            return False

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
