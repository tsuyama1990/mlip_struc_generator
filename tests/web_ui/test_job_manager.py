import time
import pytest
from nnp_gen.web_ui.job_manager import JobManager, JobStatus
from nnp_gen.core.config import AppConfig

def test_job_manager_submission(mocker, minimal_config, tmp_path):
    """Test submitting a job via JobManager."""
    # Mock subprocess.Popen instead of PipelineRunner, because JobManager launches a subprocess.
    mock_popen = mocker.patch("subprocess.Popen")
    mock_process = mock_popen.return_value
    mock_process.pid = 12345
    mock_process.wait.return_value = 0 # Success

    # Init JobManager
    # Reset singleton for testing (hacky but needed since it's a singleton)
    JobManager._instance = None
    manager = JobManager()

    # Configure logging path to tmp
    manager.logs_root = tmp_path / "logs"
    manager.logs_root.mkdir()

    # Create Config object
    config = AppConfig(**minimal_config)

    # Submit
    job_id = manager.submit_job(config)

    assert job_id is not None
    job_info = manager.get_job(job_id)
    assert job_info.status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED]

    # Wait for completion (ThreadPoolExecutor is fast for mock)
    # We poll a bit
    for _ in range(10):
        if job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(0.1)

    assert job_info.status == JobStatus.COMPLETED
    assert job_info.job_id == job_id

    # Verify effective output dir
    assert job_id in job_info.output_dir

    # Verify subprocess launched
    assert mock_popen.call_count == 1
    call_args = mock_popen.call_args
    cmd = call_args[0][0]
    assert "main.py" in cmd
    assert "--config-path" in cmd

    # Verify log file exists and contains start message
    # (Note: subprocess stdout redirection won't happen since we mocked Popen)
    # But JobManager writes "Executing..." before calling Popen
    with open(job_info.log_file_path, "r") as f:
        content = f.read()
        assert f"Executing:" in content
        assert "Job Completed Successfully" in content

def test_job_failure_handling(mocker, minimal_config, tmp_path):
    """Test handling of job failures."""
    # Mock subprocess.Popen failure
    mock_popen = mocker.patch("subprocess.Popen")
    mock_process = mock_popen.return_value
    mock_process.pid = 12345
    mock_process.wait.return_value = 1 # Error code

    JobManager._instance = None
    manager = JobManager()
    manager.logs_root = tmp_path / "logs"
    manager.logs_root.mkdir()

    config = AppConfig(**minimal_config)
    job_id = manager.submit_job(config)

    job_info = manager.get_job(job_id)

    for _ in range(10):
        if job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(0.1)

    assert job_info.status == JobStatus.FAILED
    assert "Process exited with code 1" in job_info.error_message

    # Verify log contains error
    with open(job_info.log_file_path, "r") as f:
        content = f.read()
        assert "Job Failed with exit code 1" in content
