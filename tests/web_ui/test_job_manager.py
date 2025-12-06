import time
import pytest
from nnp_gen.web_ui.job_manager import JobManager, JobStatus
from nnp_gen.core.config import AppConfig

def test_job_manager_submission(mocker, minimal_config, tmp_path):
    """Test submitting a job via JobManager."""
    # Mock PipelineRunner
    mock_runner = mocker.patch("nnp_gen.web_ui.job_manager.PipelineRunner")
    mock_instance = mock_runner.return_value
    mock_instance.run.return_value = None

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

    # Verify PipelineRunner called with correct path
    assert mock_runner.call_count == 1
    call_args = mock_runner.call_args
    passed_config = call_args[0][0]
    assert passed_config.output_dir == job_info.output_dir

    # Verify log file exists and contains start message
    with open(job_info.log_file_path, "r") as f:
        content = f.read()
        assert f"Starting Job {job_id}" in content
        assert "Completed Successfully" in content

def test_job_failure_handling(mocker, minimal_config, tmp_path):
    """Test handling of job failures."""
    mock_runner = mocker.patch("nnp_gen.web_ui.job_manager.PipelineRunner")
    mock_instance = mock_runner.return_value
    mock_instance.run.side_effect = RuntimeError("Explosion!")

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
    assert "Explosion!" in job_info.error_message

    # Verify log contains error
    with open(job_info.log_file_path, "r") as f:
        content = f.read()
        assert "Job Execution Failed: Explosion!" in content
