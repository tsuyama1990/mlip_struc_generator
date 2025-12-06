import os
import shutil
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary directory for outputs."""
    path = tmp_path / "output"
    path.mkdir()
    return str(path)

@pytest.fixture
def mock_pipeline_runner(mocker):
    """Mock PipelineRunner to avoid running actual heavy tasks."""
    runner = mocker.patch("nnp_gen.web_ui.job_manager.PipelineRunner")
    runner_instance = runner.return_value
    runner_instance.run.return_value = None
    return runner

@pytest.fixture
def minimal_config():
    """Return a minimal valid AppConfig dictionary for testing."""
    return {
        "system": {
            "type": "alloy",
            "elements": ["Fe"],
            "constraints": {"max_atoms": 100}
        },
        "exploration": {
            "method": "md",
            "model_name": "emt",
            "temperature": 300,
            "steps": 10,
            "timestep": 1.0
        },
        "sampling": {
            "strategy": "random",
            "n_samples": 5
        },
        "output_dir": "test_output"
    }
