import pytest
from unittest.mock import MagicMock
from nnp_gen.core.config import AppConfig
from nnp_gen.pipeline.runner import PipelineRunner
from ase import Atoms

def test_pipeline_run(mocker, tmp_path):
    # Setup config
    conf_dict = {
        "output_dir": str(tmp_path / "test_out"),
        "seed": 42,
        "system": {
            "type": "alloy",
            "elements": ["Cu"],
            "constraints": {},
            "pbc": [True, True, True],
            "rattle_std": 0.0,
            "vol_scale_range": [1.0, 1.0],
            "supercell_size": [1,1,1]
        },
        "exploration": {
            "method": "md",
            "temperature": 100,
            "steps": 10,
            "timestep": 1.0
        },
        "sampling": {
            "strategy": "random",
            "n_samples": 5,
            "descriptor_type": "soap",
            "min_distance": 1.0
        }
    }
    config = AppConfig(**conf_dict)

    # Setup mocks
    # Generator
    mock_gen_factory = mocker.patch("nnp_gen.pipeline.runner.GeneratorFactory")
    mock_gen = mocker.Mock()
    mock_gen_factory.get_generator.return_value = mock_gen
    mock_gen.generate.return_value = [Atoms('Cu', pbc=True)]

    # MD
    mock_md_cls = mocker.patch("nnp_gen.pipeline.runner.MDExplorer")
    mock_md = mock_md_cls.return_value
    mock_md.explore.return_value = [Atoms('Cu', pbc=True)]

    # DB
    mock_db_cls = mocker.patch("nnp_gen.pipeline.runner.ASEDbStorage")
    mock_db = mock_db_cls.return_value
    mock_db.bulk_save.return_value = [1]

    # Don't mock os.makedirs, use real fs with tmp_path
    # mocker.patch("os.makedirs")

    runner = PipelineRunner(config)
    runner.run()

    # Verify calls
    mock_gen.generate.assert_called_once()
    mock_md.explore.assert_called_once()
    assert mock_db.bulk_save.call_count >= 1
