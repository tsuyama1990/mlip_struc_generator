import pytest
import os
from ase.calculators.emt import EMT
from nnp_gen.core.config import AppConfig
from nnp_gen.pipeline.runner import PipelineRunner
from unittest.mock import patch

def test_integration_full_pipeline_emt(tmp_path):
    """
    Run full pipeline with AlloyGenerator (ASE) and MD (mocked to use EMT).
    This tests:
    1. Real generation (Alloy/ASE)
    2. Real MD loop (run_single_md) but with EMT calculator (fast/available)
    3. Real sampling (Random)
    4. Real database storage
    """
    output_dir = tmp_path / "output"

    config_dict = {
        "output_dir": str(output_dir),
        "seed": 42,
        "system": {
            "type": "alloy",
            "elements": ["Cu", "Ag"], # EMT supports Cu, Ag
            "supercell_size": [2, 2, 2],
            "rattle_std": 0.01,
            "constraints": {"r_cut": 4.0}
        },
        "exploration": {
            "method": "md",
            "temperature": 100,
            "steps": 10,
            "timestep": 1.0
        },
        "sampling": {
            "strategy": "random",
            "n_samples": 2
        }
    }

    config = AppConfig(**config_dict)

    # Patch _get_calculator to return EMT()
    # Note: run_single_md runs in a separate process via ProcessPoolExecutor.
    # Patching across process boundaries is tricky.
    # If n_workers=1, ProcessPoolExecutor might still pickle.
    # But objects must be picklable. EMT is picklable.
    # However, 'mock' objects are not picklable.
    # We can't use MagicMock return value if it's sent to another process.
    # EMT() object is picklable.

    # But wait, run_single_md calls _get_calculator INSIDE the worker process.
    # Patching in the main process does not affect the worker process if it forks/spawns.
    # If we use 'spawn' (default on Mac/Win), it won't work. On Linux (fork), it might.
    # To be safe, we should instruct MDExplorer to use serial execution or mock the runner.
    # But we want to test "Real MD loop".

    # Alternatively, we can force MDExplorer to run serial by hacking `ProcessPoolExecutor`?
    # Or setting n_workers=0?
    # My code: `n_workers = max(1, os.cpu_count() // 2 ...)`

    # If I set `concurrent.futures.ProcessPoolExecutor` to be a synchronous dummy?
    pass

    # Better approach:
    # We can define a wrapper that replaces `_get_calculator` in `nnp_gen.explorers.md_engine`
    # but we need it to persist in worker.

    # If we can't easily patch worker, we can assume test runs on Linux (GitHub Actions usually)
    # where 'fork' copies memory, so patch applies?
    # Let's try.

    with patch("nnp_gen.explorers.md_engine._get_calculator", side_effect=lambda m, d: EMT()):
        runner = PipelineRunner(config)
        # We need to force MDExplorer to use only 1 worker to ensure we don't have weird multiprocessing issues if patch doesn't carry over?
        # But wait, patch affects global namespace.

        # Let's try running. If it fails, we know why.
        runner.run()

    # Check output
    db_path = output_dir / "dataset.db"
    assert db_path.exists()

    import ase.db
    db = ase.db.connect(db_path)
    # Should have samples
    assert len(db) > 0
    row = db.get(id=1)
    assert row.is_sampled is True

    xyz_path = output_dir / "dataset.xyz"
    assert xyz_path.exists()
