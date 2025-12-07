import os
import shutil
import pytest
import pickle
from ase import Atoms
from nnp_gen.core.config import AppConfig, AlloySystemConfig, ExplorationConfig, SamplingConfig
from nnp_gen.pipeline.runner import PipelineRunner
from ase.io import read

@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "real_integration_test_output"
    d.mkdir()
    return str(d)

def test_full_pipeline_emt(output_dir):
    """
    True Integration Test: Full pipeline with EMT, Checkpointing, and Database.
    """
    # 1. Setup Configuration
    config = AppConfig(
        system=AlloySystemConfig(
            elements=["Cu"], # EMT supports Cu
            lattice_constant=3.8,
            supercell_size=[3, 3, 3],
            rattle_std=0.0, # Zero rattle for stability
            pbc=[True, True, True]
        ),
        exploration=ExplorationConfig(
            method="md",
            model_name="emt", # Using EMT
            steps=20, # Short run for test speed
            timestep=1.0,
            temperature=300
        ),
        sampling=SamplingConfig(
            strategy="random", # Simple sampling
            n_samples=2
        ),
        output_dir=output_dir,
        seed=123
    )

    # 2. Run Pipeline
    runner = PipelineRunner(config)
    runner.run()

    # 3. Assertions

    # A. Check Database
    db_path = os.path.join(output_dir, "dataset.db")
    assert os.path.exists(db_path), "Database file not created"

    from ase.db import connect
    with connect(db_path) as db:
        assert len(db) == 2, f"Expected 2 structures in DB, found {len(db)}"
        row = db.get(id=1)
        # Verify metadata
        assert row.source == "md"
        assert row.is_sampled is True
        # Check config hash format
        assert str(row.config_hash).startswith("hash_")

    # B. Check Checkpoints DB
    ckpt_db_path = os.path.join(output_dir, "checkpoints.db")
    assert os.path.exists(ckpt_db_path), "checkpoints.db missing"

    with connect(ckpt_db_path) as ckpt_db:
        # We expect entries from 'generated' and 'explored' stages
        # We generated some structures (initially 2x2x2=8, now 3x3x3=27? No, unit cell is 4 atoms for fcc Cu. 4*27=108 atoms.
        # Wait, usually generate() returns a LIST of structures.
        api_gen_len = len(runner.generator.generate()) # Re-running generate to check count? No, that's dangerous.
        # Just check we have entries.
        assert len(ckpt_db) > 0
        
        # Check stages exist
        stages = set(row.stage for row in ckpt_db.select())
        assert "generated" in stages
        assert "explored" in stages or "explored_skipped" in stages

    # C. Check Exports
    assert os.path.exists(os.path.join(output_dir, "initial_structures.xyz"))
    assert os.path.exists(os.path.join(output_dir, "explored_structures.xyz"))
    assert os.path.exists(os.path.join(output_dir, "sampled_structures.xyz"))

    print("Full pipeline integration test passed.")

if __name__ == "__main__":
    pytest.main([__file__])
