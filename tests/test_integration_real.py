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
            lattice_constant=3.61,
            supercell_size=[2, 2, 2],
            rattle_std=0.02, # Valid value
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

    # B. Check Checkpoints (Pickle files)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    assert os.path.exists(ckpt_dir), "Checkpoint directory not created"

    # Generated
    gen_ckpt = os.path.join(ckpt_dir, "checkpoint_generated.pkl")
    assert os.path.exists(gen_ckpt), "checkpoint_generated.pkl missing"
    with open(gen_ckpt, 'rb') as f:
        gen_data = pickle.load(f)
        assert isinstance(gen_data, list)
        assert len(gen_data) > 0
        assert isinstance(gen_data[0], Atoms)

    # Explored
    exp_ckpt = os.path.join(ckpt_dir, "checkpoint_explored.pkl")
    assert os.path.exists(exp_ckpt), "checkpoint_explored.pkl missing"
    with open(exp_ckpt, 'rb') as f:
        exp_data = pickle.load(f)
        assert isinstance(exp_data, list)
        # Length depends on how many survived MD, but for 20 steps likely all
        assert len(exp_data) > 0

    # Sampled
    sam_ckpt = os.path.join(ckpt_dir, "checkpoint_sampled.pkl")
    assert os.path.exists(sam_ckpt), "checkpoint_sampled.pkl missing"
    with open(sam_ckpt, 'rb') as f:
        sam_data = pickle.load(f)
        assert len(sam_data) == 2

    # C. Check Exports
    assert os.path.exists(os.path.join(output_dir, "initial_structures.xyz"))
    assert os.path.exists(os.path.join(output_dir, "explored_structures.xyz"))
    assert os.path.exists(os.path.join(output_dir, "sampled_structures.xyz"))

    print("Full pipeline integration test passed.")

if __name__ == "__main__":
    pytest.main([__file__])
