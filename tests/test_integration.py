import os
import shutil
import pytest
from ase import Atoms
from nnp_gen.core.config import AppConfig, AlloySystemConfig, ExplorationConfig, SamplingConfig
from nnp_gen.pipeline.runner import PipelineRunner
from ase.io import read

@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "integration_test_output"
    d.mkdir()
    return str(d)

def test_pipeline_integration_alloy_emt(output_dir):
    """
    Test the full pipeline with Alloy generation and EMT MD.
    Using Cu and Ag which are supported by EMT.
    """
    config = AppConfig(
        system=AlloySystemConfig(
            elements=["Cu", "Ag"],
            lattice_constant=4.2, # Safe for Ag
            supercell_size=[3, 3, 3],
            rattle_std=0.0
        ),
        exploration=ExplorationConfig(
            method="md",
            model_name="emt",
            steps=50, # Short run
            timestep=1.0,
            temperature=300,
            snapshot_interval=5 # Ensure we capture enough frames
        ),
        sampling=SamplingConfig(
            strategy="random", # Use random to avoid descriptor dependency issues if dscribe missing
            n_samples=5
        ),
        output_dir=output_dir,
        seed=42
    )

    runner = PipelineRunner(config)
    runner.run()

    # Verify outputs
    assert os.path.exists(os.path.join(output_dir, "dataset.db"))
    assert os.path.exists(os.path.join(output_dir, "initial_structures.xyz"))
    assert os.path.exists(os.path.join(output_dir, "explored_structures.xyz"))
    assert os.path.exists(os.path.join(output_dir, "sampled_structures.xyz"))

    # Verify contents
    sampled = read(os.path.join(output_dir, "sampled_structures.xyz"), index=":")
    assert len(sampled) >= 1

    # Check if DB has entries
    from ase.db import connect
    with connect(os.path.join(output_dir, "dataset.db")) as db:
        assert len(db) == 5
        row = db.get(id=1)
        # Metadata is stored as key-value pairs, which become attributes of the row
        assert row.is_sampled is True
        assert str(row.config_hash).startswith("hash_")

def test_pipeline_integration_molecule_emt(output_dir):
    """
    Test pipeline with Molecule generation and EMT MD.
    Requires RDKit.
    """
    pytest.importorskip("rdkit")

    config = AppConfig(
        system=dict(
            type="molecule",
            elements=["H", "O"], # Dummy, inferred from smiles usually
            smiles="O", # Water
            num_conformers=5,
            rattle_std=0.01,
            pbc=[False, False, False]
        ),
        exploration=ExplorationConfig(
            method="md",
            model_name="emt",
            steps=20,
            temperature=100
        ),
        sampling=SamplingConfig(
            strategy="manual", # Keep all
            n_samples=5
        ),
        output_dir=output_dir + "_mol",
        seed=42
    )

    runner = PipelineRunner(config)
    runner.run()

    assert os.path.exists(os.path.join(output_dir + "_mol", "sampled_structures.xyz"))
    sampled = read(os.path.join(output_dir + "_mol", "sampled_structures.xyz"), index=":")
    assert len(sampled) >= 1 # Might be 5 or less if some fail
