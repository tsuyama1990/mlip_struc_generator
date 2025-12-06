import pytest
import os
from ase import Atoms
from ase.io import write
from nnp_gen.core.config import UserFileSystemConfig
from nnp_gen.generators.file_loader import FileGenerator
from nnp_gen.core.exceptions import GenerationError

def test_file_generator_clone_single(tmp_path):
    """Test loading a single structure and repeating it."""
    # Create a dummy structure file
    filepath = tmp_path / "single.xyz"
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
    write(filepath, atoms)

    config = UserFileSystemConfig(
        type="user_file",
        path=str(filepath),
        repeat=5,
        elements=["H"]
    )

    generator = FileGenerator(config)
    structures = generator._generate_impl()

    assert len(structures) == 5
    for s in structures:
        assert len(s) == 2
        assert str(s.symbols) == "H2"

def test_file_generator_trajectory(tmp_path):
    """Test loading a trajectory (multiple frames) and repeating it."""
    filepath = tmp_path / "traj.xyz"
    frames = [
        Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]]),
        Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]]),
        Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.4]])
    ]
    write(filepath, frames)

    # Repeat=2 means we should get 3 * 2 = 6 structures
    config = UserFileSystemConfig(
        type="user_file",
        path=str(filepath),
        repeat=2,
        elements=["H"]
    )

    generator = FileGenerator(config)
    structures = generator._generate_impl()

    assert len(structures) == 6
    # Check order: Frame1, Frame2, Frame3, Frame1, Frame2, Frame3
    assert structures[0].positions[1, 2] == 1.0
    assert structures[1].positions[1, 2] == 1.2
    assert structures[2].positions[1, 2] == 1.4
    assert structures[3].positions[1, 2] == 1.0

def test_file_generator_missing_file():
    """Test error handling for missing file."""
    config = UserFileSystemConfig(
        type="user_file",
        path="non_existent_file.xyz",
        repeat=1,
        elements=["H"]
    )

    generator = FileGenerator(config)
    with pytest.raises(GenerationError, match="Input file not found"):
        generator._generate_impl()

def test_file_generator_validation_repeat():
    """Test config validation for repeat."""
    with pytest.raises(ValueError):
        UserFileSystemConfig(
            type="user_file",
            path="dummy.xyz",
            repeat=0,
            elements=["H"]
        )
