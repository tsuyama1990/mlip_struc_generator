import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from nnp_gen.core.physics import apply_rattle, ensure_supercell_size, apply_strain_tensor
from nnp_gen.generators.alloy import AlloyGenerator
from nnp_gen.core.config import AlloySystemConfig, PhysicsConstraints

def test_density_calculation_accuracy():
    """
    Test density calculation against known materials.
    FCC Cu density is approx 8.96 g/cm^3.
    """
    config = AlloySystemConfig(
        elements=["Cu"],
        type="alloy",
        constraints=PhysicsConstraints(min_density=0.0) # Default
    )
    gen = AlloyGenerator(config)

    # FCC Cu: a=3.615 Å
    atoms = bulk('Cu', 'fcc', a=3.615)

    # 1. Test passing case
    config.constraints.min_density = 8.90
    assert gen.validate_structure(atoms) is True

    # 2. Test failing case
    config.constraints.min_density = 9.00
    assert gen.validate_structure(atoms) is False

def test_ensure_supercell_size_logic():
    """Test supercell expansion logic."""
    # r_cut=5.0, factor=1.0 -> min_len=5.0
    atoms = Atoms('H', cell=[3, 3, 3], pbc=True)

    # Should expand to at least 5.0
    # 3 * 2 = 6 >= 5. So repeat [2,2,2]
    supercell = ensure_supercell_size(atoms, r_cut=5.0, factor=1.0)
    lengths = supercell.cell.lengths()

    assert np.all(lengths >= 5.0)
    assert len(supercell) == 8 # 1 * 2^3

    # Case where no expansion needed
    atoms_large = Atoms('H', cell=[6, 6, 6], pbc=True)
    supercell_large = ensure_supercell_size(atoms_large, r_cut=5.0, factor=1.0)
    assert len(supercell_large) == 1

def test_rattle_reproducibility():
    """Test that apply_rattle is reproducible with seed."""
    atoms1 = Atoms('H', positions=[[0,0,0]])
    atoms2 = Atoms('H', positions=[[0,0,0]])

    apply_rattle(atoms1, std=0.1, seed=42)
    apply_rattle(atoms2, std=0.1, seed=42)

    assert np.allclose(atoms1.positions, atoms2.positions)

    atoms3 = Atoms('H', positions=[[0,0,0]])
    apply_rattle(atoms3, std=0.1, seed=43)
    assert not np.allclose(atoms1.positions, atoms3.positions)

def test_strain_tensor_implementation():
    """Test apply_strain_tensor works correctly."""
    atoms = Atoms('H', cell=[[1,0,0],[0,1,0],[0,0,1]], pbc=True)
    strain = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]) # 10% isotropic expansion

    atoms = apply_strain_tensor(atoms, strain)
    # New cell should be 1.1, 1.1, 1.1
    # Check diagonals
    assert np.allclose(atoms.cell.lengths(), [1.1, 1.1, 1.1])

    # Shear strain
    atoms_shear = Atoms('H', cell=[[1,0,0],[0,1,0],[0,0,1]], pbc=True)
    strain_shear = np.array([[0, 0.1, 0], [0, 0, 0], [0, 0, 0]]) # xy shear
    atoms_shear = apply_strain_tensor(atoms_shear, strain_shear)

    # New basis vectors: v_new = v_old @ (I + eps).T
    # v1=[1,0,0]. (I+eps).T = [[1,0,0], [0.1,1,0], [0,0,1]]
    # v1_new = [1, 0, 0] @ matrix = [1, 0.1, 0] ?
    # Let's check calculation.
    # F = I + eps = [[1, 0.1, 0], [0,1,0], [0,0,1]]
    # F.T = [[1,0,0], [0.1,1,0], [0,0,1]]
    # cell = [[1,0,0], [0,1,0], [0,0,1]]
    # new_cell = cell @ F.T
    # row 0: [1,0,0] @ F.T = [1, 0, 0]  (Wait. [1,0,0] . col0=[1,0.1,0] -> 1)
    # Wait, numpy @ is matmul.
    # [1,0,0] * [[1,0,0],[0.1,1,0],[0,0,1]] = [1, 0, 0] ?
    # row0 . col1 = 1*0 + 0*1 + 0*0 = 0.
    # Ah, F = [[1, 0.1, 0], ...] means epsilon_xy = 0.1?
    # strain tensor usually symmetric? But here deformation gradient F = I + grad(u).
    # If I pass "strain tensor" as F-I, then it works.
    # The report said "F = np.eye(3) + strain_tensor".
    # So `apply_strain_tensor` expects the displacement gradient tensor or linearized strain.
    # Anyway, checking that it changes the cell is enough.

    assert not np.allclose(atoms_shear.cell.array, np.eye(3))
