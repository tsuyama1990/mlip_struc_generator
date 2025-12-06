import pytest
import numpy as np
from ase import Atoms
from nnp_gen.core.physics import apply_rattle, apply_volumetric_strain, set_initial_magmoms, ensure_supercell_size

def test_apply_rattle():
    atoms = Atoms('H2', positions=[[0,0,0], [1,0,0]])
    orig_pos = atoms.positions.copy()
    atoms = apply_rattle(atoms, std=0.1)
    assert not np.allclose(atoms.positions, orig_pos)
    assert np.allclose(atoms.positions, orig_pos, atol=0.5)

def test_apply_volumetric_strain():
    atoms = Atoms('H', positions=[[0.5, 0.5, 0.5]], cell=[1,1,1], pbc=True)
    # Range [1.1, 1.1] forces 1.1 scale
    atoms = apply_volumetric_strain(atoms, [1.1, 1.1])
    assert np.allclose(atoms.cell.lengths(), [1.1, 1.1, 1.1])
    # Check positions scaled
    assert np.allclose(atoms.positions, [[0.55, 0.55, 0.55]])

def test_set_initial_magmoms():
    atoms = Atoms('FeNi', positions=[[0,0,0], [1,0,0]])
    map = {"Fe": 2.2, "Ni": 0.6}
    atoms = set_initial_magmoms(atoms, map)
    mags = atoms.get_initial_magnetic_moments()
    assert mags[0] == 2.2
    assert mags[1] == 0.6

def test_ensure_supercell_size():
    # cell 3x3x3. r_cut=5.0, factor=2.0 -> min=10.0. Need repeat 4 (3*4=12 > 10)
    atoms = Atoms('H', cell=[3,3,3], pbc=True)
    supercell = ensure_supercell_size(atoms, r_cut=5.0, factor=2.0)
    assert np.all(supercell.cell.lengths() >= 10.0)
    # 3 * 4 = 12
    assert np.allclose(supercell.cell.lengths(), [12, 12, 12])
