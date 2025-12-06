import pytest
import numpy as np
from ase import Atoms
from pydantic import ValidationError
from nnp_gen.core.config import AlloySystemConfig, MoleculeSystemConfig, PhysicsConstraints
from nnp_gen.core.interfaces import BaseGenerator

# --- Test Helpers ---

class MockGenerator(BaseGenerator):
    def __init__(self, config, structures_to_yield):
        super().__init__(config)
        self.structures_to_yield = structures_to_yield

    def _generate_impl(self):
        return self.structures_to_yield

# --- 1. Static Constraint Test ---

def test_static_constraint_max_atoms():
    """
    Test that config validation catches supercell settings that would exceed max_atoms.
    """
    # Case 1: Valid
    # max_atoms = 200, supercell = 4x4x4 (64x)
    # If we assume 1 atom/cell, 64 < 200. OK.
    config = AlloySystemConfig(
        type="alloy",
        elements=["Fe"],
        supercell_size=[4, 4, 4],
        constraints=PhysicsConstraints(max_atoms=200)
    )
    assert config.supercell_size == [4, 4, 4]

    # Case 2: Invalid
    # max_atoms = 100, supercell = 5x5x5 (125x)
    # 125 > 100 -> Error
    with pytest.raises(ValidationError) as excinfo:
        AlloySystemConfig(
            type="alloy",
            elements=["Fe"],
            supercell_size=[5, 5, 5],
            constraints=PhysicsConstraints(max_atoms=100)
        )
    assert "exceeding max_atoms" in str(excinfo.value)

# --- 2. Dynamic Filter Test ---

def test_dynamic_filter_sanity_check():
    """
    Test that structures violating constraints are filtered out.
    """
    constraints = PhysicsConstraints(
        max_atoms=50,
        min_distance=1.0,
        min_density=0.0 # Ignore density for this test to focus on others
    )
    config = AlloySystemConfig(
        type="alloy",
        elements=["Cu"],
        constraints=constraints
    )

    # Create test structures
    # A. Valid: 1 atom
    struct_A = Atoms('Cu', positions=[[0, 0, 0]], cell=[3, 3, 3])

    # B. Too many atoms: 60 atoms (just repeating dummy)
    struct_B = Atoms('Cu60', positions=[[0,0,0]]*60, cell=[10,10,10])

    # C. Overlapping: distance 0.1 < 1.0
    struct_C = Atoms('Cu2', positions=[[0,0,0], [0,0,0.1]], cell=[3,3,3])

    # D. Valid but close to limit (distance 1.1 > 1.0)
    struct_D = Atoms('Cu2', positions=[[0,0,0], [0,0,1.1]], cell=[3,3,3])

    gen = MockGenerator(config, [struct_A, struct_B, struct_C, struct_D])

    valid_structures = gen.generate()

    # Assertions
    assert len(valid_structures) == 2
    assert valid_structures[0] == struct_A
    assert valid_structures[1] == struct_D

def test_dynamic_filter_density():
    """
    Test min_density filtering.
    """
    # Density of Cu is ~8.96 g/cm3.
    # 1 atom of Cu in 10x10x10 A^3 box = mass 63.5 / vol 1000 = 0.0635 amu/A3
    # 0.0635 * 1.66 = 0.1 g/cm3. Very low.

    constraints = PhysicsConstraints(min_density=1.0) # Require at least 1 g/cm3
    config = AlloySystemConfig(type="alloy", elements=["Cu"], constraints=constraints)

    # Low density
    struct_low = Atoms('Cu', positions=[[5,5,5]], cell=[10,10,10]) # vol 1000, dens ~0.1

    # High density (small cell)
    struct_high = Atoms('Cu', positions=[[0,0,0]], cell=[2,2,2]) # vol 8, dens ~ 13 g/cm3

    gen = MockGenerator(config, [struct_low, struct_high])
    valid = gen.generate()

    assert len(valid) == 1
    assert valid[0] == struct_high

# --- 3. Boundary Condition Test ---

def test_pbc_enforcement():
    """
    Verify PBC settings are applied.
    """
    # Crystal -> PBC True
    config_alloy = AlloySystemConfig(
        type="alloy",
        elements=["Cu"],
        pbc=[True, True, True]
    )
    # Molecule -> PBC False
    config_mol = MoleculeSystemConfig(
        type="molecule",
        elements=["H", "H"],
        smiles="HH",
        pbc=[False, False, False]
    )

    # Generator might return atoms with wrong PBC, should be fixed
    atoms_wrong = Atoms('Cu', pbc=[False, False, False])

    gen_alloy = MockGenerator(config_alloy, [atoms_wrong.copy()])
    res_alloy = gen_alloy.generate()
    assert np.all(res_alloy[0].pbc == [True, True, True])

    gen_mol = MockGenerator(config_mol, [atoms_wrong.copy()])
    res_mol = gen_mol.generate()
    assert np.all(res_mol[0].pbc == [False, False, False])
