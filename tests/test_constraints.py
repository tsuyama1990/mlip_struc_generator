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
        # Return copies to avoid modifying the input list in place if they are re-used
        return [atoms.copy() for atoms in self.structures_to_yield]

# --- 1. Static Constraint Test ---

def test_static_constraint_max_atoms_removed():
    """
    Test that config validation NO LONGER catches supercell settings based on volume factor.
    This validation was removed as it was inaccurate.
    Runtime validation in BaseGenerator handles the actual atom count check.
    """
    config = AlloySystemConfig(
        type="alloy",
        elements=["Fe"],
        supercell_size=[5, 5, 5],
        constraints=PhysicsConstraints(max_atoms=100) # 125 > 100, would fail old check
    )
    assert config.supercell_size == [5, 5, 5]

# --- 2. Dynamic Filter Test ---

def test_dynamic_filter_sanity_check():
    """
    Test that structures violating constraints are filtered out.
    """
    constraints = PhysicsConstraints(
        max_atoms=50,
        min_distance=1.0,
        min_density=0.0,
        # Disable supercell expansion for this test
        min_cell_length_factor=0.0,
        r_cut=1.0
    )
    config = AlloySystemConfig(
        type="alloy",
        elements=["Cu"],
        constraints=constraints,
        # Disable physics augmentation
        rattle_std=0.0,
        vol_scale_range=[1.0, 1.0],
        pbc=[True, True, True]
    )

    struct_A = Atoms('Cu', positions=[[0, 0, 0]], cell=[3, 3, 3])
    struct_B = Atoms('Cu60', positions=[[0,0,0]]*60, cell=[10,10,10])
    struct_C = Atoms('Cu2', positions=[[0,0,0], [0,0,0.1]], cell=[3,3,3])
    struct_D = Atoms('Cu2', positions=[[0,0,0], [0,0,1.1]], cell=[3,3,3])

    gen = MockGenerator(config, [struct_A, struct_B, struct_C, struct_D])

    valid_structures = gen.generate()

    # Assertions
    assert len(valid_structures) == 2
    # Check simple properties instead of equality because generate might do copy or minor float adjustments
    assert len(valid_structures[0]) == 1 # A
    assert len(valid_structures[1]) == 2 # D
    assert str(valid_structures[0].symbols) == "Cu"
    assert str(valid_structures[1].symbols) == "Cu2"

def test_dynamic_filter_density():
    """
    Test min_density filtering.
    """
    constraints = PhysicsConstraints(
        min_density=1.0,
        min_cell_length_factor=0.0,
        r_cut=1.0
    )
    config = AlloySystemConfig(
        type="alloy",
        elements=["Cu"],
        constraints=constraints,
        rattle_std=0.0,
        vol_scale_range=[1.0, 1.0]
    )

    struct_low = Atoms('Cu', positions=[[5,5,5]], cell=[10,10,10])
    struct_high = Atoms('Cu', positions=[[0,0,0]], cell=[2,2,2])

    gen = MockGenerator(config, [struct_low, struct_high])
    valid = gen.generate()

    assert len(valid) == 1
    assert len(valid[0]) == 1
    assert np.allclose(valid[0].cell.lengths(), [2,2,2])

# --- 3. Boundary Condition Test ---

def test_pbc_enforcement():
    """
    Verify PBC settings are applied.
    """
    config_alloy = AlloySystemConfig(
        type="alloy",
        elements=["Cu"],
        pbc=[True, True, True],
        rattle_std=0.0,
        vol_scale_range=[1.0, 1.0],
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=1.0)
    )
    config_mol = MoleculeSystemConfig(
        type="molecule",
        elements=["H", "H"],
        smiles="HH",
        pbc=[False, False, False],
        rattle_std=0.0,
        vol_scale_range=[1.0, 1.0],
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=1.0)
    )

    atoms_wrong = Atoms('Cu', pbc=[False, False, False])

    gen_alloy = MockGenerator(config_alloy, [atoms_wrong.copy()])
    res_alloy = gen_alloy.generate()
    assert np.all(res_alloy[0].pbc == [True, True, True])

    gen_mol = MockGenerator(config_mol, [atoms_wrong.copy()])
    res_mol = gen_mol.generate()
    assert np.all(res_mol[0].pbc == [False, False, False])
