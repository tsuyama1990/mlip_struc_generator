import numpy as np
import pytest
from ase import Atoms
from nnp_gen.core.physics import ensure_supercell_size
from nnp_gen.generators.ionic import IonicGenerator
from nnp_gen.core.config import IonicSystemConfig, PhysicsConstraints

def test_ensure_supercell_size_skewed():
    """
    Test that ensure_supercell_size correctly handles a skewed cell
    where vector lengths are large enough, but perpendicular width is not.
    """
    # Create a triclinic cell
    # a = [10, 0, 0]
    # b = [0, 10, 0]
    # c = [9, 0, 2] -> Length = sqrt(81+4) = 9.21
    # Height of c above ab plane is 2.

    cell = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [9.0, 0.0, 2.0]
    ])
    atoms = Atoms('H', positions=[[0,0,0]], cell=cell, pbc=True)

    r_cut = 3.0

    # Based on calculation:
    # Width_c = 2.0 < 3.0 -> Need repeat c >= 2.
    # Width_a = 2.17 < 3.0 -> Need repeat a >= 2.
    # Width_b = 10.0 > 3.0 -> Repeat b = 1.
    # Expected repeat: [2, 1, 2] -> 4 atoms.

    expanded = ensure_supercell_size(atoms, r_cut)

    assert len(expanded) == 4, f"Expected 4 atoms (2x1x2 expansion), got {len(expanded)}"


def test_ionic_generator_lattice_scaling():
    """
    Test that IonicGenerator produces significantly different lattice constants
    for materials with different ionic radii (NaCl vs MgO).
    """
    # Set r_cut small to avoid supercell expansion confusing the volume comparison
    constraints = PhysicsConstraints(r_cut=2.0, min_distance=1.0)

    # NaCl
    config_nacl = IonicSystemConfig(
        elements=["Na", "Cl"],
        oxidation_states={"Na": 1, "Cl": -1},
        num_structures=1,
        constraints=constraints,
        supercell_size=[1,1,1]
    )
    gen_nacl = IonicGenerator(config_nacl)

    # MgO
    config_mgo = IonicSystemConfig(
        elements=["Mg", "O"],
        oxidation_states={"Mg": 2, "O": -2},
        num_structures=1,
        constraints=constraints,
        supercell_size=[1,1,1]
    )
    gen_mgo = IonicGenerator(config_mgo)

    try:
        structs_nacl = gen_nacl.generate()
        structs_mgo = gen_mgo.generate()

        if structs_nacl and structs_mgo:
            vol_nacl = structs_nacl[0].get_volume()
            vol_mgo = structs_mgo[0].get_volume()

            # MgO should be significantly smaller than NaCl
            # NaCl a ~ 5.6 -> Vol ~ 176
            # MgO a ~ 4.2 -> Vol ~ 74
            assert vol_mgo < vol_nacl
            assert abs(vol_mgo - vol_nacl) > 50.0

    except ImportError:
        pytest.skip("pymatgen or other dependencies missing")
    except Exception as e:
        pytest.fail(f"Generation failed: {e}")
