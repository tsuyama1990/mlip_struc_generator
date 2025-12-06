import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.build import bulk
from ase import units
from ase.md.verlet import VelocityVerlet
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
    # Use supercell to avoid false positive vacuum detection in StructureValidator
    atoms = bulk('Cu', 'fcc', a=3.615).repeat((3, 3, 3))

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

def test_ensure_supercell_triclinic():
    """
    Test supercell expansion for triclinic cell where vector length > r_cut
    but perpendicular width < r_cut.
    """
    # Create a highly skewed cell
    # a = [10, 0, 0] (|a|=10)
    # b = [5, 8.66, 0] (|b|=10)
    # c = [9, 0, 1] (|c| ~ 9.05)
    # The height in z-direction is 1.0, which is < r_cut=5.0

    cell = [[10.0, 0.0, 0.0],
            [5.0, 8.66, 0.0],
            [9.0, 0.0, 1.0]]

    atoms = Atoms('H', cell=cell, pbc=True)
    r_cut = 5.0

    # Check initial geometry
    # Lengths are large enough
    assert np.linalg.norm(cell[0]) >= r_cut
    assert np.linalg.norm(cell[1]) >= r_cut
    assert np.linalg.norm(cell[2]) >= r_cut

    # But height perpendicular to ab plane is 1.0
    vol = atoms.get_volume() # 10 * 8.66 * 1 = 86.6
    # Area of ab face = |a x b| = 10 * 8.66 = 86.6
    # h_c = vol / area = 1.0

    supercell = ensure_supercell_size(atoms, r_cut=r_cut)

    # We expect expansion in c direction
    # r_cut=5, h=1 -> need 5x expansion

    sc_cell = supercell.cell.array
    # Calculate new heights
    sc_vol = abs(supercell.get_volume())

    # New height in c direction (approx)
    # The supercell matrix should be repeat * cell
    # If repeat is [1, 1, 5], then new c vector is 5*c = [45, 0, 5]. Height is 5.

    # Check that we have enough atoms
    # original 1 atom. if repeat is [1,1,5], we expect 5 atoms.
    assert len(supercell) >= 5

    # Verify strict height condition
    v = sc_cell
    areas = [
        np.linalg.norm(np.cross(v[1], v[2])),
        np.linalg.norm(np.cross(v[0], v[2])),
        np.linalg.norm(np.cross(v[0], v[1]))
    ]
    heights = [sc_vol / a for a in areas]

    assert np.all(np.array(heights) >= r_cut - 1e-4)

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

    assert not np.allclose(atoms_shear.cell.array, np.eye(3))

def test_nve_energy_conservation():
    """
    Test energy conservation in NVE ensemble.
    Strict threshold: < 0.1% drift over 100 steps.
    """
    # 1. Setup a simple system (Cu FCC)
    # Using a 2x2x2 supercell to have enough atoms for statistics but keep it fast
    atoms = bulk('Cu', 'fcc', a=3.61) * (3, 3, 3)
    atoms.calc = EMT()

    # 2. Initial Condition: Perturb positions slightly to have potential energy
    # We don't want to start at perfect equilibrium (0 force)
    atoms.rattle(stdev=0.01, seed=42)

    # 3. NVE Run (Velocity Verlet)
    # Timestep 1.0 fs
    dyn = VelocityVerlet(atoms, 1.0 * units.fs)

    energies = []

    def log_energy():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        energies.append(etot)

    dyn.attach(log_energy, interval=1)

    # Run for 100 steps
    dyn.run(100)

    # 4. Analysis
    energies = np.array(energies)
    initial_energy = energies[0]
    final_energy = energies[-1]

    # Drift calculation
    # Drift = (E_final - E_initial) / |E_initial|
    drift = (final_energy - initial_energy) / abs(initial_energy)

    print(f"Initial Energy: {initial_energy:.4f} eV")
    print(f"Final Energy: {final_energy:.4f} eV")
    print(f"Energy drift: {drift:.6%}")

    # Assert drift is less than 0.1%
    assert abs(drift) < 0.001, f"Energy drift too high: {drift:.4%} (Threshold: 0.1%)"

if __name__ == "__main__":
    pytest.main([__file__])
