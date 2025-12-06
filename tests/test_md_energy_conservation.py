import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.build import bulk
from ase import units
from ase.md.langevin import Langevin
from nnp_gen.core.physics import ensure_supercell_size

def test_md_energy_conservation_nve():
    """
    Test energy conservation in NVE ensemble.
    Although we use Langevin (NVT) in production, checking NVE conservation
    with the same calculator ensures the physics and timestep are reasonable.
    """
    # 1. Setup a simple system
    atoms = bulk('Cu', 'fcc', a=3.6) * (2, 2, 2)
    atoms.calc = EMT()

    # 2. Equilibration (NVT) to set temperature
    temp = 300
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=temp, friction=0.002)
    dyn.run(100) # Short equilibration

    # 3. NVE Run (Velocity Verlet)
    # We use VelocityVerlet for NVE. Note: ASE's standard MD is often NVE if no thermostat is attached,
    # but `VelocityVerlet` is the explicit class.
    from ase.md.verlet import VelocityVerlet

    dyn_nve = VelocityVerlet(atoms, 1.0 * units.fs)

    energies = []

    def log_energy():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        energies.append(etot)

    dyn_nve.attach(log_energy, interval=1)
    dyn_nve.run(200) # Run for 200 steps

    # 4. Analysis
    energies = np.array(energies)
    drift = (energies[-1] - energies[0]) / abs(energies[0])

    print(f"Energy drift: {drift:.6%}")

    # Allow 1% drift for this simple test with EMT and 1fs timestep
    # EMT is not perfectly conservative with large timesteps, but 1fs should be okay.
    assert abs(drift) < 0.01, f"Energy drift too high: {drift:.2%}"

def test_langevin_thermostat_temperature():
    """
    Test that Langevin thermostat maintains target temperature.
    """
    atoms = bulk('Cu', 'fcc', a=3.6) * (3, 3, 3)
    atoms.calc = EMT()

    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    target_temp = 300
    MaxwellBoltzmannDistribution(atoms, temperature_K=target_temp)
    Stationary(atoms)

    # Using higher friction to ensure coupling for this test
    # 0.01 atomic units ~ 0.01 / (24 fs) -> very high coupling
    # Friction in ASE is in inverse time units. 0.01 is actually quite weak if units are default (fs-related?)
    # ASE units: friction is in fs^-1 usually? No, check doc: "Friction strength in atomic units (inverse time)."
    # Actually ASE documentation says: friction: Friction coefficient in atomic units.
    # To get strong coupling we might need higher value.
    # Let's increase friction to 0.02

    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=target_temp, friction=0.02)

    temps = []
    def log_temp():
        temps.append(atoms.get_temperature())

    dyn.attach(log_temp, interval=5)
    dyn.run(4000) # Run longer

    # Check average temperature of last half
    avg_temp = np.mean(temps[len(temps)//2:])

    # Allow 20% deviation (small system fluctuations)
    # 3x3x3 Cu is 108 atoms. Fluctuations should be ~ 1/sqrt(3*108) ~ 5%. 20% is generous but safe.
    assert abs(avg_temp - target_temp) < target_temp * 0.2, f"Temperature deviation too high: {avg_temp} vs {target_temp}"
