import pytest
import numpy as np
from ase import Atoms
from pydantic import ValidationError
from nnp_gen.core.config import (
    IonicSystemConfig,
    ExplorationConfig,
    MonteCarloConfig,
    MCStrategy,
    EnsembleType,
    FileSystemConfig
)
from nnp_gen.core.physics import detect_vacuum
from nnp_gen.explorers.md_engine import _get_integrator
from nnp_gen.explorers.mc_moves import perform_mc_swap as _perform_mc_swap
from ase.md.npt import NPT
from ase.md.langevin import Langevin
from nnp_gen.generators.file_loader import FileGenerator
import os

# --- 1. Config Validation Tests ---

def test_vacancy_config_validation():
    # Valid
    config = IonicSystemConfig(
        elements=["Na", "Cl"],
        oxidation_states={"Na": 1, "Cl": -1},
        vacancy_concentration=0.1
    )
    assert config.vacancy_concentration == 0.1

    # Invalid (> 0.25)
    with pytest.raises(ValidationError):
        IonicSystemConfig(
            elements=["Na", "Cl"],
            oxidation_states={"Na": 1, "Cl": -1},
            vacancy_concentration=0.3
        )

    # Invalid (< 0.0)
    with pytest.raises(ValidationError):
        IonicSystemConfig(
            elements=["Na", "Cl"],
            oxidation_states={"Na": 1, "Cl": -1},
            vacancy_concentration=-0.1
        )

def test_mc_config_validation():
    # Valid
    mc = MonteCarloConfig(
        enabled=True,
        strategy=[MCStrategy.SWAP],
        swap_pairs=[("Fe", "Ni")]
    )
    assert mc.swap_pairs == [("Fe", "Ni")]

    # List of Lists input (should be converted to Tuples by validator)
    mc2 = MonteCarloConfig(
        enabled=True,
        swap_pairs=[["Na", "Cl"]]
    )
    assert mc2.swap_pairs == [("Na", "Cl")]

# --- 2. Physics / Vacuum Detection Tests ---

def test_detect_vacuum():
    # 1. Slab (Vacuum in Z) - Non-Periodic Z
    slab = Atoms("Cu4", positions=[[0,0,0], [1,0,0], [0,1,0], [0,0,1]], cell=[10,10,10], pbc=[True, True, False])
    assert detect_vacuum(slab) is True

    # 2. Bulk (Full PBC, filled)
    bulk = Atoms("Cu", positions=[[0,0,0]], cell=[2,2,2], pbc=[True, True, True])
    # Single atom in small cell is bulk-ish if cell small
    # But let's check histogram logic.
    # With 1 atom at 0, gap is L=2. Gap is (2-0)+0 = 2?
    # Actually code calculates gaps.
    # L=2. coords=[0]. Periodic Gap = (2-0)+0 = 2.
    # If threshold=5.0, 2 < 5. So False (Bulk).
    assert detect_vacuum(bulk, threshold=5.0) is False

    # 3. Slab manually created in PBC box (Vacuum layer > threshold)
    # Cell 20A. Atoms cluster at 0. Gap ~20.
    slab_pbc = Atoms("Cu", positions=[[0,0,0]], cell=[20,20,20], pbc=[True, True, True])
    # Gap will be 20. > 5.0 -> True.
    assert detect_vacuum(slab_pbc, threshold=5.0) is True

# --- 3. MC Logic Tests ---

def test_integrator_selection():
    # Setup Config
    expl_config = ExplorationConfig(
        ensemble=EnsembleType.AUTO,
        timestep=1.0,
        temperature=300
    )

    # Bulk -> NPT
    atoms_bulk = Atoms("Cu", positions=[[0,0,0]], cell=[3,3,3], pbc=True)
    integ = _get_integrator(atoms_bulk, expl_config, 300)
    assert isinstance(integ, NPT)

    # Vacuum -> NVT
    atoms_vac = Atoms("Cu", positions=[[0,0,0]], cell=[20,20,20], pbc=True)
    integ2 = _get_integrator(atoms_vac, expl_config, 300)
    assert isinstance(integ2, Langevin)

def test_mc_charge_safety():
    # Setup Atoms with Charges
    atoms = Atoms("NaCl", positions=[[0,0,0], [2,0,0]], cell=[10,10,10], pbc=True)
    atoms.set_initial_charges([1.0, -1.0])

    # Config: Mismatch NOT allowed
    mc_config = MonteCarloConfig(
        enabled=True,
        strategy=[MCStrategy.SWAP],
        swap_pairs=[("Na", "Cl")],
        allow_charge_mismatch=False
    )

    # Attempt Swap (should fail/return False)
    # We need to mock Random to force SWAP choice and pair choice?
    # But we only have 1 strategy and 1 pair.
    # We assume _perform_mc_swap picks the only valid pair.

    # Pass a dummy calculator (can be None if we mock get_potential_energy, but code calls it)
    # Let's mock atoms.get_potential_energy
    atoms.calc = None
    # We can just monkeypatch atoms.get_potential_energy
    atoms.get_potential_energy = lambda: 0.0

    result = _perform_mc_swap(atoms, mc_config, 300.0, None)

    assert result is False
    assert atoms.symbols[0] == "Na" # Unchanged

    # Config: Mismatch ALLOWED
    mc_config.allow_charge_mismatch = True
    result = _perform_mc_swap(atoms, mc_config, 300.0, None)

    # Metropolis acceptance is random if dE=0 (prob=1).
    # dE = 0 -> accepted.
    assert result is True
    # Symbols swapped? Or reverted if rejected? Accepted -> Swapped.
    assert atoms.symbols[0] == "Cl"

# --- 4. File Loading Tests (Epic 5) ---

def test_file_generator_vacancies(tmp_path):
    # Create dummy file with VALID structure (FCC Cu)
    from ase.build import bulk
    fpath = tmp_path / "test.xyz"
    # Create supercell to have enough atoms. cubic=True gives 4 atoms per cell.
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True) * (4, 4, 4) # 4 * 64 = 256 atoms
    atoms.write(fpath)

    # Config
    config = FileSystemConfig(
        type="from_files",
        elements=["Cu"],
        path=str(fpath),
        vacancy_concentration=0.1,
        repeat=1,
        constraints={"max_atoms": 500}
    )

    gen = FileGenerator(config)
    results = gen.generate()

    assert len(results) == 1
    # 256 atoms. 10% vacancies = 25 atoms removed. 256 - 25 = 231.
    assert len(results[0]) == 231

# --- 5. Pydantic Verification Tests ---

def test_user_file_validation_failure():
    with pytest.raises(ValidationError) as excinfo:
        FileSystemConfig(
            type="from_files",
            elements=["Cu"],
            path="dummy.cif",
            repeat=0 # Invalid, must be >= 1
        )
    assert "repeat" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
         FileSystemConfig(
            type="from_files",
            elements=["Cu"],
            path="dummy.cif",
            vacancy_concentration=0.5 # Invalid, > 0.25
        )
    assert "vacancy_concentration" in str(excinfo.value)

def test_exploration_validation_failure():
    with pytest.raises(ValidationError) as excinfo:
        ExplorationConfig(
            timestep=0.05 # Invalid, <= 0.1
        )
    assert "timestep" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        ExplorationConfig(
            steps=0 # Invalid
        )
    assert "steps" in str(excinfo.value)

from nnp_gen.core.config import AppConfig, SamplingConfig

def test_app_config_integration():
    # Construct a full valid config to ensure integration
    app_config = AppConfig(
        system=IonicSystemConfig(
            elements=["Na", "Cl"],
            oxidation_states={"Na": 1, "Cl": -1},
            vacancy_concentration=0.1
        ),
        exploration=ExplorationConfig(
            method="hybrid_mc_md",
            ensemble=EnsembleType.AUTO,
            mc_config=MonteCarloConfig(enabled=True, swap_pairs=[("Na", "Cl")], allow_charge_mismatch=True)
        ),
        sampling=SamplingConfig(),
        output_dir="test_out",
        seed=123
    )
    assert app_config.system.vacancy_concentration == 0.1
    assert app_config.exploration.mc_config.enabled is True
