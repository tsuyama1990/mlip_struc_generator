import pytest
import sys
from unittest.mock import MagicMock, call
from contextlib import contextmanager
import numpy as np
from ase import Atoms
from nnp_gen.explorers.md_engine import run_single_md_thread, MDExplorer, _get_calculator, CalculatorPool
from nnp_gen.core.config import ExplorationConfig, EnsembleType, MonteCarloConfig, MCStrategy

class MockCalculator:
    def get_forces(self, atoms):
        return np.zeros((len(atoms), 3))
    def get_potential_energy(self, atoms):
        return 0.0
    def get_stress(self, atoms):
        return np.zeros(6)

@pytest.fixture
def mock_deps(mocker):
    mocker.patch.dict("sys.modules", {
        "mace": MagicMock(),
        "mace.calculators": MagicMock(),
        "torch": MagicMock(),
    })

def test_get_calculator_mace(mock_deps, mocker):
    # Setup sys.modules for mace.calculators
    mace_pkg = sys.modules["mace.calculators"]

    # Mock mace_mp function which is tried first
    mace_pkg.mace_mp = MagicMock(return_value=MockCalculator())

    # Also mock MACECalculator just in case
    mace_pkg.MACECalculator = MagicMock(return_value=MockCalculator())

    calc = _get_calculator("mace", "cpu")
    assert isinstance(calc, MockCalculator)

# Helper to mock pool
class MockPool:
    def __init__(self):
        self.calc = MockCalculator()
    
    @contextmanager
    def get_calculator(self):
        yield self.calc

@pytest.fixture
def base_config():
    return ExplorationConfig(
        method="md",
        model_name="mace",
        temperature_mode="constant",
        temperature=300,
        steps=10,
        timestep=1.0,
        ensemble=EnsembleType.NVT
    )

def test_run_single_md_success(mocker, base_config):
    # Mock _get_calculator to avoid imports and logic
    mocker.patch("nnp_gen.explorers.md_engine._get_calculator", return_value=MockCalculator())

    # Mock integrator to avoid real dynamics
    mock_dyn = MagicMock()
    mocker.patch("nnp_gen.explorers.md_engine._get_integrator", return_value=mock_dyn)

    atoms = Atoms('H2', positions=[[0,0,0], [2,0,0]])
    mock_pool = MockPool()
    
    traj = run_single_md_thread(atoms, expl_config=base_config, calc_pool=mock_pool, timeout_seconds=10)

    assert traj is not None
    # Check that dyn.run was called
    assert mock_dyn.run.called

def test_run_single_md_explosion(mocker, base_config):
    mocker.patch("nnp_gen.explorers.md_engine._get_calculator", return_value=MockCalculator())
    mock_dyn = MagicMock()
    mocker.patch("nnp_gen.explorers.md_engine._get_integrator", return_value=mock_dyn)

    # Atoms very close -> 0.1 < 0.6
    atoms = Atoms('H2', positions=[[0,0,0], [0.1,0,0]])
    mock_pool = MockPool()
    
    traj = run_single_md_thread(atoms, expl_config=base_config, calc_pool=mock_pool, timeout_seconds=10)

    assert traj is None

def test_explore_parallel(mocker, base_config):
    # Mock ThreadPoolExecutor (changed from ProcessPoolExecutor)
    mock_executor = MagicMock()
    mock_executor.__enter__.return_value = mock_executor
    mocker.patch("concurrent.futures.ThreadPoolExecutor", return_value=mock_executor)

    # Mock submit
    mock_future = MagicMock()
    mock_future.result.return_value = [Atoms('H')]
    mock_executor.submit.return_value = mock_future

    # Mock as_completed
    mocker.patch("concurrent.futures.as_completed", return_value=[mock_future, mock_future])

    # Mock CalculatorPool init to avoid real calc creation
    mocker.patch("nnp_gen.explorers.md_engine.CalculatorPool")

    config = MagicMock()
    config.exploration = base_config

    explorer = MDExplorer(config)
    seeds = [Atoms('H'), Atoms('He')]

    results = explorer.explore(seeds, n_workers=2)

    # 2 seeds -> 2 futures -> each returns [Atoms('H')] -> 2 atoms total
    assert len(results) == 2
    assert results[0].symbols == 'H'

def test_run_single_md_gradient_fix(mocker):
    # Test for the fix: verifying that set_temperature AND attribute update happen

    # We use MC config to force smaller chunks so we can observe the gradient update
    mc_config = MonteCarloConfig(
        enabled=True,
        strategy=[MCStrategy.SWAP],
        swap_interval=10 # 10 steps per chunk
    )

    config = ExplorationConfig(
        method="md",
        model_name="mace",
        temperature_mode="gradient",
        temp_start=300,
        temp_end=400,
        steps=100,
        timestep=1.0,
        ensemble=EnsembleType.NVT,
        mc_config=mc_config
    )

    mocker.patch("nnp_gen.explorers.md_engine._get_calculator", return_value=MockCalculator())
    mocker.patch("nnp_gen.explorers.mc_moves.perform_mc_swap", return_value=True) # Mock MC move

    # Mock integrator
    mock_dyn = MagicMock()
    # Mock attribute for Langevin
    mock_dyn.temp = 300.0
    # Delete 'temperature' to simulate Langevin which only has 'temp'
    del mock_dyn.temperature

    mocker.patch("nnp_gen.explorers.md_engine._get_integrator", return_value=mock_dyn)

    atoms = Atoms('H2', positions=[[0,0,0], [2,0,0]])
    mock_pool = MockPool()

    run_single_md_thread(atoms, expl_config=config, calc_pool=mock_pool, timeout_seconds=10)

    # Verify set_temperature was called with increasing values
    assert mock_dyn.set_temperature.called

    # Verify the attribute 'temp' was updated
    # Last chunk starts at 90. 90/100 = 0.9. 300 + 100*0.9 = 390.
    # So it should be close to 400.
    assert mock_dyn.temp > 350.0
    assert abs(mock_dyn.temp - 390.0) < 5.0 # Should be 390.0

    # Also verify that if we use a NPT mock (with .temperature), it works
    mock_dyn_npt = MagicMock()
    mock_dyn_npt.temperature = 300.0
    del mock_dyn_npt.temp
    mocker.patch("nnp_gen.explorers.md_engine._get_integrator", return_value=mock_dyn_npt)

    run_single_md_thread(atoms, expl_config=config, calc_pool=mock_pool, timeout_seconds=10)
    assert mock_dyn_npt.temperature > 350.0
    assert abs(mock_dyn_npt.temperature - 390.0) < 5.0
