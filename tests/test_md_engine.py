import pytest
import sys
from unittest.mock import MagicMock
from contextlib import contextmanager
import numpy as np
from ase import Atoms
from nnp_gen.explorers.md_engine import run_single_md_process, MDExplorer, _get_calculator
from nnp_gen.core.config import ExplorationConfig, MonteCarloConfig

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
    mace_pkg.MACECalculator = MagicMock(return_value=MockCalculator())

    # Mock CalculatorFactory
    mocker.patch("nnp_gen.core.calculators.CalculatorFactory.get", return_value=MockCalculator())

    calc = _get_calculator("mace", "cpu")
    assert isinstance(calc, MockCalculator)

def test_run_single_md_success(mocker):
    # Mock _get_calculator
    mocker.patch("nnp_gen.explorers.md_engine._get_calculator", return_value=MockCalculator())

    atoms = Atoms('H2', positions=[[0,0,0], [2,0,0]])
    atoms.pbc = [True, True, True]
    atoms.set_cell([10, 10, 10])
    
    expl_config = ExplorationConfig(
        method="md",
        model_name="mace",
        temperature_mode="constant",
        temperature=300.0,
        steps=10,
        timestep=1.0,
        ensemble="NVT"
    )

    # Signature: (atoms, expl_config, model_name, device, timeout_seconds)
    traj = run_single_md_process(atoms, expl_config, "mace", "cpu", 3600)

    assert traj is not None
    # steps=10, swap/chunk check might lead to snapshot.
    # snap_interval = max(1, steps//50) = 1.
    # every step might be snapshot if not for chunk logic?
    # MDExplorer: snap_interval = max(1, 10//50) = 1.
    # Logic: if total_steps_done % snap_interval < chunk
    # chunk is likely 10.
    # 0 % 1 < 10 -> True. snap at start?
    # No, logic is inside loop.
    # total_steps_done starts at 0.
    # chunk = 10.
    # dyn.run(10).
    # total_steps_done = 10.
    # if 10 % 1 < 10: 0 < 10 True.
    # traj.append(atoms).
    # so we get 1 frame at end.

    assert len(traj) >= 1

def test_run_single_md_explosion(mocker):
    mocker.patch("nnp_gen.explorers.md_engine._get_calculator", return_value=MockCalculator())

    # Atoms very close -> 0.1 < 0.6
    atoms = Atoms('H2', positions=[[0,0,0], [0.1,0,0]])
    atoms.pbc = [True, True, True]
    atoms.set_cell([10, 10, 10])

    expl_config = ExplorationConfig(
        method="md",
        model_name="mace",
        temperature_mode="constant",
        temperature=300.0,
        steps=10,
        timestep=1.0
    )

    try:
        run_single_md_process(atoms, expl_config, "mace", "cpu", 3600)
        assert False, "Should raise exception"
    except Exception as e:
        # It dumps and re-raises
        assert "min_dist" in str(e) or "overlap" in str(e)

def test_explore_parallel(mocker):
    # Mock ProcessPoolExecutor
    mock_executor = MagicMock()
    mock_executor.__enter__.return_value = mock_executor
    mocker.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_executor)

    # Mock submit
    mock_future = MagicMock()
    mock_future.result.return_value = [Atoms('H')]
    mock_executor.submit.return_value = mock_future

    # Mock as_completed
    mocker.patch("concurrent.futures.as_completed", return_value=[mock_future, mock_future])

    # We mock run_single_md_process so it doesn't run locally in test context?
    # Actually MDExplorer calls run_single_md_process inside submit.
    # We don't need to mock it if executor is mocked.

    config = MagicMock()
    config.exploration.temperature = 300
    config.exploration.steps = 100
    config.exploration.model_name = "mace"
    config.exploration.method = "md"

    explorer = MDExplorer(config)
    seeds = [Atoms('H'), Atoms('He')]

    results = explorer.explore(seeds, n_workers=2)

    # 2 seeds -> 2 futures -> each returns [Atoms('H')] -> 2 atoms total
    assert len(results) == 2
    assert results[0].symbols == 'H'
