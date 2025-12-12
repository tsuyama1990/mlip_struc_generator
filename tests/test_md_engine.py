import pytest
import sys
from unittest.mock import MagicMock
from contextlib import contextmanager
import numpy as np
from ase import Atoms
from nnp_gen.explorers.md_engine import run_single_md_process, MDExplorer, _get_calculator
from nnp_gen.core.config import ExplorationConfig, MonteCarloConfig

class MockCalculator:
    implemented_properties = ["energy", "forces", "stress"]
    def get_forces(self, atoms):
        return np.zeros((len(atoms), 3))
    def get_potential_energy(self, atoms):
        return 0.0
    def get_stress(self, atoms):
        return np.zeros(6)
    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        pass
    def reset(self):
        pass

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

    # Signature: (atoms, expl_config, model_name, device, progress_queue, timeout_seconds, stop_event)
    # The signature in source code is: (atoms, expl_config, model_name, device, progress_queue=None, timeout_seconds=3600, stop_event=None)
    # But in the test call above it was passing 3600 as the 5th argument.
    # 5th argument is `progress_queue`.
    # So `progress_queue` became `3600` (int).
    # And then `progress_queue.put(chunk)` failed with AttributeError: 'int' object has no attribute 'put'.

    traj = run_single_md_process(atoms, expl_config, "mace", "cpu", progress_queue=None, timeout_seconds=3600)

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
        run_single_md_process(atoms, expl_config, "mace", "cpu", progress_queue=None, timeout_seconds=3600)
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
    mock_future1 = MagicMock()
    mock_future1.result.return_value = [Atoms('H')]
    mock_future2 = MagicMock()
    mock_future2.result.return_value = [Atoms('H')]

    # Use side_effect to return different futures
    mock_executor.submit.side_effect = [mock_future1, mock_future2]

    # Mock concurrent.futures.wait
    # It returns (done, not_done).
    # We simulate immediate completion of all futures.
    # Note: wait takes a set of futures. We return the same set as done.
    # We need a side_effect that returns whatever was passed as first arg?
    # Or just return both our mocks.
    mocker.patch("concurrent.futures.wait", return_value=({mock_future1, mock_future2}, set()))

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
