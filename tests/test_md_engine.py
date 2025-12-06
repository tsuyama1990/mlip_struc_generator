import pytest
import sys
from unittest.mock import MagicMock
import numpy as np
from ase import Atoms
from nnp_gen.explorers.md_engine import run_single_md, MDExplorer, _get_calculator

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

    calc = _get_calculator("mace", "cpu")
    assert isinstance(calc, MockCalculator)

def test_run_single_md_success(mocker):
    # Mock _get_calculator to avoid imports and logic
    mocker.patch("nnp_gen.explorers.md_engine._get_calculator", return_value=MockCalculator())

    atoms = Atoms('H2', positions=[[0,0,0], [2,0,0]])
    traj = run_single_md(atoms, temp=300, steps=10, interval=2, calculator_params={})

    assert traj is not None
    # 10 steps. (i+1)%2 == 0 -> 2, 4, 6, 8, 10. 5 frames.
    assert len(traj) == 5

def test_run_single_md_explosion(mocker):
    mocker.patch("nnp_gen.explorers.md_engine._get_calculator", return_value=MockCalculator())

    # Atoms very close -> 0.1 < 0.6
    atoms = Atoms('H2', positions=[[0,0,0], [0.1,0,0]])
    traj = run_single_md(atoms, temp=300, steps=10, interval=1, calculator_params={})

    assert traj is None

def test_explore_parallel(mocker):
    # Mock ProcessPoolExecutor to avoid spawning processes
    mock_executor = MagicMock()
    mock_executor.__enter__.return_value = mock_executor
    mocker.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_executor)

    # Mock submit
    mock_future = MagicMock()
    mock_future.result.return_value = [Atoms('H')]
    mock_executor.submit.return_value = mock_future

    # Mock as_completed
    # Since explore calls as_completed(futures), we simulate it returning the futures
    # We pass 2 seeds, so 2 futures submitted
    mocker.patch("concurrent.futures.as_completed", return_value=[mock_future, mock_future])

    config = MagicMock()
    config.exploration.temperature = 300
    config.exploration.steps = 100

    explorer = MDExplorer(config)
    seeds = [Atoms('H'), Atoms('He')]

    results = explorer.explore(seeds, n_workers=2)

    # 2 seeds -> 2 futures -> each returns [Atoms('H')] -> 2 atoms total
    assert len(results) == 2
    assert results[0].symbols == 'H'
