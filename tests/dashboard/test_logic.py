import pytest
import pandas as pd
from pathlib import Path
from pydantic import ValidationError
from dashboard.logic.config_manager import ConfigManager, SimulationConfig
from dashboard.logic.simulation_mock import MockSimulation
from dashboard.logic.analysis import AnalysisData

# --- Test ConfigManager ---

def test_config_model_validation():
    # Valid config
    config = SimulationConfig(composition="FeNi", temperature=300.0, atom_limit=100)
    assert config.temperature == 300.0

    # Invalid temperature (too high)
    with pytest.raises(ValidationError):
        SimulationConfig(composition="FeNi", temperature=6000.0, atom_limit=100)

    # Invalid atom limit (negative)
    with pytest.raises(ValidationError):
        SimulationConfig(composition="FeNi", temperature=300.0, atom_limit=-5)

def test_config_manager_load_save(tmp_path):
    config_file = tmp_path / "config.yaml"
    manager = ConfigManager(config_path=str(config_file))

    # Save
    original_config = SimulationConfig(composition="CuZr", temperature=500.0)
    manager.save_config(original_config)

    assert config_file.exists()

    # Load
    loaded_config = manager.load_config()
    assert loaded_config.composition == "CuZr"
    assert loaded_config.temperature == 500.0

# --- Test SimulationMock ---

def test_mock_simulation(capsys):
    sim = MockSimulation()
    sim.run({"test": "config"})

    captured = capsys.readouterr()
    assert "Starting mock simulation" in captured.out
    assert "Mock simulation finished" in captured.out

# --- Test AnalysisData ---

def test_analysis_data_generation(tmp_path):
    # Use tmp_path to avoid messing with real data
    data_dir = tmp_path / "data"
    analysis = AnalysisData(data_dir=str(data_dir))

    # Check if files generated
    assert (data_dir / "pca.csv").exists()
    assert (data_dir / "structures").exists()

    # Check data loading
    df_pca = analysis.get_pca_data()
    assert not df_pca.empty
    assert "pc1" in df_pca.columns

    # Check structure loading
    struct_id = df_pca.iloc[0]['structure_id']
    df_struct = analysis.get_structure(struct_id)
    assert not df_struct.empty
    assert "x" in df_struct.columns
