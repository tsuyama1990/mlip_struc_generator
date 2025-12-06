import pytest
from pydantic import ValidationError
from nnp_gen.core.config import (
    AppConfig,
    SystemConfig,
    AlloySystemConfig,
    IonicSystemConfig,
    MoleculeSystemConfig
)

def test_alloy_config_valid():
    config_data = {
        "type": "alloy",
        "elements": ["Cu", "Au"],
        "lattice_constant": 3.8
    }
    config = AlloySystemConfig(**config_data)
    assert config.type == "alloy"
    assert config.elements == ["Cu", "Au"]
    assert config.lattice_constant == 3.8

def test_ionic_config_valid():
    config_data = {
        "type": "ionic",
        "elements": ["Na", "Cl"],
        "oxidation_states": {"Na": 1, "Cl": -1}
    }
    config = IonicSystemConfig(**config_data)
    assert config.type == "ionic"
    assert config.oxidation_states["Na"] == 1

def test_ionic_config_missing_field():
    config_data = {
        "type": "ionic",
        "elements": ["Na", "Cl"]
        # Missing oxidation_states
    }
    with pytest.raises(ValidationError):
        IonicSystemConfig(**config_data)

def test_app_config_polymorphism():
    """Test that AppConfig correctly parses the system config based on type"""
    app_data = {
        "system": {
            "type": "molecule",
            "elements": ["H", "O"],
            "smiles": "O"
        },
        "exploration": {
            "method": "md",
            "temperature": 300
        },
        "sampling": {
            "strategy": "fps",
            "n_samples": 50
        }
    }
    app_config = AppConfig(**app_data)
    assert isinstance(app_config.system, MoleculeSystemConfig)
    assert app_config.system.smiles == "O"

def test_app_config_polymorphism_alloy():
    app_data = {
        "system": {
            "type": "alloy",
            "elements": ["Fe", "Ni"]
        },
        "exploration": {},
        "sampling": {}
    }
    app_config = AppConfig(**app_data)
    assert isinstance(app_config.system, AlloySystemConfig)
