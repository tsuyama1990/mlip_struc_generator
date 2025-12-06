from nnp_gen.core.config import (
    VacuumAdsorbateSystemConfig,
    SolventAdsorbateSystemConfig,
    AdsorbateConfig,
    AdsorbateMode,
    AlloySystemConfig
)
import pytest
from pydantic import ValidationError

def test_adsorbate_config_validation():
    # Valid
    config = AdsorbateConfig(source="O", mode=AdsorbateMode.ATOM, count=1, height=2.0)
    assert config.count == 1

    # Invalid count
    with pytest.raises(ValidationError):
        AdsorbateConfig(source="O", mode=AdsorbateMode.ATOM, count=0)

    with pytest.raises(ValidationError):
        AdsorbateConfig(source="O", mode=AdsorbateMode.ATOM, count=-1)

def test_vacuum_adsorbate_system_config_validation():
    substrate_config = {
        "type": "alloy",
        "elements": ["Fe"],
        "lattice_constant": 2.86,
    }

    # Valid
    config = VacuumAdsorbateSystemConfig(
        substrate=substrate_config,
        miller_indices=[(1, 0, 0)],
        defect_rate=0.5,
        elements=["Fe"] # Should be optional if extracted, but let's test explicit first or extracted logic
    )
    assert config.defect_rate == 0.5

    # Invalid defect_rate
    with pytest.raises(ValidationError):
        VacuumAdsorbateSystemConfig(
            substrate=substrate_config,
            miller_indices=[(1, 0, 0)],
            defect_rate=1.1,
            elements=["Fe"]
        )

    with pytest.raises(ValidationError):
        VacuumAdsorbateSystemConfig(
            substrate=substrate_config,
            miller_indices=[(1, 0, 0)],
            defect_rate=-0.1,
            elements=["Fe"]
        )

def test_element_extraction():
    substrate_config = {
        "type": "alloy",
        "elements": ["Au"],
        "lattice_constant": 4.08,
    }

    # Validation should extract elements from substrate
    config = VacuumAdsorbateSystemConfig(
        substrate=substrate_config,
        miller_indices=[(1, 1, 1)]
    )
    assert "Au" in config.elements
    assert len(config.elements) == 1

def test_solvent_adsorbate_inheritance():
    substrate_config = {
        "type": "alloy",
        "elements": ["Pt"],
    }
    config = SolventAdsorbateSystemConfig(
        substrate=substrate_config,
        miller_indices=[(1, 1, 1)],
        solvent_density=1.0,
        solvent_smiles="O"
    )
    assert config.solvent_density == 1.0
    assert config.solvent_smiles == "O"
    assert config.type == "solvent_adsorbate"
