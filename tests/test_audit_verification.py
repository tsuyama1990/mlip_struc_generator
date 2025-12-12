
import pytest
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock, patch
from nnp_gen.core.config import AlloySystemConfig, IonicSystemConfig, PhysicsConstraints
from nnp_gen.generators.alloy import AlloyGenerator
from nnp_gen.generators.ionic import IonicGenerator, PymatgenRadiusStrategy, FallbackRadiusStrategy

def test_mismatched_alloy_generation_no_overlap():
    """
    Test that generating a highly mismatched alloy (e.g. Li-Cs) 
    does not fail with atomic overlap when using method='max'.
    Li radius ~1.52 A, Cs radius ~2.65 A. 
    If we use Vegard's law (mean), a ~ 3.5 (FCC). Cs-Cs dist would be 2.5 < 2*2.65.
    If we use 'max', a ~ 9.0 (Cs bulk). Plenty of weighted space.
    """
    config = AlloySystemConfig(
        elements=["Li", "Cs"],
        type="alloy",
        lattice_estimation_method="max", # The Fix
        constraints=PhysicsConstraints(min_distance=2.0) # 2.0 A is generous check
    )
    gen = AlloyGenerator(config)
    
    # Generate 5 structures to be sure
    structures = []
    for _ in range(5):
        structures.extend(gen.generate())
    
    assert len(structures) == 5
    for atoms in structures:
        # Check that we passed validation inside generate
        assert len(atoms) > 0
        
        # Manually check nearest neighbor distance
        from ase.neighborlist import neighbor_list
        d = neighbor_list('d', atoms, cutoff=2.0)
        # Should have NO neighbors closer than 2.0 A
        # Actually Cs-Cs bond is huge (~5.2 A), Li-Li (~3.0 A).
        # Even Li-Cs (~4.1 A). 
        # So 2.0 A is safe lower bound.
        assert len(d) == 0, f"Found atoms closer than 2.0 A: {np.min(d)}"

def test_ionic_generator_radius_strategy():
    """
    Test that IonicGenerator correctly selects strategies.
    """
    # 1. Test Strict Mode -> Pymatgen Strategy (should raise ImportError if missing, or succeed if present)
    # We mock import to fail to test strict mode failure
    # 1. Test Strict Mode -> Pymatgen Strategy (should raise ImportError if missing, or succeed if present)
    # We mock import to fail to test strict mode failure
    # Patch both pymatgen and pymatgen.core to be safe against different import patterns
    with patch.dict('sys.modules', {'pymatgen': None, 'pymatgen.core': None}):
        config = IonicSystemConfig(
            elements=["Na", "Cl"],
            oxidation_states={"Na": 1, "Cl": -1},
            strict_mode=True
        )
        # Should raise ImportError (or GenerationError wrapping it)
        with pytest.raises((ImportError, Exception)):
             IonicGenerator(config)

    # 2. Test Non-Strict Mode -> Fallback Strategy
    with patch.dict('sys.modules', {'pymatgen': None, 'pymatgen.core': None}):
        config = IonicSystemConfig(
            elements=["Na", "Cl"],
            oxidation_states={"Na": 1, "Cl": -1},
            strict_mode=False
        )
        gen = IonicGenerator(config)
        assert isinstance(gen.strategy, FallbackRadiusStrategy)
        assert gen.has_pmg is False
        
        # Check radius retrieval works via fallback
        # Na covalent radius ~1.54
        r = gen._get_radius("Na") 
        assert r > 1.0

def test_validator_message_content(caplog):
    """
    Test that validation failures log the specific symbols involved.
    """
    from nnp_gen.core.validation import StructureValidator
    from nnp_gen.core.config import PhysicsConstraints
    import logging
    
    # Create a dummy bad structure
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.5]]) # 0.5 A apart
    
    validator = StructureValidator(constraints=PhysicsConstraints(min_distance=1.0))
    
    with caplog.at_level(logging.WARNING):
        result = validator._check_min_distance(atoms)
        
    assert result is False
    # Check log message
    assert "Overlap detected between H-H" in caplog.text
    assert "0.500 A" in caplog.text

if __name__ == "__main__":
    pytest.main([__file__])
