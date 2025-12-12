import pytest
from unittest.mock import MagicMock, patch
from nnp_gen.core.calculators import CalculatorFactory
from nnp_gen.explorers.md_engine import MDExplorer
from nnp_gen.core.config import ExplorationConfig

def test_missing_mace_dependency():
    """
    Test that missing mace dependency raises a clear ImportError
    with helpful installation instructions.
    """
    with patch.dict('sys.modules', {'mace': None, 'mace.calculators': None}):
        # Try to build mace calculator
        # We need to access the builder directly since get() might mask things or rely on registration
        # But CalculatorFactory.get("mace", ...) is the public API
        
        with pytest.raises(ImportError) as exc_info:
             CalculatorFactory.get("mace", "cpu", model_paths="small")
        
        assert "pip install mace-torch" in str(exc_info.value)

def test_md_explorer_pre_flight_check():
    """
    Test that MDExplorer.explore() performs a pre-flight check
    and fails FAST if the calculator is missing, without spawning workers.
    """
    
    # Mock config
    config = MagicMock()
    config.exploration = MagicMock(spec=ExplorationConfig)
    config.exploration.model_name = "mace"
    config.exploration.method = "md"
    config.exploration.device = "cpu"
    
    # Mock CalculatorFactory.get to raise ImportError (simulating missing dependency)
    with patch("nnp_gen.explorers.md_engine._get_calculator", side_effect=ImportError("Mocked Missing Module")):
         explorer = MDExplorer(config)
         
         # Should verify calculator availability and fail
         with pytest.raises(RuntimeError) as exc_info:
             explorer.explore(seeds=[])
             
         assert "Calculator 'mace' failed to initialize" in str(exc_info.value)
         assert "Mocked Missing Module" in str(exc_info.value)

