import logging
from typing import Callable, Any, Dict, Optional
from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)

class CalculatorFactory:
    _builders: Dict[str, Callable[[str, Dict[str, Any]], Calculator]] = {}

    @classmethod
    def register(cls, name: str, builder: Callable[[str, Dict[str, Any]], Calculator]):
        """
        Register a calculator builder.
        """
        cls._builders[name] = builder

    @classmethod
    def get(cls, name: str, device: str, **kwargs) -> Calculator:
        """
        Get a calculator instance by name.
        """
        if name not in cls._builders:
            raise ValueError(f"Unknown calculator model: {name}. Available: {list(cls._builders.keys())}")

        return cls._builders[name](device, kwargs)

# Define builders (Lazy imports)

def build_mace(device: str, kwargs: Dict[str, Any]) -> Calculator:
    try:
        from mace.calculators import mace_mp
        import torch
        
        # Check if CUDA is requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        
        model_type = kwargs.get('model_paths', 'small')
        return mace_mp(model=model_type, device=device, default_dtype="float32")
    except ImportError:
         raise ImportError("The 'mace' package is not installed. Please install it using 'pip install mace-torch'.")

def build_sevenn(device: str, kwargs: Dict[str, Any]) -> Calculator:
    from sevenn.calculators import SevenNetCalculator
    import torch
    
    # Check if CUDA is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    return SevenNetCalculator(model=kwargs.get('model', '7net-0'), device=device)

def build_emt(device: str, kwargs: Dict[str, Any]) -> Calculator:
    from ase.calculators.emt import EMT
    return EMT()

# Register default calculators
CalculatorFactory.register("mace", build_mace)
CalculatorFactory.register("sevenn", build_sevenn)
CalculatorFactory.register("emt", build_emt)
