from nnp_gen.core.config import SystemConfig
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.generators.alloy import AlloyGenerator
from nnp_gen.generators.ionic import IonicGenerator
from nnp_gen.generators.covalent import CovalentGenerator
from nnp_gen.generators.molecule import MoleculeGenerator
from nnp_gen.generators.interface import InterfaceGenerator
from nnp_gen.generators.adsorption import VacuumAdsorbateGenerator, SolventAdsorbateGenerator
from nnp_gen.generators.file_loader import FileGenerator
from typing import Optional
from nnp_gen.generators.knowledge import KnowledgeBasedGenerator

class GeneratorFactory:
    """
    Factory class to instantiate the appropriate Generator based on SystemConfig.
    """
    @staticmethod
    def get_generator(config: SystemConfig, seed: Optional[int] = 42) -> BaseGenerator:
        """
        Returns an instance of a BaseGenerator subclass.

        Args:
            config (SystemConfig): The system configuration.
            seed (Optional[int]): Random seed.

        Returns:
            BaseGenerator: The generator instance.

        Raises:
            ValueError: If the config type is unknown.
        """
        if config.type == "alloy":
            return AlloyGenerator(config, seed=seed)
        elif config.type == "ionic":
            return IonicGenerator(config, seed=seed)
        elif config.type == "covalent":
            return CovalentGenerator(config, seed=seed)
        elif config.type == "molecule":
            return MoleculeGenerator(config, seed=seed)
        elif config.type == "interface":
            return InterfaceGenerator(config, seed=seed)
        elif config.type == "vacuum_adsorbate":
            return VacuumAdsorbateGenerator(config, seed=seed)
        elif config.type == "solvent_adsorbate":
            return SolventAdsorbateGenerator(config, seed=seed)
        elif config.type == "user_file":
            return FileGenerator(config, seed=seed)
        elif config.type == "knowledge":
            return KnowledgeBasedGenerator(config, seed=seed)
        else:
            raise ValueError(f"Unknown system type: {config.type}")
