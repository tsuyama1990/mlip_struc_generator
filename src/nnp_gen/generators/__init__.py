from nnp_gen.core.config import (
    IonicSystemConfig,
    AlloySystemConfig,
    CovalentSystemConfig,
    InterfaceSystemConfig,
    MoleculeSystemConfig,
    VacuumAdsorbateSystemConfig,
    SolventAdsorbateSystemConfig,
    KnowledgeSystemConfig,
    FileSystemConfig,
    RandomSystemConfig,
    DesignatedSystemConfig,
    SystemConfig
)
from nnp_gen.generators.ionic import IonicGenerator
from nnp_gen.generators.alloy import AlloyGenerator
from nnp_gen.generators.covalent import CovalentGenerator
from nnp_gen.generators.interface import InterfaceGenerator
from nnp_gen.generators.molecule import MoleculeGenerator
from nnp_gen.generators.adsorption import VacuumAdsorbateGenerator, SolventAdsorbateGenerator
from nnp_gen.generators.knowledge import KnowledgeBasedGenerator
from nnp_gen.generators.file_loader import FileGenerator
from nnp_gen.generators.random_gen import RandomGenerator
from nnp_gen.generators.designated import DesignatedGenerator
from typing import Type

class GeneratorFactory:
    @staticmethod
    def get_generator(config: SystemConfig):
        if isinstance(config, IonicSystemConfig):
            return IonicGenerator(config)
        elif isinstance(config, AlloySystemConfig):
            return AlloyGenerator(config)
        elif isinstance(config, CovalentSystemConfig):
            return CovalentGenerator(config)
        elif isinstance(config, InterfaceSystemConfig):
            return InterfaceGenerator(config)
        elif isinstance(config, MoleculeSystemConfig):
            return MoleculeGenerator(config)
        elif isinstance(config, VacuumAdsorbateSystemConfig):
            return VacuumAdsorbateGenerator(config)
        elif isinstance(config, SolventAdsorbateSystemConfig):
            return SolventAdsorbateGenerator(config)
        elif isinstance(config, KnowledgeSystemConfig):
            return KnowledgeBasedGenerator(config)
        elif isinstance(config, FileSystemConfig):
            return FileGenerator(config)
        elif isinstance(config, RandomSystemConfig):
            return RandomGenerator(config)
        elif isinstance(config, DesignatedSystemConfig):
            return DesignatedGenerator(config)
        else:
            raise ValueError(f"Unknown system type: {type(config)}")
