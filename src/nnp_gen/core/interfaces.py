from abc import ABC, abstractmethod
from typing import List, Any
from ase import Atoms
from nnp_gen.core.config import SystemConfig

class BaseGenerator(ABC):
    """
    Abstract Base Class for structure generators.
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    @abstractmethod
    def generate(self) -> List[Atoms]:
        """
        Generates initial structures.

        Returns:
            List[Atoms]: A list of ASE Atoms objects.
        """
        pass
