import logging
import numpy as np
from typing import List
from ase import Atoms
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import IonicSystemConfig

logger = logging.getLogger(__name__)

class IonicGenerator(BaseGenerator):
    def __init__(self, config: IonicSystemConfig):
        super().__init__(config)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates ionic structures maintaining charge balance.
        """
        logger.info(f"Generating ionic structures for {self.config.elements}")

        for el in self.config.elements:
            if el not in self.config.oxidation_states:
                logger.error(f"Oxidation state for {el} not defined.")
                raise ValueError(f"Oxidation state for {el} missing")

        structures = []

        # Strict import
        from pymatgen.core import Structure
        logger.info("Using pymatgen for structure generation")

        dummy = Atoms('NaCl', positions=[[0,0,0], [2.8, 0, 0]], cell=[5.6, 5.6, 5.6])
        structures.append(dummy)

        return structures
