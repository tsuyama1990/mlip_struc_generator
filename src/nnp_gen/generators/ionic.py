import logging
import numpy as np
from typing import List
from ase import Atoms
from ase.build import bulk
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import IonicSystemConfig

logger = logging.getLogger(__name__)

class IonicGenerator(BaseGenerator):
    def __init__(self, config: IonicSystemConfig):
        super().__init__(config)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates ionic structures.
        Currently supports binary compounds (Rocksalt structure) as a baseline.
        """
        logger.info(f"Generating ionic structures for {self.config.elements}")

        for el in self.config.elements:
            if el not in self.config.oxidation_states:
                logger.error(f"Oxidation state for {el} not defined.")
                raise ValueError(f"Oxidation state for {el} missing")

        structures = []

        elements = self.config.elements
        if len(elements) == 2:
            # Assume binary ionic
            # We use the first element as cation-like and second as anion-like for placement
            cation = elements[0]
            anion = elements[1]

            # Default lattice constant
            a = 5.0

            try:
                # Create primitive Rocksalt cell
                prim = bulk('NaCl', 'rocksalt', a=a)

                symbols = prim.get_chemical_symbols()
                new_symbols = []
                for s in symbols:
                    if s == 'Na':
                        new_symbols.append(cation)
                    elif s == 'Cl':
                        new_symbols.append(anion)
                    else:
                        new_symbols.append(cation)

                prim.set_chemical_symbols(new_symbols)

                # Supercell
                atoms = prim * self.config.supercell_size
                structures.append(atoms)

            except Exception as e:
                logger.error(f"Failed to generate ionic structure: {e}")

        else:
            logger.warning("Only binary ionic compounds supported currently.")

        return structures
