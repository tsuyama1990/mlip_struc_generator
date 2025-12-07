import logging
import numpy as np
from typing import List
from ase import Atoms
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import CovalentSystemConfig
from nnp_gen.core.physics import apply_vacancies

logger = logging.getLogger(__name__)

class CovalentGenerator(BaseGenerator):
    def __init__(self, config: CovalentSystemConfig):
        super().__init__(config)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates covalent structures using random symmetry groups (PyXTal).
        """
        logger.info(f"Generating covalent structures for {self.config.elements}")

        try:
            from pyxtal import pyxtal
        except ImportError:
            logger.error("pyxtal not installed.")
            raise ImportError("pyxtal is required for CovalentGenerator")

        structures = []
        n_structures = 5 # Default number to generate

        rng = np.random.RandomState(42)

        for _ in range(n_structures):
            try:
                struc = pyxtal()

                dim = self.config.dimensionality
                if dim == 3:
                    sg = rng.randint(1, 231)
                elif dim == 2:
                    sg = rng.randint(1, 81)
                elif dim == 1:
                    sg = rng.randint(1, 76)
                else:
                    sg = 1

                # Random stoichiometry: 1 to 4 atoms per element type
                num_ions = [rng.randint(1, 5) for _ in self.config.elements]

                # from_random(dim, group, species, numIons)
                struc.from_random(dim, sg, self.config.elements, num_ions)

                if struc.valid:
                    ase_atoms = struc.to_ase()

                    # Apply Vacancies
                    if self.config.vacancy_concentration > 0.0:
                        ase_atoms = apply_vacancies(ase_atoms, self.config.vacancy_concentration, rng)

                    structures.append(ase_atoms)
            except Exception as e:
                # pyxtal generation can fail often, just skip
                logger.debug(f"Generation failed: {e}")
                continue

        if not structures:
            logger.warning("No valid structures generated via PyXTal.")

        return structures
