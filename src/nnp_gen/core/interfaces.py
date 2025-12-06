from abc import ABC, abstractmethod
from typing import List
import logging
import numpy as np
from ase import Atoms
from nnp_gen.core.config import SystemConfig

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """
    Abstract Base Class for structure generators with built-in validation.
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    def generate(self) -> List[Atoms]:
        """
        Generates initial structures and filters them based on physics constraints.

        Returns:
            List[Atoms]: A list of valid ASE Atoms objects.
        """
        raw_structures = self._generate_impl()
        valid_structures = []

        for i, atoms in enumerate(raw_structures):
            # Enforce PBC from config before validation to ensure distance checks use correct MIC
            atoms.set_pbc(self.config.pbc)

            if self.validate_structure(atoms):
                valid_structures.append(atoms)
            else:
                logger.warning(f"Structure {i} rejected by SanityFilter.")

        return valid_structures

    @abstractmethod
    def _generate_impl(self) -> List[Atoms]:
        """
        Implementation of the specific generation logic.
        """
        pass

    def validate_structure(self, atoms: Atoms) -> bool:
        """
        Validates a single structure against constraints.
        """
        constraints = self.config.constraints

        # 1. Max Atoms
        if len(atoms) > constraints.max_atoms:
            logger.warning(f"Validation Failed: {len(atoms)} atoms > max {constraints.max_atoms}")
            return False

        # 2. Min Distance
        # self-interaction=False (don't check distance to self)
        # mic=True (minimum image convention) if pbc is True
        # Note: get_all_distances with mic=True can be slow for large systems but for <200 atoms it's fast.
        if len(atoms) > 1:
            # We use mic=True if any PBC is set, but let's just assume we check real distances
            # If atoms are overlapping, it's bad.
            # get_all_distances returns N x N matrix
            dists = atoms.get_all_distances(mic=any(atoms.pbc))
            # Set diagonal to infinity to ignore self-distance
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)

            if min_dist < constraints.min_distance:
                logger.warning(f"Validation Failed: Min distance {min_dist:.2f} < {constraints.min_distance}")
                return False

        # 3. Min Density
        # density = mass / volume. (g/cm^3)
        # ASE has no direct get_density() but likely we can compute if volume is defined
        try:
            vol = atoms.get_volume()
            if vol > 1e-6: # Avoid div by zero
                # sum of masses
                total_mass = sum(atoms.get_masses()) # amu
                # 1 amu/A^3 = 1.66054 g/cm^3
                density = (total_mass / vol) * 1.66054

                if density < constraints.min_density:
                     logger.warning(f"Validation Failed: Density {density:.2f} < {constraints.min_density}")
                     return False
        except Exception:
            # get_volume might fail for non-periodic systems or if cell is zero
            pass

        return True
