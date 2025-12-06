from abc import ABC, abstractmethod
from typing import List
import logging
import numpy as np
from ase import Atoms
from nnp_gen.core.config import SystemConfig
from nnp_gen.core.physics import (
    apply_rattle,
    apply_volumetric_strain,
    set_initial_magmoms,
    ensure_supercell_size
)

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
            # Pipeline Logic

            # 0. Enforce PBC from config (Crucial for ensure_supercell checks)
            atoms.set_pbc(self.config.pbc)

            # 1. ensure_supercell_size
            atoms = ensure_supercell_size(
                atoms,
                r_cut=self.config.constraints.r_cut,
                factor=self.config.constraints.min_cell_length_factor
            )

            # 2. set_initial_magmoms (if applicable)
            if hasattr(self.config, 'default_magmoms') and self.config.default_magmoms:
                atoms = set_initial_magmoms(atoms, self.config.default_magmoms)

            # 3. apply_volumetric_strain
            if self.config.vol_scale_range:
                atoms = apply_volumetric_strain(atoms, self.config.vol_scale_range)

            # 4. apply_rattle
            if self.config.rattle_std > 0:
                atoms = apply_rattle(atoms, self.config.rattle_std)

            # 5. validate_structure
            if self.validate_structure(atoms):
                valid_structures.append(atoms)
            else:
                logger.warning(f"Structure {i} rejected by SanityFilter.")

        return valid_structures

    @abstractmethod
    def _generate_impl(self) -> List[Atoms]:
        """
        Implementation of the specific generation logic.
        Must return ase.Atoms objects.
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
            # We use mic=True if any PBC is set
            dists = atoms.get_all_distances(mic=any(atoms.pbc))
            # Set diagonal to infinity to ignore self-distance
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)

            if min_dist < constraints.min_distance:
                logger.warning(f"Validation Failed: Min distance {min_dist:.2f} < {constraints.min_distance}")
                return False

        # 3. Min Density
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
