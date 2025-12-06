
import logging
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.units import _amu, _e
from nnp_gen.core.config import PhysicsConstraints

logger = logging.getLogger(__name__)

# 1 amu = 1.66053906660e-24 g
# 1 Å³ = 1e-24 cm³
AMU_TO_G = 1.66053906660e-24  # g/amu
A3_TO_CM3 = 1e-24  # cm³/Å³
conversion_factor = AMU_TO_G / A3_TO_CM3

class StructureValidator:
    """
    Validates atomic structures against physics constraints.
    """
    def __init__(self, constraints: PhysicsConstraints):
        self.constraints = constraints

    def validate(self, atoms: Atoms) -> bool:
        """
        Validates the structure.

        Args:
            atoms: The ASE Atoms object to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not self._check_max_atoms(atoms):
            return False

        if not self._check_min_distance(atoms):
            return False

        if not self._check_min_density(atoms):
            return False

        return True

    def _check_max_atoms(self, atoms: Atoms) -> bool:
        if len(atoms) > self.constraints.max_atoms:
            logger.warning(f"Validation Failed: {len(atoms)} atoms > max {self.constraints.max_atoms}")
            return False
        return True

    def _check_min_distance(self, atoms: Atoms) -> bool:
        if len(atoms) <= 1:
            return True

        min_dist_threshold = self.constraints.min_distance

        try:
            i, j, d = neighbor_list('ijd', atoms, cutoff=min_dist_threshold, self_interaction=False)

            if len(d) > 0:
                min_found = np.min(d)
                if min_found < min_dist_threshold:
                    logger.warning(f"Validation Failed: Min distance {min_found:.2f} < {min_dist_threshold}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Validation Error during distance check: {e}")
            return False

    def _check_min_density(self, atoms: Atoms) -> bool:
        # Skip density check for non-periodic systems
        if not any(atoms.pbc):
            return True

        # Vacuum Detection Heuristic (Histogram Method)
        # Project atoms onto X, Y, Z axes and check for empty bins
        positions = atoms.get_positions()
        cell_lengths = atoms.cell.lengths()

        has_vacuum = False
        for i in range(3):
            # Only check dimensions that are periodic
            if not atoms.pbc[i]:
                continue

            L = cell_lengths[i]
            # Use ~2.0 A bins, but at least 5 bins if L is large enough to matter
            nbins = max(5, int(L / 2.0))
            if nbins < 2:
                continue

            # Histogram of coordinates along axis i
            # Use range=(0, L) to cover full cell
            hist, _ = np.histogram(positions[:, i], bins=nbins, range=(0, L))

            if np.any(hist == 0):
                has_vacuum = True
                break

        if has_vacuum:
            # Skip density check if vacuum is detected (likely a slab or porous material)
            return True

        # Calculate Density
        try:
            vol = atoms.get_volume()
        except ValueError:
            logger.warning("Validation Failed: Could not calculate volume.")
            return False

        if vol < 1e-9:
             logger.warning(f"Validation Failed: Volume {vol} is too small.")
             return False

        total_mass = sum(atoms.get_masses()) # amu

        # Calculate conversion factor using ASE units
        # 1 amu = _amu kg = _amu * 1000 g
        # 1 Angstrom^3 = 1e-24 cm^3
        AMU_TO_G = _amu * 1.0e3
        A3_TO_CM3 = 1.0e-24
        conversion_factor = AMU_TO_G / A3_TO_CM3

        density = (total_mass / vol) * conversion_factor

        if density < self.constraints.min_density:
             logger.warning(f"Validation Failed: Density {density:.2f} < {self.constraints.min_density}")
             return False

        return True
