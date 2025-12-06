
import logging
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.units import _amu
from nnp_gen.core.config import PhysicsConstraints

logger = logging.getLogger(__name__)

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

        # Use neighbor_list for O(N) scaling
        # We only care if ANY distance is < min_dist_threshold.
        # So we can set cutoff = min_dist_threshold.
        # If any neighbor is found within this cutoff (excluding self), it's a violation.

        try:
            # neighbor_list returns (i, j, d, ...)
            # 'd' is distances.
            # We use cutoff slightly larger than threshold to be safe or exactly threshold?
            # If we set cutoff=min_dist, neighbor_list finds atoms with d < cutoff.
            # Wait, neighbor_list finds atoms within cutoff radius.
            # So if we find ANY pair, it's invalid.

            # self_interaction=False means we don't get i-i pairs.
            # But we get i-j and j-i.

            # Using neighbor_list is much faster for large systems than get_all_distances.
            # However, neighbor_list accounts for PBC automatically based on atoms.pbc.

            # Check for non-periodic axes?
            # neighbor_list respects atoms.pbc.

            # Note: neighbor_list returns 1-D arrays.
            # If any distance is returned, it means it is <= cutoff.

            # To be strictly > min_dist, we need to check if min(d) < min_dist.
            # If we set cutoff = min_dist, and we find neighbors, then we have d <= min_dist.
            # If d == min_dist, is it valid? "min_distance" usually means strictly greater?
            # Or >=? Usually we want d >= min_dist. So if d < min_dist it is bad.

            # Let's set cutoff = min_dist - epsilon? No.
            # If we set cutoff = min_dist, we find all pairs with d < min_dist.
            # Actually neighbor_list uses strict inequality? Or <=?
            # ASE docs: "The cutoff radius". Neighbors are atoms within this radius.

            # Let's use get_distances approach if N is small? No, O(N^2).
            # Let's use neighbor_list.

            i, j, d = neighbor_list('ijd', atoms, cutoff=min_dist_threshold, self_interaction=False)

            if len(d) > 0:
                # We found pairs shorter than cutoff
                min_found = np.min(d)
                # Double check against threshold just to be sure about boundary conditions
                if min_found < min_dist_threshold:
                    logger.warning(f"Validation Failed: Min distance {min_found:.2f} < {min_dist_threshold}")
                    return False

            return True

        except Exception as e:
            # Catch unexpected errors but log them properly.
            # Ideally we re-raise or handle specific errors.
            # But for validation, if we can't validate, we should probably fail safe (reject).
            logger.error(f"Validation Error during distance check: {e}")
            return False

    def _check_min_density(self, atoms: Atoms) -> bool:
        # Skip density check for non-periodic systems (molecules/clusters)
        if not any(atoms.pbc):
            return True

        # Also skip density check if the system has vacuum (is a slab)
        # Slabs typically have vacuum in the Z direction (pbc[2]=True often, but empty space)
        # Heuristic: Check if vacuum layers exist or if pbc suggests 2D?
        # But standard slabs have pbc=[True, True, True].
        # User prompt says: "If system is 2D (has vacuum), skip the bulk density check"
        # How to detect "has vacuum"?
        # One way is to check the config type if we had access to it, but here we only have atoms.
        # However, we can check if there's a large empty region in Z?
        # Or we can relax density check if density is suspiciously low but structure is valid otherwise?
        # But we don't want to accept exploded bulk.

        # Best approach given the architecture:
        # If we can't access config type here, maybe we should trust the generator to not produce bad density
        # unless it's a slab?
        # BUT, the prompt explicitly says: "Update validate_structure in validation.py. Logic: If system is 2D (has vacuum), skip the bulk density check or use a modified "Planar Density" check."

        # How to detect 2D/Slab from 'atoms' object?
        # A common heuristic for slab in 3D PBC: Large empty space in one direction.
        # Let's check vacuum fraction in Z?
        # Or simply check if the calculated density is low but we are in a 'valid low density' regime?
        # That's risky.

        # Better: Check if atoms.info has a flag?
        # We can set atoms.info['system_type'] in the generator.
        # But let's look for a purely geometric heuristic as requested if possible.
        # "If system is 2D (has vacuum)"

        # Let's check for vacuum gap in Z direction > 5A?
        positions = atoms.get_positions()
        if len(positions) > 0:
            z_coords = np.sort(positions[:, 2])
            z_diffs = np.diff(z_coords)
            # Check for large gap (vacuum) in sorted Z coordinates, taking PBC into account (gap between top and bottom image)
            # cell_z = atoms.cell[2, 2]
            # gap_pbc = cell_z - (z_coords[-1] - z_coords[0])

            # If gap_pbc > 5.0 (arbitrary vacuum size), it's likely a slab.
            # Let's use 5.0A as a threshold for vacuum detection.

            cell_z = atoms.get_cell()[2][2]
            if z_coords.size > 1:
                vacuum_size = cell_z - (z_coords[-1] - z_coords[0])
                if vacuum_size > 5.0:
                    # Likely a slab/surface with vacuum
                    return True

        # Density check
        try:
            vol = atoms.get_volume()
        except ValueError:
            # Typically happens if cell is not defined or degenerate
            # For periodic systems, this is a failure.
            logger.warning("Validation Failed: Could not calculate volume for periodic system.")
            return False

        if vol < 1e-6:
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
