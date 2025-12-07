import logging
import random
import numpy as np
from ase import Atoms
from ase import units
from typing import List, Tuple, Optional
from nnp_gen.core.config import MonteCarloConfig, MCStrategy
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

logger = logging.getLogger(__name__)

def _check_hard_sphere_overlap(atoms: Atoms, indices_to_check: List[int], tolerance: float = 0.6) -> bool:
    """
    Checks if the specified atoms are too close to any other atoms.
    Uses covalent radii sum * tolerance as the threshold.
    Returns True if overlap detected (unsafe).
    """
    try:
        # We only need to check distances for the moved atoms
        # neighbor_list can be slow for large systems, but robust for PBC
        # 'd' returns distances

        # Optimization: We only care if ANY distance < threshold
        # We can iterate over indices_to_check

        positions = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        numbers = atoms.get_atomic_numbers()

        for i in indices_to_check:
            # Simple check against all others (O(N) per moved atom)
            # For very large N, neighbor list is better. For N < 1000, this is fine.
            # But let's use ASE's built-in tools which handle PBC correctly.

            # Using neighbor_list for just one atom is not direct API.
            # Let's use get_distances()

            # Get distances from atom i to all others
            # mic=True applies Minimum Image Convention (PBC)
            dists = atoms.get_distances(i, range(len(atoms)), mic=True)

            r_i = covalent_radii[numbers[i]]

            for j, d in enumerate(dists):
                if i == j: continue

                r_j = covalent_radii[numbers[j]]
                limit = (r_i + r_j) * tolerance

                if d < limit:
                    # logger.debug(f"Overlap detected: {i}-{j} d={d:.2f} < {limit:.2f}")
                    return True
        return False

    except Exception as e:
        logger.warning(f"Overlap check failed: {e}")
        return True # Fail safe

def perform_mc_swap(atoms: Atoms, mc_config: MonteCarloConfig, temp: float, calc) -> bool:
    """
    Perform a Monte Carlo swap or vacancy hop.
    Returns True if accepted, False otherwise.

    Args:
        atoms: The ASE Atoms object.
        mc_config: The Monte Carlo configuration.
        temp: Temperature in Kelvin.
        calc: The calculator to use for energy evaluation (already attached to atoms, but passed for clarity/mocking).

    Returns:
        bool: True if the move was accepted.
    """
    if not mc_config.enabled:
        return False

    if temp <= 0:
        return False

    # Choose Strategy
    strategy = random.choice(mc_config.strategy)

    # Calculate initial energy
    try:
        e_old = atoms.get_potential_energy()
    except Exception as e:
        logger.warning(f"MC Energy calculation failed: {e}")
        return False

    accepted = False

    indices_changed = []
    original_positions = []
    original_symbols = []

    # --- STRATEGY A: VACANCY HOP (Smart Rattle) ---
    if strategy == MCStrategy.VACANCY_HOP:
        # Pick random atom
        idx = random.randint(0, len(atoms) - 1)

        # Large displacement (2.5 A) in random direction
        # Why 2.5? Typical interatomic distance. Jumping to a neighbor site (void).
        disp_mag = 2.5
        direction = np.random.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return False # Fail safe
        direction /= norm
        displacement = direction * disp_mag

        # Store old state
        indices_changed = [idx]
        original_positions = [atoms.positions[idx].copy()]

        # Apply move
        atoms.positions[idx] += displacement

        # Safety: Geometric Overlap Check
        if _check_hard_sphere_overlap(atoms, [idx]):
            logger.info("Vacancy Hop rejected due to hard sphere overlap.")
            atoms.positions[idx] = original_positions[0] # Revert immediately
            return False

    # --- STRATEGY B: SWAP ---
    elif strategy == MCStrategy.SWAP:
        if not mc_config.swap_pairs:
            return False

        # Pick a pair type
        pair_type = random.choice(mc_config.swap_pairs)
        el1, el2 = pair_type

        # Find indices
        indices1 = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == el1]
        indices2 = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == el2]

        if not indices1 or not indices2:
            return False # Cannot swap

        idx1 = random.choice(indices1)
        idx2 = random.choice(indices2)

        if idx1 == idx2:
            return False

        # Safety Check: Charge
        if atoms.has("initial_charges"):
            charges = atoms.get_initial_charges()
            q1 = charges[idx1]
            q2 = charges[idx2]

            if abs(q1 - q2) > 0.1 and not mc_config.allow_charge_mismatch:
                logger.warning(f"Aliovalent swap rejected: {el1}({q1}) <-> {el2}({q2})")
                return False

        # Apply Swap
        indices_changed = [idx1, idx2]
        original_symbols = [atoms.symbols[idx1], atoms.symbols[idx2]]

        # Swap symbols
        atoms.symbols[idx1], atoms.symbols[idx2] = atoms.symbols[idx2], atoms.symbols[idx1]

        if atoms.has("initial_charges"):
            q = atoms.get_initial_charges()
            q[idx1], q[idx2] = q[idx2], q[idx1]
            atoms.set_initial_charges(q)

        if atoms.has("initial_magnetic_moments"):
            m = atoms.get_initial_magnetic_moments()
            m[idx1], m[idx2] = m[idx2], m[idx1]
            atoms.set_initial_magnetic_moments(m)

    # --- METROPOLIS ---
    try:
        e_new = atoms.get_potential_energy()
        delta_e = e_new - e_old

        if delta_e < 0:
            accepted = True
        else:
            # Boltzmann
            k_b = units.kB # eV/K
            prob = np.exp(-delta_e / (k_b * temp))
            if random.random() < prob:
                accepted = True

    except Exception as e:
        logger.warning(f"MC New Energy calculation failed: {e}")
        accepted = False

    if accepted:
        logger.info(f"MC {strategy} Accepted. dE = {delta_e:.4f} eV")
        return True
    else:
        # Revert
        if strategy == MCStrategy.VACANCY_HOP:
            atoms.positions[indices_changed[0]] = original_positions[0]
        elif strategy == MCStrategy.SWAP:
            # Revert symbols
            atoms.symbols[indices_changed[0]] = original_symbols[0]
            atoms.symbols[indices_changed[1]] = original_symbols[1]
            # Revert arrays
            if atoms.has("initial_charges"):
                q = atoms.get_initial_charges()
                q[indices_changed[0]], q[indices_changed[1]] = q[indices_changed[1]], q[indices_changed[0]]
                atoms.set_initial_charges(q)
            if atoms.has("initial_magnetic_moments"):
                m = atoms.get_initial_magnetic_moments()
                m[indices_changed[0]], m[indices_changed[1]] = m[indices_changed[1]], m[indices_changed[0]]
                atoms.set_initial_magnetic_moments(m)

        return False
