import numpy as np
from typing import Dict, Tuple, List, Optional
from ase import Atoms
from math import ceil

def apply_rattle(atoms: Atoms, std: float) -> Atoms:
    """
    Apply Gaussian noise to atomic positions.

    Args:
        atoms (Atoms): The atoms object to modify.
        std (float): Standard deviation of the Gaussian noise in Angstroms.

    Returns:
        Atoms: The modified atoms object (in-place modification but returned for chaining).
    """
    atoms.rattle(stdev=std, seed=None)
    return atoms

def apply_volumetric_strain(atoms: Atoms, scale_range: List[float]) -> Atoms:
    """
    Apply isotropic strain to the unit cell.

    Args:
        atoms (Atoms): The atoms object to modify.
        scale_range (List[float]): [min, max] scaling factors (linear scale).

    Returns:
        Atoms: The modified atoms object.
    """
    if len(scale_range) != 2:
        raise ValueError("scale_range must be a list of 2 floats [min, max]")

    s = np.random.uniform(scale_range[0], scale_range[1])

    # Check if cell exists and is not zero
    if atoms.cell is None or np.all(atoms.cell == 0):
        # Can't scale zero cell or non-periodic without cell?
        # But instructions say use set_cell(cell * s).
        # If no cell, this operation is meaningless or effectively scales positions if scale_atoms=True?
        # ASE set_cell with scale_atoms=True on vacuum structure just scales positions?
        pass
    else:
        new_cell = atoms.get_cell() * s
        atoms.set_cell(new_cell, scale_atoms=True)

    return atoms

def set_initial_magmoms(atoms: Atoms, magmom_map: Dict[str, float]) -> Atoms:
    """
    Set initial magnetic moments based on element map.

    Args:
        atoms (Atoms): The atoms object.
        magmom_map (Dict[str, float]): Map of element symbol to magnetic moment.

    Returns:
        Atoms: The modified atoms object.
    """
    if not magmom_map:
        return atoms

    magmoms = []
    for atom in atoms:
        mag = magmom_map.get(atom.symbol, 0.0)
        magmoms.append(mag)

    atoms.set_initial_magnetic_moments(magmoms)
    return atoms

def ensure_supercell_size(atoms: Atoms, r_cut: float, factor: float = 2.0) -> Atoms:
    """
    Expand the unit cell to be at least r_cut * factor in all dimensions.

    Args:
        atoms (Atoms): The atoms object.
        r_cut (float): Cutoff radius.
        factor (float): Factor to multiply r_cut by (default 2.0).

    Returns:
        Atoms: The expanded supercell.
    """
    # Only applicable for periodic systems
    if not any(atoms.pbc):
        return atoms

    l_min = r_cut * factor
    cell_lengths = atoms.cell.lengths()

    repeat = [1, 1, 1]
    for i in range(3):
        if atoms.pbc[i]:
            if cell_lengths[i] < 1e-6:
                # Avoid division by zero, though unlikely for valid crystal
                repeat[i] = 1
            else:
                repeat[i] = max(1, int(ceil(l_min / cell_lengths[i])))

    # If repeat is [1,1,1], make_supercell returns a copy usually.
    # We want to return a new object to be safe.
    if repeat == [1, 1, 1]:
        return atoms.copy()

    # make_supercell returns a new Atoms object
    # Note: ase.build.make_supercell or atoms * repeat
    # atoms * repeat works
    supercell = atoms * repeat
    return supercell
