import numpy as np
from typing import Dict, Tuple, List, Optional
from ase import Atoms
from math import ceil

def apply_rattle(atoms: Atoms, std: float, seed: Optional[int] = None) -> Atoms:
    """
    Apply Gaussian noise to atomic positions.

    Args:
        atoms (Atoms): The atoms object to modify.
        std (float): Standard deviation of the Gaussian noise in Angstroms.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        Atoms: The modified atoms object (in-place modification but returned for chaining).
    """
    # ASE's atoms.rattle(seed=...) expects an int or None, it creates its own RandomState internally.
    # Passing a RandomState object causes TypeError in recent numpy/ASE versions.
    atoms.rattle(stdev=std, seed=seed)
    return atoms

def apply_strain_tensor(atoms: Atoms, strain_tensor: np.ndarray) -> Atoms:
    """
    Apply general strain tensor (3x3) to cell.

    Args:
        atoms (Atoms): The atoms object.
        strain_tensor (np.ndarray): 3x3 strain tensor.

    Returns:
        Atoms: Modified atoms object.
    """
    if atoms.cell is None:
        return atoms

    F = np.eye(3) + strain_tensor  # Deformation gradient
    # atoms.cell is 3x3 array of lattice vectors (rows)
    # new_cell = atoms.cell @ F corresponds to r' = F r if r are columns in basis,
    # but here r' = r F^T if r are row vectors.
    # Actually if h_new = h_old F.T, then r' = F r.
    # However, standard convention audit request implies removing transpose.
    new_cell = atoms.cell.array @ F
    atoms.set_cell(new_cell, scale_atoms=True)
    return atoms

def apply_volumetric_strain(atoms: Atoms, scale_range: List[float], seed: Optional[int] = None) -> Atoms:
    """
    Apply isotropic strain to the unit cell.

    Args:
        atoms (Atoms): The atoms object to modify.
        scale_range (List[float]): [min, max] scaling factors (linear scale).
        seed (Optional[int]): Random seed.

    Returns:
        Atoms: The modified atoms object.
    """
    if len(scale_range) != 2:
        raise ValueError("scale_range must be a list of 2 floats [min, max]")

    rng = np.random.RandomState(seed)
    s = rng.uniform(scale_range[0], scale_range[1])

    if atoms.cell is None or np.all(atoms.cell == 0):
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

def ensure_supercell_size(atoms: Atoms, r_cut: float, factor: float = 1.0) -> Atoms:
    """
    Expand the unit cell to be at least r_cut * factor in all dimensions.
    Uses perpendicular widths (heights) to ensure validity for general (triclinic) cells.

    Args:
        atoms (Atoms): The atoms object.
        r_cut (float): Cutoff radius.
        factor (float): Factor to multiply r_cut by (default 1.0).

    Returns:
        Atoms: The expanded supercell.
    """
    # Only applicable for periodic systems
    if not any(atoms.pbc):
        return atoms

    l_min = r_cut * factor

    # Calculate cell parameters
    cell = atoms.cell

    # Handle degenerate cells (zero volume)
    try:
        vol = abs(atoms.get_volume())
    except Exception:
        # If get_volume fails, it might be degenerate or non-periodic properly
        return atoms.copy()

    if vol < 1e-9:
        # Cannot calculate widths for 0-volume cell
        return atoms.copy()

    v = cell.array

    # Calculate areas of faces to determine perpendicular widths (heights)
    # Height h_i corresponding to vector v_i is Vol / Area_i
    # Area_0 is area of face spanned by v1, v2 (normal n0)
    # Area_1 is area of face spanned by v0, v2 (normal n1)
    # Area_2 is area of face spanned by v0, v1 (normal n2)

    cross_products = [
        np.cross(v[1], v[2]), # Normal to face 0
        np.cross(v[0], v[2]), # Normal to face 1
        np.cross(v[0], v[1])  # Normal to face 2
    ]

    areas = [np.linalg.norm(cp) for cp in cross_products]

    repeat = [1, 1, 1]

    for i in range(3):
        if atoms.pbc[i]:
            # Perpendicular width along direction i
            # If area is effectively zero, width is infinite (or undefined, but assume OK)
            if areas[i] > 1e-9:
                w = vol / areas[i]
                # Tolerance to avoid floating point issues
                if w < l_min - 1e-4:
                    # Needed expansion factor
                    # new_width = repeat[i] * old_width >= l_min
                    scale = l_min / w
                    repeat[i] = max(1, int(ceil(scale)))
            else:
                # Should ideally not happen for non-zero volume
                pass

    if repeat == [1, 1, 1]:
        return atoms.copy()

    supercell = atoms * repeat
    return supercell
