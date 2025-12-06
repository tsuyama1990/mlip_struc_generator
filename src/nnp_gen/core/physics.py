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
    # New lattice vectors = Old lattice vectors * F^T
    new_cell = atoms.cell.array @ F.T
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

    # Calculate cell parameters (lengths and angles) or use cell perpendicular widths
    # For orthogonal cells, lengths are enough. For triclinic, we need perpendicular widths.
    # ASE doesn't have a direct "perpendicular width" function easily accessible for all cases without geometry.
    # But we can iterate.

    # Logic:
    # 1. Start with repeat=[1,1,1]
    # 2. Check if current supercell has all perpendicular widths >= l_min
    # 3. If not, increase repeat in the deficient direction
    # This might be iterative.

    # Simplified approach:
    # Calculate heights of the parallelepiped.
    # h_i = Volume / Area_i
    # This works for general cells.

    cell = atoms.cell
    vol = abs(atoms.get_volume())

    # Calculate cross products of vectors to get areas of faces
    # v0, v1, v2
    v = cell.array

    # area_0 (defined by v1, v2) = |v1 x v2|
    # area_1 (defined by v0, v2) = |v0 x v2|
    # area_2 (defined by v0, v1) = |v0 x v1|

    repeat = [1, 1, 1]

    # Pre-check simple lengths first to get a baseline (fast)
    lengths = cell.lengths()
    for i in range(3):
        if atoms.pbc[i] and lengths[i] > 1e-6:
             repeat[i] = max(1, int(ceil(l_min / lengths[i])))

    # Refine for skewed cells
    # We apply the current repeat and check widths

    # We do a small loop to ensure convergence (usually 1 pass is enough unless very skewed)
    for _ in range(3):
        current_cell = (atoms.cell.array.T * np.array(repeat)).T # Scale vectors
        current_vol = np.abs(np.linalg.det(current_cell))

        # Recalculate widths
        # Width 0: distance between faces spanned by v1, v2.
        # w0 = Vol / |v1 x v2|
        # etc.

        v = current_cell

        # areas
        areas = [
            np.linalg.norm(np.cross(v[1], v[2])), # area normal to v0
            np.linalg.norm(np.cross(v[0], v[2])), # area normal to v1
            np.linalg.norm(np.cross(v[0], v[1]))  # area normal to v2
        ]

        converged = True
        for i in range(3):
            if atoms.pbc[i]:
                if areas[i] > 1e-6:
                    w = current_vol / areas[i]
                else:
                    w = 0.0

                # Protection against division by zero if w is very small
                if w < 1e-9:
                     # Degenerate cell or very flat. Cannot expand properly based on volume.
                     # Just assume it's bad and maybe try to expand but scale would be infinite.
                     # We skip expansion update to avoid crash/inf.
                     continue

                if w < l_min - 1e-4: # epsilon tolerance
                    # We need to increase repeat[i]
                    # Scaling factor needed: l_min / w
                    # We multiply current repeat by this factor
                    # new_repeat = old_repeat * (l_min / w)
                    # But we only increment by integer steps.
                    scale = l_min / w
                    repeat[i] = int(ceil(repeat[i] * scale))
                    converged = False

        if converged:
            break

    if repeat == [1, 1, 1]:
        return atoms.copy()

    supercell = atoms * repeat
    return supercell
