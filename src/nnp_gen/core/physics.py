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

def estimate_lattice_constant(elements: List[str], structure: str = "fcc") -> float:
    """
    Estimates the lattice constant for an alloy using Vegard's Law (volume averaging).
    
    Args:
        elements (List[str]): List of element symbols.
        structure (str): Target crystal structure ('fcc', 'bcc', 'sc').
    
    Returns:
        float: Estimated lattice constant in Angstroms.
    """
    from ase.data import atomic_numbers, reference_states, covalent_radii

    lattice_constants = []
    
    for el in elements:
        el = el.capitalize()
        Z = atomic_numbers.get(el)
        if Z is None:
            continue
            
        ref = reference_states[Z]
        
        v_atom = None
        
        # 1. Try to get atomic volume from reference state
        if ref is not None:
            sym = ref.get('symmetry')
            a_ref = ref.get('a')
            
            if sym and a_ref:
                if sym == 'fcc':
                    v_atom = (a_ref ** 3) / 4.0
                elif sym == 'bcc':
                    v_atom = (a_ref ** 3) / 2.0
                elif sym == 'sc':
                    v_atom = (a_ref ** 3)
                elif sym == 'diamond':
                    v_atom = (a_ref ** 3) / 8.0
        
        # 2. Fallback: Use Covalent Radius to estimate volume
        if v_atom is None:
             r = covalent_radii[Z]
             # Estimate based on hard sphere packing roughly 74% (FCC/HCP)
             # V_cell_fcc = (r * 2 * sqrt(2))^3 = 22.6 * r^3
             # V_atom = V_cell / 4 = 5.65 * r^3
             # Sphere vol = 4/3 pi r^3 = 4.18 r^3
             # Let's approximate V_atom approx (2*r)^3 / sqrt(2) for FCC-like packing
             v_atom = (4/3 * np.pi * (r ** 3)) / 0.74

        # 3. Convert Atomic Volume to Target Lattice Constant
        if structure == 'fcc':
            a_target = (v_atom * 4.0) ** (1/3)
        elif structure == 'bcc':
            a_target = (v_atom * 2.0) ** (1/3)
        elif structure == 'sc':
            a_target = v_atom ** (1/3)
        else:
            # Default to FCC assumption if unknown
            a_target = (v_atom * 4.0) ** (1/3)
            
        lattice_constants.append(a_target)

    if not lattice_constants:
        return 3.6 # Generic fallback
        
    return float(np.mean(lattice_constants))

def detect_vacuum(atoms: Atoms, threshold: float = 5.0) -> bool:
    """
    Detect if the system has vacuum (is a slab, wire, or cluster) or is fully bulk.

    Logic:
    1. If any PBC is False -> Vacuum (Slab/Wire/Cluster).
    2. If all PBC are True -> Check for gaps in atomic distribution along axes.
       Projects positions onto each axis and checks for empty space > threshold.

    Args:
        atoms (Atoms): Structure to check.
        threshold (float): Minimum gap size in Angstroms to consider as vacuum.

    Returns:
        bool: True if vacuum is detected.
    """
    if any(not p for p in atoms.pbc):
        return True

    # Histogram check for bulk vacuum (e.g. a slab manually created inside a periodic box)
    # We check each dimension by looking at gaps in sorted coordinates.

    positions = atoms.get_positions()
    cell = atoms.get_cell()

    for i in range(3):
        L = np.linalg.norm(cell[i])

        # If cell dimension is smaller than threshold, it can't contain a vacuum of that size
        if L < threshold:
            continue

        coords = positions[:, i]
        # Sort coordinates
        coords_sorted = np.sort(coords)

        # Calculate gaps
        gaps = np.diff(coords_sorted)
        max_internal_gap = np.max(gaps) if len(gaps) > 0 else 0

        # Check periodic gap (wrap around)
        # Gap between last atom and first atom across boundary
        # Note: positions are not wrapped, so last - first could be large if atoms are spread out,
        # but we care about the empty space in the periodic box.
        # Ideally, we map to [0, L] first.
        # But even simpler: L - (max - min) roughly if one cluster? No.

        # Correct Periodic Gap: (L - coords_sorted[-1]) + coords_sorted[0]
        # This assumes coords are in [0, L]. If they are outside, we should wrap them first.
        # But wrapping might split a molecule.

        # Let's rely on standard ASE wrap to be safe
        # Create a copy to not modify original
        temp_atoms = atoms.copy()
        temp_atoms.wrap()
        wrapped_coords = np.sort(temp_atoms.positions[:, i])

        wrapped_gaps = np.diff(wrapped_coords)
        max_wrapped_gap = np.max(wrapped_gaps) if len(wrapped_gaps) > 0 else 0

        periodic_gap = (L - wrapped_coords[-1]) + wrapped_coords[0]

        if max(max_wrapped_gap, periodic_gap) > threshold:
            return True

    return False

def apply_vacancies(atoms: Atoms, concentration: float, rng: np.random.RandomState) -> Atoms:
    """
    Randomly remove atoms to create vacancies.

    Args:
        atoms (Atoms): Structure to modify.
        concentration (float): Fraction of atoms to remove (0.0 to 1.0).
        rng (np.random.RandomState): Random number generator.

    Returns:
        Atoms: Modified structure (in-place usually, but returned).
    """
    if concentration <= 0.0:
        return atoms

    n_atoms = len(atoms)
    n_vacancies = int(n_atoms * concentration)

    if n_vacancies == 0:
        return atoms

    if n_vacancies >= n_atoms:
        # Should not happen given constraints, but safety check
        return atoms

    # Select indices to remove
    indices_to_remove = rng.choice(n_atoms, size=n_vacancies, replace=False)

    # Sort indices in descending order to avoid index shift issues during pop (if looping)
    # But ASE del atoms[indices] works if list/array provided?
    # ASE del atoms[[1, 5]] works.

    del atoms[indices_to_remove]

    return atoms
