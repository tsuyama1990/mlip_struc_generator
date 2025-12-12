import numpy as np
from typing import Dict, Tuple, List, Optional
from ase import Atoms
from math import ceil
from scipy.spatial import KDTree
from scipy.ndimage import label
from ase.data import covalent_radii
from nnp_gen.core.exceptions import GenerationError

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

def estimate_lattice_constant(elements: List[str], structure: str = "fcc", method: str = "max") -> float:
    """
    Estimates the lattice constant for an alloy.
    
    Args:
        elements (List[str]): List of element symbols.
        structure (str): Target crystal structure ('fcc', 'bcc', 'sc').
        method (str): Aggregation method. 
                      'mean' = Vegard's Law (average). Good for dense packing, bad for size mismatch.
                      'max' = Use largest atom. Safe against overlap, but lower density.
    
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
        raise GenerationError(f"Unable to estimate lattice constant for elements: {elements}")
        
    if method == "max":
        return float(np.max(lattice_constants))
    else:
        # Default to mean (Vegard's Law)
        return float(np.mean(lattice_constants))

def detect_vacuum(atoms: Atoms, threshold: float = 5.0) -> bool:
    """
    Detect if the system has vacuum (is a slab, wire, or cluster) or is fully bulk.

    Uses Grid-Based Void Detection with NeighborList (PBC-aware):
    1. Grid spacing ~1.0 A.
    2. Create grid points covering the cell.
    3. Use neighbor list to check distance from grid points to any atom.
    4. If any grid point is further than (threshold/2) from all atoms, it's a void.
    """
    if any(not p for p in atoms.pbc):
        return True

    # Setup Grid
    lengths = atoms.cell.lengths()
    grid_spacing = 1.0 # Angstrom

    # Grid dimensions
    ngrid = np.ceil(lengths / grid_spacing).astype(int)
    ngrid = np.maximum(ngrid, 1)

    # Generate grid points in scaled coords
    x = np.linspace(0, 1, ngrid[0], endpoint=False)
    y = np.linspace(0, 1, ngrid[1], endpoint=False)
    z = np.linspace(0, 1, ngrid[2], endpoint=False)

    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    scaled_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)

    # Convert to Cartesian
    grid_points = atoms.cell.cartesian_positions(scaled_points)
    
    # Create Dummy Atoms for the grid
    # We use a dummy symbol 'H' (doesn't matter)
    grid_atoms = Atoms('H' * len(grid_points), positions=grid_points, cell=atoms.cell, pbc=atoms.pbc)
    
    # Combine real atoms and grid atoms
    # Real atoms are indices 0 to N-1
    # Grid atoms are indices N to N+M-1
    combined = atoms.copy()
    combined += grid_atoms
    
    n_real = len(atoms)

    # Setup NeighborList
    # We want to know if a grid point has ANY neighbor within cutoff.
    # cutoff = threshold / 2.0. If no neighbor within this cutoff, it's a large void.
    # Actually, to reproduce "Largest Empty Sphere" > threshold/2 logic:
    # We need the distance to the nearest atom.
    # If nearest_dist > threshold/2, then vacuum exists.

    # Using neighbor_list from ase
    from ase.neighborlist import neighbor_list

    # We need to query distances between grid_atoms and atoms.
    # neighbor_list('d', combined, cutoff) returns all pairs.
    # We filter pairs where (i < n_real AND j >= n_real) or vice versa.

    # Optimization: Use a cutoff slightly larger than threshold/2 to be safe?
    # No, we want to know if there exists a point with dist > threshold/2.
    # So if we query with cutoff = threshold/2, and a point has NO neighbors,
    # then it is far from all atoms.

    # However, atoms have radii. The "void" definition usually accounts for atomic radii.
    # The original code used: max_void_radius > threshold/2.
    # And distance check was to atom centers? Yes, KDTree on positions.
    # Wait, original KDTree approach used `dists` which are distances to centers.
    # So "void size" in original code implies distance to center.
    # So we stick to distance to center.

    cutoff = threshold / 2.0

    # Get indices of neighbors
    # 'i' are indices of first atom, 'j' are indices of second atom.
    # We want to find grid points (indices >= n_real) that are NOT in the neighbor list of any real atom.

    # We only care about interactions between Real (0..n_real-1) and Grid (n_real..end)
    # But neighbor_list computes all.
    # We can use update() but neighbor_list function is easier.

    i_indices, j_indices = neighbor_list('ij', combined, cutoff)

    # We are looking for grid points (index g) such that there is NO (r, g) pair with dist < cutoff.
    # In neighbor_list output, we look for j where i < n_real.

    # Filter for pairs between Real and Grid
    # Because bothways=True (default implied? No, neighbor_list defaults? check doc)
    # ase.neighborlist.neighbor_list(quantities, a, cutoff, self_interaction=False)
    # It returns both i-j and j-i.

    # Find all grid indices that HAVE a neighbor
    connected_grid_indices = set()

    # Mask for interactions between Real (i < n_real) and Grid (j >= n_real)
    mask = (i_indices < n_real) & (j_indices >= n_real)
    connected_grid_indices.update(j_indices[mask])

    # Also check j < n_real and i >= n_real (symmetry)
    mask2 = (i_indices >= n_real) & (j_indices < n_real)
    connected_grid_indices.update(i_indices[mask2])

    # Normalize grid indices to 0..n_grid-1
    # grid index in combined is k. Normalized is k - n_real.

    num_grid_points = len(grid_atoms)

    # If the number of connected grid points is less than total grid points,
    # then there is at least one grid point with no neighbors within cutoff.
    # That point is a "void center" with radius > threshold/2.

    # Adjust for set containing global indices
    # We need to count how many unique grid indices are covered.

    covered_count = 0
    for idx in connected_grid_indices:
        if idx >= n_real:
            covered_count += 1

    if covered_count < num_grid_points:
        return True # Found a void

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
