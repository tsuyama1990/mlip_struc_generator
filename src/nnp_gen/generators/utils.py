
import logging
import numpy as np
from typing import List, Optional
from ase import Atoms
from ase.build import surface
from ase.optimize import FIRE
from nnp_gen.core.zbl import ZBLCalculator

logger = logging.getLogger(__name__)

def generate_random_surfaces(
    base_structure: Atoms, 
    n_samples: int, 
    rng: np.random.RandomState, 
    min_thickness_angstrom: float = 8.0,
    vacuum: float = 10.0,
    source_prefix: str = "surface",
    max_atoms: Optional[int] = None
) -> List[Atoms]:
    """
    Generates random surface slabs from a bulk structure.

    Args:
        base_structure (Atoms): The bulk structure.
        n_samples (int): Number of surfaces to generate.
        rng (np.random.RandomState): Random number generator.
        min_thickness_angstrom (float): Minimum slab thickness in Angstrom (used to pad lateral size).
            Actually used for lateral (xy) repetition to avoid thin cells, 
            AND also ensures z-thickness via layers selection? 
            Currently implementation picks layers=3-6.
        vacuum (float): Vacuum padding in Angstrom.
        source_prefix (str): Prefix for info['config_source'].
        max_atoms (Optional[int]): Upper limit on the number of atoms.

    Returns:
        List[Atoms]: Generated surface slabs.
    """
    structures = []
    
    # Low-index planes to sample from
    # (111), (110), (100) are standard for cubic. 
    # (001), (100) for others.
    # For general systems, we stick to low indices.
    candidate_millers = [(1, 1, 1), (1, 1, 0), (1, 0, 0), (2, 1, 1), (2, 1, 0), (0, 0, 1)]
    candidate_layers = [3, 4, 5, 6]
    
    attempts = 0
    max_attempts = n_samples * 5  # Limit attempts to avoid infinite loops
    
    while len(structures) < n_samples and attempts < max_attempts:
        attempts += 1
        try:
            # Randomly pick Miller and Layers
            idx_m = rng.randint(0, len(candidate_millers))
            miller = candidate_millers[idx_m]
            
            layers = rng.choice(candidate_layers)
            
            # Cut surface
            # Note: ase.build.surface might fail for weird cells or non-standard orientations.
            slab = surface(base_structure, miller, layers=layers, vacuum=vacuum)
            slab.center(vacuum=vacuum, axis=2)
            
            # Expand lateral supercell for surfaces so they aren't too thin
            # Check vectors 0 and 1 (surface plane)
            cell = slab.get_cell()
            nx = int(np.ceil(min_thickness_angstrom / np.linalg.norm(cell[0]))) if np.linalg.norm(cell[0]) > 1e-3 else 1
            ny = int(np.ceil(min_thickness_angstrom / np.linalg.norm(cell[1]))) if np.linalg.norm(cell[1]) > 1e-3 else 1
            
            if nx > 1 or ny > 1:
                slab *= (nx, ny, 1)
                
            # Check max_atoms constraint
            if max_atoms is not None and len(slab) > max_atoms:
                # Heuristic: try to reduce lateral expansion if we exceeded limit
                if nx > 1 or ny > 1:
                     # Try minimal 1x1 first? Or just reduce logic?
                     # Let's try to scale down.
                     # If we went 3x3, maybe 2x2 fits. 
                     # For now, just simplistic fallback: don't expand
                     base_slab = surface(base_structure, miller, layers=layers, vacuum=vacuum)
                     base_slab.center(vacuum=vacuum, axis=2)
                     if len(base_slab) <= max_atoms:
                         slab = base_slab
                     else:
                         # Even minimal slab is too big
                         logger.debug(f"Skipping surface {miller} with {len(slab)} atoms (> {max_atoms})")
                         continue
                else:
                     logger.debug(f"Skipping surface {miller} with {len(slab)} atoms (> {max_atoms})")
                     continue

            slab.info['config_source'] = f"{source_prefix}_{miller}"
            
            # Preserve parent info
            if 'target_composition' in base_structure.info:
                slab.info['target_composition'] = base_structure.info['target_composition']
            if 'generated_lattice_constant' in base_structure.info:
                slab.info['generated_lattice_constant'] = base_structure.info['generated_lattice_constant']

            structures.append(slab)
            
        except Exception as e:
            # Common failure: "Input atoms must have a unit cell" (if pbc=False), or linear dep.
            # Or if standard surface construction fails for complex cell.
            logger.debug(f"Skipping failed surface generation sample: {e}")
            continue

    if len(structures) < n_samples:
        logger.warning(f"Could only generate {len(structures)}/{n_samples} surfaces within constraints.")

    return structures

def relax_structure(atoms: Atoms, steps: int = 100, fmax: float = 0.1, rc: float = 2.0) -> Atoms:
    """
    Relax structure to resolve overlaps using SoftSphereCalculator.
    
    Args:
        atoms (Atoms): Structure to relax.
        steps (int): Max steps.
        fmax (float): Force convergence criteria.
        rc (float): Cutoff radius for overlap check. 
                    Default 1.2A is conservative to just fix very bad overlaps 
                    without destroying crystal structure.
    
    Returns:
        Atoms: Relaxed structure.
    """
    # Preserve original calculator if any
    old_calc = atoms.calc
    
    
    # Set ZBL calculator
    # ZBL is robust for resolving overlaps
    atoms.calc = ZBLCalculator(cutoff=rc) 
    
    try:
        # Run minimization
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=fmax, steps=steps)
        
    except Exception as e:
        logger.warning(f"Relaxation (overlap resolution) failed: {e}")
        
    finally:
        # Restore (likely None)
        atoms.calc = old_calc
        
    return atoms
