
import logging
import numpy as np
from typing import List, Optional
from ase import Atoms
from ase.build import bulk
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import RandomSystemConfig

logger = logging.getLogger(__name__)

class RandomGenerator(BaseGenerator):
    def __init__(self, config: RandomSystemConfig, seed: Optional[int] = None):
        super().__init__(config, seed=seed)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates N random alloy structures with varying parameters.
        Each structure will sample its own parameters (lattice, spacegroup, composition).
        """
        structures = []
        
        # Base RNG
        seed_val = self.seed if self.seed is not None else 42
        rng = np.random.RandomState(seed_val)

        # Imports for surface generation
        from ase.build import surface
        
        for i in range(self.config.n_initial_structures):
            try:
                # 1. Select Spacegroup
                if self.config.spacegroup_mode == "fixed":
                    sg = 225 # Default if fixed but not specified? Or should use 'spacegroups' list[0]?
                    if self.config.spacegroups:
                        sg = self.config.spacegroups[0]
                    else:
                        sg = 225 # FCC default
                elif self.config.spacegroup_mode == "random_list":
                    if not self.config.spacegroups:
                        # Fallback
                        sg = 225 
                    else:
                        sg = rng.choice(self.config.spacegroups)
                else: # random_all
                    # Simplified: pick from common high-symmetry sets?
                    # Or truly random 1-230 is likely to fail with small unit cells if not handled carefully.
                    # For MVP, let's pick from a safe set (FCC 225, BCC 229, HCP 194, SC 221)
                    # unless we integrate pyxtal for true symmetry generation.
                    # Given 'alloy' context, simple lattices are best.
                    sg = rng.choice([225, 229, 221])

                # 2. Sample Lattice Constant
                l_min, l_max = self.config.lattice_constant_range
                a = rng.uniform(l_min, l_max)
                
                # 3. Build Primitive
                # Similar logic to AlloyGenerator but dynamic
                struc_type = 'fcc'
                if sg == 229:
                    struc_type = 'bcc'
                elif sg == 221:
                    struc_type = 'sc'
                
                # Use first element as template
                template_el = self.config.elements[0]
                prim = bulk(template_el, struc_type, a=a)
                
                # 4. Supercell
                # Maybe modify supercell size randomly too? 
                # For now use config size.
                supercell_atoms = prim * self.config.supercell_size
                
                # 5. Composition
                n_atoms = len(supercell_atoms)
                elements = self.config.elements
                
                if self.config.composition_mode == "balanced":
                    # Equal distribution
                    n_types = len(elements)
                    element_counts = [n_atoms // n_types] * n_types
                    # Handle remainder
                    remainder = n_atoms % n_types
                    for r in range(remainder):
                        element_counts[r] += 1
                        
                    symbols = []
                    for el, count in zip(elements, element_counts):
                        symbols.extend([el] * count)
                    rng.shuffle(symbols)
                    
                elif self.config.composition_mode == "range":
                    # Sample composition from ranges
                    # Norm logic:
                    # x_i ~ U(min, max)
                    # X_i = x_i / sum(x_i)
                    raw_props = []
                    for el in elements:
                        r_min, r_max = 0.0, 1.0
                        if self.config.composition_ranges and el in self.config.composition_ranges:
                            r_min, r_max = self.config.composition_ranges[el]
                        
                        val = rng.uniform(r_min, r_max)
                        raw_props.append(val)
                    
                    total_p = sum(raw_props)
                    if total_p == 0:
                        norm_props = [1.0/len(elements)] * len(elements)
                    else:
                        norm_props = [p/total_p for p in raw_props]
                        
                    # Convert to counts
                    # Use probabilistic choice for each atom individually to be truly random?
                    # Or fixed counts based on probability?
                    # Probabilistic per atom allows micro-fluctuations and easier implementation.
                    symbols = rng.choice(elements, size=n_atoms, p=norm_props)
                    
                else: # random
                    # Uniformly random choice per atom
                    symbols = rng.choice(elements, size=n_atoms)
                
                supercell_atoms.set_chemical_symbols(symbols)
                
                # Info metadata
                supercell_atoms.info['config_source'] = "random_generator"
                supercell_atoms.info['generated_lattice_constant'] = a
                supercell_atoms.info['generated_spacegroup'] = int(sg)
                
                structures.append(supercell_atoms)
                
                # Generate Surfaces?
                if self.config.n_surface_samples > 0:
                    from nnp_gen.generators.utils import generate_random_surfaces
                    surfaces = generate_random_surfaces(
                        base_structure=supercell_atoms,
                        n_samples=self.config.n_surface_samples,
                        rng=rng,
                        source_prefix=f"random_surface_sg{sg}",
                        max_atoms=self.config.constraints.max_atoms
                    )
                    structures.extend(surfaces)
                
            except Exception as e:
                logger.warning(f"Failed to generate random structure {i}: {e}")
                continue
                
        return structures
