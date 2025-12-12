import logging
import numpy as np
from typing import List, Optional
from ase import Atoms
from ase.build import bulk
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import AlloySystemConfig
from nnp_gen.core.physics import estimate_lattice_constant, apply_vacancies

logger = logging.getLogger(__name__)

class AlloyGenerator(BaseGenerator):
    def __init__(self, config: AlloySystemConfig, seed: Optional[int] = None):
        super().__init__(config, seed=seed)
        self.config = config

    def _apply_symbols(self, atoms: Atoms, element_probs: Optional[List[float]], rng: np.random.RandomState):
        """
        Applies chemical symbols to atoms based on config and probabilities.
        """
        n_atoms = len(atoms)
        elements = self.config.elements
        
        if self.config.composition_mode == "balanced":
             # Equal distribution with exact counts
             n_types = len(elements)
             element_counts = [n_atoms // n_types] * n_types
             remainder = n_atoms % n_types
             for r in range(remainder):
                 element_counts[r] += 1
             
             symbols = []
             for el, count in zip(elements, element_counts):
                 symbols.extend([el] * count)
             rng.shuffle(symbols)
             atoms.set_chemical_symbols(symbols)
             
        elif element_probs is not None:
             # Use the provided probabilities (from range)
             symbols = rng.choice(elements, size=n_atoms, p=element_probs)
             atoms.set_chemical_symbols(symbols)
             atoms.info['target_composition'] = {el: p for el, p in zip(elements, element_probs)}
             
        else:
             # Random (uniform)
             symbols = rng.choice(elements, size=n_atoms)
             atoms.set_chemical_symbols(symbols)


    def _generate_impl(self) -> List[Atoms]:
        """
        Generates alloy structures using random substitution.
        """
        logger.info(f"Generating alloy structures for {self.config.elements}")
        structures = []

        # Determine structure string for estimation
        sg = self.config.spacegroup
        struc_type = 'fcc'
        if sg == 229:
            struc_type = 'bcc'

        # Determine lattice constant
        if self.config.lattice_constant:
             a = self.config.lattice_constant
        else:
             a = estimate_lattice_constant(self.config.elements, structure=struc_type, method=self.config.lattice_estimation_method)
             logger.info(f"Estimated lattice constant a={a:.3f} A")

        try:
            if sg == 229: # BCC
                prim = bulk('Fe', 'bcc', a=a)
            elif sg == 225 or sg is None: # FCC default
                # Use first element as dummy species
                prim = bulk('Cu', 'fcc', a=a)
            else:
                # Basic support for now
                logger.warning(f"Unsupported spacegroup {sg}, defaulting to FCC")
                prim = bulk('Cu', 'fcc', a=a)
        except Exception as e:
            logger.error(f"Failed to build primitive cell: {e}")
            return []

        # Create supercell
        size = self.config.supercell_size
        atoms = prim * size

        # Random Number Generator
        seed_val = self.seed if self.seed is not None else 42
        rng = np.random.RandomState(seed_val)

        # Generate structures
        for i in range(self.config.n_initial_structures):
            # 1. Determine probabilities for this iteration
            element_probs = None
            if self.config.composition_mode == "range":
                raw_props = []
                for el in self.config.elements:
                    r_min, r_max = 0.0, 1.0
                    if self.config.composition_ranges and el in self.config.composition_ranges:
                        r_min, r_max = self.config.composition_ranges[el]
                    val = rng.uniform(r_min, r_max)
                    raw_props.append(val)
                total_p = sum(raw_props)
                if total_p > 0:
                    element_probs = [p/total_p for p in raw_props]
                else:
                    element_probs = [1.0/len(self.config.elements)] * len(self.config.elements)

            # 2. Build Bulk
            base = prim * size
            self._apply_symbols(base, element_probs, rng)
            
            if self.config.vacancy_concentration > 0.0:
                base = apply_vacancies(base, self.config.vacancy_concentration, rng)
            
            # Relax to remove overlaps
            from nnp_gen.generators.utils import relax_structure
            base = relax_structure(base)
            
            base.info['config_source'] = "alloy_bulk"
            structures.append(base)
            
            # 3. Use PRIMITIVE cell for surface generation to control size
            if self.config.n_surface_samples > 0:
                from nnp_gen.generators.utils import generate_random_surfaces, relax_structure
                
                # generate_random_surfaces returns slabs with dummy composition (from prim)
                surfaces = generate_random_surfaces(
                    base_structure=prim,
                    n_samples=self.config.n_surface_samples,
                    rng=rng,
                    source_prefix="alloy_surface",
                    max_atoms=self.config.constraints.max_atoms
                )
                
                # Apply correct composition and vacancies to these slabs
                for s in surfaces:
                    self._apply_symbols(s, element_probs, rng)
                    if self.config.vacancy_concentration > 0.0:
                        s = apply_vacancies(s, self.config.vacancy_concentration, rng)
                    
                    # Relax surfaces
                    s = relax_structure(s)
                    
                    structures.append(s)

        return structures
