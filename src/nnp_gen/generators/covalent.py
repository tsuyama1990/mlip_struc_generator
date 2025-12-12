import logging
import numpy as np
from typing import List
from ase import Atoms
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import CovalentSystemConfig
from nnp_gen.core.physics import apply_vacancies

logger = logging.getLogger(__name__)

class CovalentGenerator(BaseGenerator):
    def __init__(self, config: CovalentSystemConfig, seed: int = 42):
        super().__init__(config, seed=seed)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates covalent structures using random symmetry groups (PyXTal).
        """
        logger.info(f"Generating covalent structures for {self.config.elements}")

        try:
            from pyxtal import pyxtal
        except ImportError:
            logger.error("pyxtal not installed.")
            raise ImportError("pyxtal is required for CovalentGenerator")

        structures = []
        n_structures = self.config.n_initial_structures
        
        # Use passed seed
        seed_val = self.seed if self.seed is not None else 42
        rng = np.random.RandomState(seed_val)

        attempts = 0
        max_attempts = n_structures * 10
        
        count = 0
        while count < n_structures and attempts < max_attempts:
            attempts += 1
            try:
                struc = pyxtal()

                dim = self.config.dimensionality
                if dim == 3:
                    sg = rng.randint(1, 231)
                elif dim == 2:
                    sg = rng.randint(1, 81)
                elif dim == 1:
                    sg = rng.randint(1, 76)
                else:
                    sg = 1

                # Random stoichiometry: 1 to 4 atoms per element type (per primitive cell)
                num_ions = [rng.randint(1, 5) for _ in self.config.elements]

                # from_random(dim, group, species, numIons)
                struc.from_random(dim, sg, self.config.elements, num_ions)

                if struc.valid:
                    ase_atoms = struc.to_ase()
                    
                    # Apply Supercell
                    if hasattr(self.config, 'supercell_size'):
                        ase_atoms *= self.config.supercell_size

                    # Apply Vacancies
                    if self.config.vacancy_concentration > 0.0:
                        ase_atoms = apply_vacancies(ase_atoms, self.config.vacancy_concentration, rng)
                    
                    ase_atoms.info['config_source'] = f"covalent_random_sg{sg}"
                    structures.append(ase_atoms)
                    
                    # Generate Surfaces?
                    if self.config.n_surface_samples > 0:
                        from nnp_gen.generators.utils import generate_random_surfaces
                        surfaces = generate_random_surfaces(
                            base_structure=ase_atoms,
                            n_samples=self.config.n_surface_samples,
                            rng=rng,
                            source_prefix=f"covalent_surface_sg{sg}",
                            max_atoms=self.config.constraints.max_atoms
                        )
                        structures.extend(surfaces)

                    count += 1
            except Exception as e:
                # pyxtal generation can fail often, just skip
                logger.debug(f"Generation failed: {e}")
                continue

        if not structures:
            logger.warning("No valid structures generated via PyXTal.")

        return structures
