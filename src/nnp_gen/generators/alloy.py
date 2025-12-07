import logging
import numpy as np
from typing import List
from ase import Atoms
from ase.build import bulk
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import AlloySystemConfig
from nnp_gen.core.physics import estimate_lattice_constant, apply_vacancies

logger = logging.getLogger(__name__)

class AlloyGenerator(BaseGenerator):
    def __init__(self, config: AlloySystemConfig):
        super().__init__(config)
        self.config = config

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
             a = estimate_lattice_constant(self.config.elements, structure=struc_type)
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

        # Random substitution
        # We assign elements randomly.
        # For better physics, we could use SQS if icet is installed,
        # but random solution is a valid baseline.

        n_atoms = len(atoms)
        elements = self.config.elements

        rng = np.random.RandomState(42)
        symbols = rng.choice(elements, size=n_atoms)
        atoms.set_chemical_symbols(symbols)

        # Apply Vacancies
        if self.config.vacancy_concentration > 0.0:
            atoms = apply_vacancies(atoms, self.config.vacancy_concentration, rng)

        structures.append(atoms)

        return structures
