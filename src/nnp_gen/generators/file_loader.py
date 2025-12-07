import logging
import numpy as np
from typing import List, Optional
from ase import Atoms
import ase.io
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import UserFileSystemConfig
from nnp_gen.core.physics import apply_vacancies
from nnp_gen.core.exceptions import GenerationError

logger = logging.getLogger(__name__)

class FileGenerator(BaseGenerator):
    """
    Generator that loads structures from a user-specified file.
    Supports repeating structures and injecting vacancies.
    """
    def __init__(self, config: UserFileSystemConfig):
        super().__init__(config)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Loads structures from the file specified in the config.
        """
        logger.info(f"Loading structures from file: {self.config.path}")

        try:
            # Load structures
            # ase.io.read can return Atoms or List[Atoms] depending on 'index'
            # We use index=':' to always get a list
            loaded = ase.io.read(self.config.path, index=':', format=self.config.format)
            if isinstance(loaded, Atoms):
                loaded = [loaded]

            if not loaded:
                raise GenerationError(f"No structures found in {self.config.path}")

            logger.info(f"Loaded {len(loaded)} structures from file.")

        except Exception as e:
            logger.error(f"Failed to load file {self.config.path}: {e}")
            raise GenerationError(f"File loading failed: {e}")

        final_structures = []
        rng = np.random.RandomState(42)

        # Clone logic
        n_repeats = self.config.repeat
        if n_repeats < 1:
            n_repeats = 1

        for atom_obj in loaded:
            # Basic sanitization
            # Ensure elements match config if config.elements is strict?
            # BaseSystemConfig validator checks elements list provided in config.
            # But the file might contain other elements.
            # We trust the user here or we could validate.
            # We should probably update the config elements to match the file if empty?
            # But config validation happens before generation.

            for _ in range(n_repeats):
                # Copy to ensure independence
                new_atoms = atom_obj.copy()

                # Apply Vacancies
                if self.config.vacancy_concentration > 0.0:
                    new_atoms = apply_vacancies(new_atoms, self.config.vacancy_concentration, rng)

                final_structures.append(new_atoms)

        return final_structures
