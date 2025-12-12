import logging
import numpy as np
from typing import List, Optional
from ase import Atoms
import ase.io
from nnp_gen.core.interfaces import BaseGenerator

import os
import glob
from nnp_gen.core.config import FileSystemConfig

class FileGenerator(BaseGenerator):
    """
    Generator that loads structures from a user-specified file or directory.
    Supports repeating structures and injecting vacancies.
    """
    def __init__(self, config: FileSystemConfig, seed: Optional[int] = None):
        super().__init__(config, seed=seed)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Loads structures from the file or directory specified in the config.
        """
        paths = []
        
        if os.path.isdir(self.config.path):
            logger.info(f"Scanning directory: {self.config.path} (Recursive: {self.config.recursive}, Pattern: {self.config.pattern})")
            # Walk directory
            if self.config.recursive:
                for root, dirs, files in os.walk(self.config.path):
                    # Filter files using glob pattern matching
                    # fnmatch is easier for simple globs than glob.glob with full paths
                    import fnmatch
                    for f in files:
                        if fnmatch.fnmatch(f, self.config.pattern):
                            paths.append(os.path.join(root, f))
            else:
                 # Non-recursive
                 import fnmatch
                 for f in os.listdir(self.config.path):
                     if os.path.isfile(os.path.join(self.config.path, f)) and fnmatch.fnmatch(f, self.config.pattern):
                         paths.append(os.path.join(self.config.path, f))
        else:
            # Single file
            if os.path.exists(self.config.path):
                paths.append(self.config.path)
            else:
                raise GenerationError(f"Path not found: {self.config.path}")

        if not paths:
            logger.warning(f"No files found at {self.config.path} matching pattern {self.config.pattern}")
            return []

        logger.info(f"Found {len(paths)} files to load.")
        
        all_loaded = []
        
        for p in paths:
            try:
                # Load structures
                # ase.io.read can return Atoms or List[Atoms] depending on 'index'
                # We use index=':' to always get a list
                loaded = ase.io.read(p, index=':', format=self.config.format)
                if isinstance(loaded, Atoms):
                    loaded = [loaded]
                all_loaded.extend(loaded)
            except Exception as e:
                logger.warning(f"Failed to load file {p}: {e}")
                continue # Skip bad files

        if not all_loaded:
             if self.config.strict_mode:
                 raise GenerationError(f"Failed to load any valid structures from {len(paths)} files.")
             return []

        logger.info(f"Successfully loaded {len(all_loaded)} total structures.")

        final_structures = []
        seed_val = self.seed if self.seed is not None else 42
        rng = np.random.RandomState(seed_val)

        # Clone logic
        n_repeats = self.config.repeat
        if n_repeats < 1:
            n_repeats = 1

        for atom_obj in all_loaded:
            for _ in range(n_repeats):
                # Copy to ensure independence
                new_atoms = atom_obj.copy()

                # Apply Vacancies
                if self.config.vacancy_concentration > 0.0:
                    new_atoms = apply_vacancies(new_atoms, self.config.vacancy_concentration, rng)

                final_structures.append(new_atoms)

        return final_structures
