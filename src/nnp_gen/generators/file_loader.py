import logging
import os
from typing import List, Union
from ase import Atoms
import ase.io
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import UserFileSystemConfig
from nnp_gen.core.exceptions import GenerationError

logger = logging.getLogger(__name__)

class FileGenerator(BaseGenerator):
    def __init__(self, config: UserFileSystemConfig):
        super().__init__(config)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Loads structures from a user-specified file and duplicates them if configured.
        """
        path = self.config.path
        if not os.path.exists(path):
            raise GenerationError(f"Input file not found: {path}")

        logger.info(f"Loading structures from {path}")

        try:
            # ase.io.read with index=':' always returns a list of Atoms
            # If format is None, ASE infers it.
            structures = ase.io.read(path, index=':', format=self.config.format)

            # Ensure it is a list (ase.io.read can return Atoms if index is not specified,
            # but with index=':' it should be a list. However, let's be safe.)
            if isinstance(structures, Atoms):
                structures = [structures]
            elif not isinstance(structures, list):
                 # Should not happen with index=':'
                 structures = [structures]

        except Exception as e:
            raise GenerationError(f"Failed to read file {path}: {e}")

        if not structures:
             logger.warning(f"File {path} contained no structures.")
             return []

        # Cloning Logic
        repeat_count = self.config.repeat
        if repeat_count > 1:
            logger.info(f"Duplicating {len(structures)} structures {repeat_count} times.")
            # Depending on desire, we can duplicate the whole list n times
            # or duplicate each item n times.
            # "Clone that structure 50 times".
            # If input has 1000 frames, and repeat=1, we get 1000 structures.
            # If input has 1 frame, and repeat=50, we get 50 structures.
            # Usually repeat is for single structure seeds.
            # But let's assume we multiply the list.

            # A simple multiplication of list works: [A, B] * 2 = [A, B, A, B]
            # But we should copy them to be safe if we modify them later (rattle is done in base class on copies/in-place?)
            # BaseGenerator calls _generate_impl, then applies rattle in-place.
            # So we MUST return independent copies.

            final_structures = []
            for _ in range(repeat_count):
                for atoms in structures:
                    final_structures.append(atoms.copy())

            structures = final_structures
        else:
            # Even if repeat=1, we should probably ensure they are copies if ASE cached them,
            # though ASE read usually creates new objects.
            pass

        return structures
