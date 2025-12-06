from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import hashlib
import numpy as np
from ase import Atoms, units
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list
from nnp_gen.core.config import SystemConfig
from nnp_gen.core.physics import (
    apply_rattle,
    apply_volumetric_strain,
    set_initial_magmoms,
    ensure_supercell_size
)
from nnp_gen.core.exceptions import GenerationError
from nnp_gen.core.models import StructureMetadata
from nnp_gen.core.validation import StructureValidator

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """
    Abstract Base Class for structure generators with built-in validation.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.validator = StructureValidator(config.constraints)

    def generate(self) -> List[Atoms]:
        """
        Generates initial structures and filters them based on physics constraints.

        Returns:
            List[Atoms]: A list of valid ASE Atoms objects.

        Raises:
            GenerationError: If underlying generation logic fails.
        """
        try:
            raw_structures = self._generate_impl()
        except GenerationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generator: {e}")
            raise GenerationError(f"Generation failed: {e}")

        valid_structures = []

        for i, atoms in enumerate(raw_structures):
            # Pipeline Logic

            # Seed for this structure's random operations
            # STRENGTHENED SEED GENERATION
            # Issue: Collisions with just formula + index.
            # Fix: Mix global seed (if any), formula, index, and positions hash.

            # Use global seed if in config? config is SystemConfig.
            # AppConfig has seed? SystemConfig might not.
            # Assuming deterministic generation is goal, we rely on structure data itself.

            positions_bytes = atoms.positions.tobytes()
            # If atoms.positions is not writable or weird, copy?
            # tobytes() is safe.

            # Mix components
            mix_str = f"{self.config.system_seed if hasattr(self.config, 'system_seed') else ''}_{atoms.get_chemical_formula()}_{i}"

            hasher = hashlib.sha256()
            hasher.update(mix_str.encode('utf-8'))
            hasher.update(positions_bytes)

            seed_hash = hasher.hexdigest()
            struct_seed = int(seed_hash, 16) % (2**32)

            # 0. Enforce PBC from config
            atoms.set_pbc(self.config.pbc)

            # 1. set_initial_magmoms (if applicable)
            if hasattr(self.config, 'default_magmoms') and self.config.default_magmoms:
                atoms = set_initial_magmoms(atoms, self.config.default_magmoms)

            # 2. apply_volumetric_strain (Moved BEFORE ensure_supercell_size)
            if self.config.vol_scale_range:
                atoms = apply_volumetric_strain(atoms, self.config.vol_scale_range, seed=struct_seed)

            # 3. apply_rattle (Moved BEFORE ensure_supercell_size)
            if self.config.rattle_std > 0:
                atoms = apply_rattle(atoms, self.config.rattle_std, seed=struct_seed)

            # 4. ensure_supercell_size
            # Now we check supercell size on the potentially strained/rattled cell
            atoms = ensure_supercell_size(
                atoms,
                r_cut=self.config.constraints.r_cut,
                factor=self.config.constraints.min_cell_length_factor
            )

            # 5. validate_structure
            if self.validate_structure(atoms):
                valid_structures.append(atoms)
            else:
                logger.warning(f"Structure {i} rejected by SanityFilter.")

        return valid_structures

    @abstractmethod
    def _generate_impl(self) -> List[Atoms]:
        """
        Implementation of the specific generation logic.
        Must return ase.Atoms objects.
        """
        pass

    def validate_structure(self, atoms: Atoms) -> bool:
        """
        Validates a single structure against constraints.
        Delegates to StructureValidator.
        """
        return self.validator.validate(atoms)

class IExplorer(ABC):
    """Interface for exploration methods (e.g., MD)."""
    @abstractmethod
    def explore(self, structures: List[Atoms], n_workers: Optional[int] = None) -> List[Atoms]:
        pass

class ISampler(ABC):
    """Interface for sampling strategies."""
    @abstractmethod
    def sample(self, structures: List[Atoms], n_samples: int) -> List[Atoms]:
        pass

class IStorage(ABC):
    """Interface for database storage."""
    @abstractmethod
    def bulk_save(self, structures: List[Atoms], metadata: List[StructureMetadata]) -> List[int]:
        pass

class IExporter(ABC):
    """Interface for file exporting."""
    @abstractmethod
    def export(self, structures: List[Atoms], output_path: str):
        pass
