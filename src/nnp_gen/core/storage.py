import ase.db
import logging
from typing import List, Iterator, Optional, Any
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from nnp_gen.core.models import StructureMetadata
from nnp_gen.core.interfaces import IStorage

logger = logging.getLogger(__name__)

class DatabaseManager(IStorage):
    """
    Manager for ASE Database persistence with Pydantic metadata validation.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = ase.db.connect(db_path)

    def save_atoms(self, atoms: Atoms, metadata: StructureMetadata) -> int:
        """
        Save a single structure with metadata.

        Args:
            atoms (Atoms): The structure to save.
            metadata (StructureMetadata): Validated metadata.

        Returns:
            int: The database ID of the saved row.
        """
        # Convert metadata to dictionary
        kv_pairs = metadata.model_dump(exclude_none=True)

        # Prepare for data blob
        data = {}
        descriptor = atoms.info.pop('descriptor', None)
        original_calc = atoms.calc

        try:
            if descriptor is not None:
                data['descriptor'] = descriptor

            # Handle 'energy' special case
            # ASE DB treats 'energy' as a reserved property to be stored in the main table.
            # We must not put it in key_value_pairs.
            if 'energy' in kv_pairs:
                energy_val = kv_pairs.pop('energy')
                # We enforce this energy value by attaching a SinglePointCalculator
                atoms.calc = SinglePointCalculator(atoms, energy=energy_val)

            # Write
            # key_value_pairs are merged with what's in atoms.info
            id = self.db.write(atoms, key_value_pairs=kv_pairs, data=data)
            return id
        finally:
            # Restore descriptor
            if descriptor is not None:
                atoms.info['descriptor'] = descriptor
            # Restore calc
            atoms.calc = original_calc

    def bulk_save(self, atoms_list: List[Atoms], metadata_list: List[StructureMetadata]) -> List[int]:
        """
        Save multiple structures in a transaction.
        """
        ids = []
        try:
            with self.db:
                for atoms, meta in zip(atoms_list, metadata_list):
                    ids.append(self.save_atoms(atoms, meta))
            return ids
        except Exception as e:
            logger.error(f"Database write failed: {e}")
            raise e

    def update_sampling_status(self, ids: List[int], method: str):
        """
        Update is_sampled and sampling_method for given IDs.
        """
        with self.db:
            for id in ids:
                self.db.update(id, is_sampled=True, sampling_method=method)

    def get_sampled_structures(self) -> Iterator[Atoms]:
        """
        Yield atoms that have is_sampled=True.
        Descriptors in 'data' column are restored to atoms.info['descriptor'].
        """
        # Select rows where is_sampled is True
        for row in self.db.select(is_sampled=True):
            atoms = row.toatoms()

            # Restore data blob to info if present
            if hasattr(row, 'data') and 'descriptor' in row.data:
                atoms.info['descriptor'] = row.data['descriptor']

            yield atoms
