import ase.db
import logging
from typing import List, Iterator, Optional, Any, Union, Dict
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from nnp_gen.core.models import StructureMetadata
from nnp_gen.core.interfaces import IStorage

logger = logging.getLogger(__name__)

class ASEDbStorage(IStorage):
    """
    Concrete implementation of IStorage using ASE's SQLite database.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = ase.db.connect(db_path)

    def _save_atoms_impl(self, atoms: Atoms, metadata: Union[StructureMetadata, Dict[str, Any]]) -> int:
        """
        Implementation of saving without opening a new transaction context.
        Internal use for bulk_save.
        """
        # Convert metadata to dictionary if it's a Pydantic model
        if isinstance(metadata, StructureMetadata):
            kv_pairs = metadata.model_dump(exclude_none=True)
        else:
            kv_pairs = metadata.copy()

        # Prepare for data blob
        data = {}
        descriptor = atoms.info.pop('descriptor', None)
        original_calc = atoms.calc

        try:
            if descriptor is not None:
                data['descriptor'] = descriptor

            # Handle 'energy' special case
            # ASE DB treats 'energy' as a reserved property.
            if 'energy' in kv_pairs:
                energy_val = kv_pairs.pop('energy')
                # We enforce this energy value by attaching a SinglePointCalculator
                if atoms.calc is None or not isinstance(atoms.calc, SinglePointCalculator):
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

    def save_atoms(self, atoms: Atoms, metadata: Union[StructureMetadata, Dict[str, Any]]) -> int:
        """
        Save a single structure with metadata.
        Uses its own transaction.
        """
        with self.db:
            return self._save_atoms_impl(atoms, metadata)

    def bulk_save(self, atoms_list: List[Atoms], metadata_list: List[Union[StructureMetadata, Dict[str, Any]]]) -> List[int]:
        """
        Save multiple structures in a transaction.
        """
        ids = []
        try:
            # Use 'with self.db:' to ensure transaction behavior
            with self.db:
                for atoms, meta in zip(atoms_list, metadata_list):
                    ids.append(self._save_atoms_impl(atoms, meta))
            return ids
        except Exception as e:
            logger.error(f"Bulk save failed, rolling back transaction. Error: {e}")
            raise e

    def update_sampling_status(self, ids: List[int], method: str):
        """
        Update is_sampled and sampling_method for given IDs.
        """
        try:
            with self.db:
                for id in ids:
                    self.db.update(id, is_sampled=True, sampling_method=method)
        except Exception as e:
            logger.error(f"Failed to update sampling status: {e}")
            raise e

    def get_sampled_structures(self) -> Iterator[Atoms]:
        """
        Yield atoms that have is_sampled=True.
        """
        for row in self.db.select(is_sampled=True):
            atoms = row.toatoms()

            # Restore data blob to info if present
            if hasattr(row, 'data') and 'descriptor' in row.data:
                atoms.info['descriptor'] = row.data['descriptor']

            yield atoms
