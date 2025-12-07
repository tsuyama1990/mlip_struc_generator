import os
import pytest
import shutil
from ase import Atoms
from nnp_gen.core.storage import ASEDbStorage
from nnp_gen.core.models import StructureMetadata
from nnp_gen.core.exceptions import PipelineError

@pytest.fixture
def storage_path(tmp_path):
    return str(tmp_path / "test_storage.db")

def test_storage_save_and_load(storage_path):
    storage = ASEDbStorage(storage_path)
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.info['descriptor'] = {'data': [1, 2, 3]} # Mock descriptor

    meta = StructureMetadata(source="seed", config_hash="test_hash")

    # Save
    id = storage.save_atoms(atoms, meta)
    assert id == 1

    # Load via iterator
    loaded_atoms = list(storage.get_sampled_structures())
    # Should be empty because is_sampled defaults to False in metadata and get_sampled_structures filters by is_sampled=True
    assert len(loaded_atoms) == 0

    # Update status
    storage.update_sampling_status([id], method="random")

    loaded_atoms = list(storage.get_sampled_structures())
    assert len(loaded_atoms) == 1
    assert loaded_atoms[0].get_chemical_formula() == "H2O"
    assert loaded_atoms[0].info['descriptor'] == {'data': [1, 2, 3]}

def test_checkpoint_resilience(storage_path):
    """
    Test creating a checkpoint DB, 'crashing' (creating a new instance), and verifying data persists.
    """
    storage1 = ASEDbStorage(storage_path)
    structures = [Atoms('H2', positions=[[0,0,0], [0,0,d]]) for d in range(1, 11)]

    # Bulk save with dict metadata (checkpoint style)
    meta_list = [{"source": "seed", "config_hash": "hash", "stage": "generated"} for _ in range(10)]
    storage1.bulk_save(structures, meta_list)

    del storage1 # Simulate closure

    # New instance
    storage2 = ASEDbStorage(storage_path)

    # Verify we can read them back.
    # ASEDbStorage doesn't have a generic "read all" method in IStorage interface,
    # but we can use the underlying db object for testing.
    rows = list(storage2.db.select())
    assert len(rows) == 10
    assert rows[0].stage == "generated"

def test_bulk_save_transaction(storage_path):
    storage = ASEDbStorage(storage_path)
    atoms_list = [Atoms('H'), Atoms('He')]
    # Second metadata is invalid (missing required fields for StructureMetadata if we used it,
    # but here we use loose dicts. Let's force an error by passing something non-iterable where expected or causing DB error)

    # To force a DB error in sqlite is hard with simple writes.
    # Let's mock self.db.write to raise an exception on the second item.

    original_write = storage.db.write

    def mock_write(atoms, **kwargs):
        if atoms.get_chemical_formula() == "He":
            raise RuntimeError("DB Crash")
        return original_write(atoms, **kwargs)

    storage.db.write = mock_write

    meta_list = [{"source": "seed"}, {"source": "seed"}]

    try:
        storage.bulk_save(atoms_list, meta_list)
    except RuntimeError:
        pass

    # Verify rollback: count should be 0
    # Note: ASE's sqlite `with db:` handles transaction.
    # If using standard sqlite3 connection it works. ase.db.core.Database doesn't always expose transaction context
    # the same way unless backend supports it. ASE SQLite does.

    rows = list(storage.db.select())
    assert len(rows) == 0
