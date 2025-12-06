import pytest
import os
import numpy as np
from ase import Atoms
from pydantic import ValidationError
from nnp_gen.core.models import StructureMetadata
from nnp_gen.core.storage import DatabaseManager

@pytest.fixture
def db_manager(tmp_path):
    # Use temporary file
    db_file = tmp_path / "test.db"
    return DatabaseManager(str(db_file))

def test_schema_validation():
    # Valid
    meta = StructureMetadata(source="seed", config_hash="abc")
    assert meta.source == "seed"

    # Invalid source
    with pytest.raises(ValidationError):
        StructureMetadata(source="invalid_source", config_hash="abc")

    # Missing required
    with pytest.raises(ValidationError):
        StructureMetadata(source="seed") # missing config_hash

def test_round_trip(db_manager):
    atoms = Atoms('H2', positions=[[0,0,0], [0.74,0,0]])
    # Add descriptor
    desc = np.random.rand(100)
    atoms.info['descriptor'] = desc

    meta = StructureMetadata(source="md", config_hash="hash_123", energy=-1.5)

    # Save
    id = db_manager.save_atoms(atoms, meta)
    assert id == 1

    # Check descriptor was restored in atoms object
    assert 'descriptor' in atoms.info
    assert np.allclose(atoms.info['descriptor'], desc)

    # Load back via direct DB access to verify storage
    row = db_manager.db.get(id=id)
    assert row.source == "md"
    assert row.energy == -1.5
    # Check descriptor in data blob
    assert hasattr(row, 'data')
    assert 'descriptor' in row.data
    assert np.allclose(row.data['descriptor'], desc)

def test_query_filtering(db_manager):
    atoms1 = Atoms('H')
    meta1 = StructureMetadata(source="seed", config_hash="hash_1")
    id1 = db_manager.save_atoms(atoms1, meta1)

    atoms2 = Atoms('He')
    meta2 = StructureMetadata(source="seed", config_hash="hash_2")
    id2 = db_manager.save_atoms(atoms2, meta2)

    # Update 2 as sampled
    db_manager.update_sampling_status([id2], method="fps")

    # Query using manager
    sampled = list(db_manager.get_sampled_structures())
    assert len(sampled) == 1
    assert str(sampled[0].symbols) == 'He'

    # Verify metadata update
    row2 = db_manager.db.get(id=id2)
    assert row2.is_sampled is True
    assert row2.sampling_method == "fps"

    # Verify descriptor retrieval logic works in get_sampled_structures (even if None here)
    assert 'descriptor' not in sampled[0].info

def test_bulk_save(db_manager):
    atoms_list = [Atoms('H'), Atoms('He')]
    meta_list = [
        StructureMetadata(source="seed", config_hash="hash_1"),
        StructureMetadata(source="seed", config_hash="hash_2")
    ]
    ids = db_manager.bulk_save(atoms_list, meta_list)
    assert len(ids) == 2
    assert db_manager.db.count() == 2
