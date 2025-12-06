import pytest
import numpy as np
from ase import Atoms
from nnp_gen.samplers.selector import DescriptorManager, FPSSampler, RandomSampler

def test_descriptor_manager_rdf():
    # Force RDF by not having dscribe (or assuming mock missing)
    # But dscribe is not installed, so it should use fallback.
    dm = DescriptorManager(rcut=5.0)
    structures = [
        Atoms('H2', positions=[[0,0,0], [1,0,0]]),
        Atoms('H2', positions=[[0,0,0], [2,0,0]])
    ]
    feats = dm.calculate(structures)
    assert feats.shape[0] == 2
    assert feats.shape[1] > 0
    # verify dist 1 is in different bin than dist 2 if bins resolve it
    # bins linspace(0, 5, 20) -> step 0.25. 1.0 is in bin 4. 2.0 in bin 8.
    assert not np.allclose(feats[0], feats[1])

def test_fps_sampler():
    dm = DescriptorManager()
    fps = FPSSampler(dm)

    # 3 structures: A, B(close to A), C(far from A)
    # A=[0], B=[0.1], C=[5.0]
    struct_A = Atoms('H2', positions=[[0,0,0], [1,0,0]])
    struct_B = Atoms('H2', positions=[[0,0,0], [1.05,0,0]]) # Very similar to A
    struct_C = Atoms('H2', positions=[[0,0,0], [4,0,0]])    # Distinct

    structures = [struct_A, struct_B, struct_C]
    # Sample 2.
    # If random start A, should pick C (max dist).
    # If random start B, should pick C.
    # If random start C, should pick A or B (whichever is further, approx same).
    # So results should be {A, C} or {B, C}.
    # {A, B} is unlikely (distance small).

    sampled = fps.sample(structures, 2)
    assert len(sampled) == 2

    # Check if C is in sampled
    # We compare positions.
    has_C = False
    for s in sampled:
        if np.allclose(s.positions, struct_C.positions):
            has_C = True
            break

    assert has_C

def test_random_sampler():
    rs = RandomSampler()
    structures = [Atoms('H') for _ in range(10)]
    sampled = rs.sample(structures, 5)
    assert len(sampled) == 5
    # indices unique by checking object identity if created unique
    ids = [id(s) for s in sampled]
    assert len(set(ids)) == 5
