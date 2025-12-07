import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
from pymatgen.core import Structure, Lattice, Composition, Element
from nnp_gen.core.config import KnowledgeSystemConfig
from nnp_gen.generators.knowledge import KnowledgeBasedGenerator
from nnp_gen.core.exceptions import GenerationError

@pytest.fixture
def config():
    return KnowledgeSystemConfig(
        formula="LiFe0.5Co0.5O2",
        use_cod=True,
        use_prototypes=True,
        use_symmetry_generation=True,
        max_supercell_atoms=50
    )

@pytest.fixture
def generator(config):
    return KnowledgeBasedGenerator(config)

def test_guess_end_member(generator):
    """Test guessing logic for LiFe0.5Co0.5O2 -> LiCoO2 (assuming Co > Fe in EN)"""
    comp = Composition("LiFe0.5Co0.5O2")
    end_member = generator._guess_end_member(comp)

    # Expect Li(1) Co(1) O(2) -> LiCoO2
    assert "Co" in end_member
    assert "Fe" not in end_member
    assert "Li" in end_member
    assert "O" in end_member

    em_comp = Composition(end_member)
    assert em_comp.get_el_amt_dict()["Co"] == 1.0

def test_group_by_stoichiometry(generator):
    comp = Composition("LiFe0.5Co0.5O2")
    groups = generator._group_by_stoichiometry(comp)

    # Should have 3 groups: Li(1), O(2), (Fe,Co)(1)
    assert len(groups) == 3

    amounts = sorted([g['amount'] for g in groups])
    assert np.isclose(amounts[0], 1.0)
    assert np.isclose(amounts[1], 1.0)
    assert np.isclose(amounts[2], 2.0)

    # Find the mixed group
    mixed_group = next(g for g in groups if len(g['elements']) == 2)
    assert "Fe" in [e.symbol for e in mixed_group['elements']]
    assert "Co" in [e.symbol for e in mixed_group['elements']]

def test_apply_doping_vegard(generator):
    """Test that volume scales correctly when doping"""
    target_comp = Composition("LiFeO2")
    proto_struct = Structure(Lattice.cubic(10.0), ["Li", "Co", "O", "O"],
                             [[0,0,0], [0.5,0.5,0.5], [0,0.5,0], [0.5,0,0]])

    with patch("nnp_gen.generators.knowledge.OrderDisorderedStructureTransformation") as MockTrans:
        instance = MockTrans.return_value

        # Return a fresh clean ordered structure
        ordered_return = Structure(Lattice.cubic(10.0), ["Li", "Fe", "O", "O"],
                                   [[0,0,0], [0.5,0.5,0.5], [0,0.5,0], [0.5,0,0]])

        instance.apply_transformation.side_effect = lambda s, return_ranked_list=1: [{'structure': ordered_return}]

        with patch.object(Structure, "scale_lattice", wraps=proto_struct.scale_lattice) as mock_scale:
            res = generator._apply_doping(target_comp, proto_struct)
            assert mock_scale.called
            assert isinstance(res, Atoms)

def test_generate_cod_cascade(generator):
    """Test the full flow with mocks"""

    with patch("nnp_gen.generators.knowledge.COD") as MockCOD:
        cod_instance = MockCOD.return_value

        dummy_struct = Structure(Lattice.cubic(5.0), ["Li", "Co", "O", "O"],
                                 [[0,0,0], [0.5,0.5,0.5], [0,0.5,0], [0.5,0,0]])

        # Side effect: 1. Exact match [], 2. Prototype [dummy]
        cod_instance.get_structure_by_formula.side_effect = [[], [dummy_struct]]

        with patch("nnp_gen.generators.knowledge.OrderDisorderedStructureTransformation") as MockTrans:
            instance = MockTrans.return_value
            # Return a FRESH ordered structure
            # If we return dummy_struct again, it might be modified by the doping step
            clean_struct = Structure(Lattice.cubic(5.0), ["Li", "Fe", "O", "O"],
                                     [[0,0,0], [0.5,0.5,0.5], [0,0.5,0], [0.5,0,0]])

            instance.apply_transformation.side_effect = lambda s, return_ranked_list=1: [{'structure': clean_struct}]

            atoms_list = generator.generate()

            assert len(atoms_list) > 0
            assert cod_instance.get_structure_by_formula.call_count == 2
