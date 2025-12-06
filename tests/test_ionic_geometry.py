
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from ase import Atoms
from nnp_gen.generators.ionic import IonicGenerator
from nnp_gen.core.config import IonicSystemConfig

class TestIonicGeometry:

    @pytest.fixture
    def config(self):
        return IonicSystemConfig(
            name="test",
            elements=["Na", "Cl"],
            oxidation_states={"Na": 1, "Cl": -1},
            supercell_size=[1, 1, 1],
            rattle_std=0.0,
            vol_scale_range=[1.0, 1.0] # No strain
        )

    def test_geometric_factors(self, config):
        """
        Test that lattice constant estimation correctly applies geometric factors
        for different structure types.
        """
        generator = IonicGenerator(config)

        # Mock pymatgen presence
        generator.has_pmg = True
        generator.pmg = MagicMock()
        generator.Lattice = MagicMock()
        generator.Species = MagicMock()

        # Mock radii
        # R_A = 1.0, R_B = 2.0 -> Sum = 3.0
        with patch.object(generator, '_get_heuristic_radius', side_effect=lambda s, c: 1.0 if s == "Na" else 2.0):

            # Mock _pmg_to_ase to simply return an Atoms object with the cell
            # because we can't easily mock the complex pymatgen Structure logic fully
            # without a lot of setup.
            # Instead, we will spy on the Lattice.cubic call to see what 'a' was passed.

            generator.pmg.Structure.from_spacegroup = MagicMock()

            # Call the method under test directly
            # args: species_a, species_b, radii_sum, types
            radii_sum = 3.0

            # Test Rocksalt
            # a = 2 * sum = 6.0
            generator._create_binary_prototypes("Na", "Cl", radii_sum, ["rocksalt"])

            # Check arguments to Lattice.cubic(a)
            # The structure creation is: self.pmg.Structure.from_spacegroup("Fm-3m", self.Lattice.cubic(a), ...)
            # So we check if Lattice.cubic was called with ~6.0

            # Retrieve all calls to cubic
            calls = generator.Lattice.cubic.call_args_list
            assert len(calls) > 0

            # Find the call corresponding to rocksalt (should be the last one or close)
            # Since we just ran one, it should be there.
            args, _ = calls[-1]
            a_rocksalt = args[0]
            assert np.isclose(a_rocksalt, 2.0 * radii_sum)

            # Test CsCl
            # a = 2/sqrt(3) * sum approx 1.1547 * 3.0 = 3.464
            generator._create_binary_prototypes("Na", "Cl", radii_sum, ["cscl"])
            args, _ = generator.Lattice.cubic.call_args_list[-1]
            a_cscl = args[0]
            expected_cscl = (2.0 / np.sqrt(3.0)) * radii_sum
            assert np.isclose(a_cscl, expected_cscl)

            # Test Zincblende
            # a = 4/sqrt(3) * sum approx 2.309 * 3.0 = 6.928
            generator._create_binary_prototypes("Na", "Cl", radii_sum, ["zincblende"])
            args, _ = generator.Lattice.cubic.call_args_list[-1]
            a_zb = args[0]
            expected_zb = (4.0 / np.sqrt(3.0)) * radii_sum
            assert np.isclose(a_zb, expected_zb)

    def test_radii_sum_logic(self, config):
        """
        Verify that we are indeed summing radii, not diameters.
        """
        generator = IonicGenerator(config)
        generator.has_pmg = True

        # Mock _get_radius to return known radii
        with patch.object(generator, '_get_heuristic_radius', side_effect=lambda s, c: 1.5 if s == "Na" else 0.5):
            # Sum should be 2.0
            # Mock _create_binary_prototypes to capture the passed sum
            with patch.object(generator, '_create_binary_prototypes') as mock_create:
                generator._generate_with_pymatgen()

                # Check call arguments
                # call args: (el1, el2, radii_sum, types)
                assert mock_create.called
                args, _ = mock_create.call_args
                passed_sum = args[2]
                assert np.isclose(passed_sum, 2.0)
