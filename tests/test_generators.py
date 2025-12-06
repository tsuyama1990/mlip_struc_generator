import pytest
import sys
from unittest.mock import MagicMock
from nnp_gen.core.config import AlloySystemConfig, IonicSystemConfig, CovalentSystemConfig, MoleculeSystemConfig
from nnp_gen.generators.factory import GeneratorFactory
from nnp_gen.generators.alloy import AlloyGenerator
from nnp_gen.generators.ionic import IonicGenerator

# Mock objects for external libraries
@pytest.fixture
def mock_icet(mocker):
    mock = MagicMock()
    mocker.patch.dict("sys.modules", {"icet": mock, "icet.tools": mock.tools})
    return mock

@pytest.fixture
def mock_pymatgen(mocker):
    mock = MagicMock()
    mocker.patch.dict("sys.modules", {"pymatgen": mock, "pymatgen.core": mock.core})
    return mock

@pytest.fixture
def mock_pyxtal(mocker):
    mock = MagicMock()
    mocker.patch.dict("sys.modules", {"pyxtal": mock})
    return mock

@pytest.fixture
def mock_rdkit(mocker):
    mock = MagicMock()
    mocker.patch.dict("sys.modules", {"rdkit": mock, "rdkit.Chem": mock.Chem, "rdkit.Chem.AllChem": mock.Chem.AllChem})
    return mock

def test_factory_creation():
    config = AlloySystemConfig(elements=["Cu", "Zr"], type="alloy")
    gen = GeneratorFactory.get_generator(config)
    assert isinstance(gen, AlloyGenerator)

    config_ionic = IonicSystemConfig(elements=["Na", "Cl"], oxidation_states={"Na":1, "Cl":-1}, type="ionic")
    gen_ionic = GeneratorFactory.get_generator(config_ionic)
    assert isinstance(gen_ionic, IonicGenerator)

def test_alloy_generator_mocked(mock_icet):
    """Test that AlloyGenerator attempts to import icet when available."""
    config = AlloySystemConfig(elements=["Cu", "Au"], type="alloy")
    gen = AlloyGenerator(config)

    structures = gen.generate()

    # Check symbol to differentiate (mock path returns Cu4)
    assert str(structures[0].symbols) == "Cu4"

def test_ionic_generator_mocked(mock_pymatgen):
    config = IonicSystemConfig(elements=["Li", "F"], oxidation_states={"Li":1, "F":-1}, type="ionic")
    gen = IonicGenerator(config)
    structures = gen.generate()
    assert len(structures) > 0
    # Implementation returns NaCl dummy in the main block
    assert str(structures[0].symbols) == "NaCl"

def test_covalent_generator_mocked(mock_pyxtal):
    config = CovalentSystemConfig(elements=["C"], type="covalent")
    from nnp_gen.generators.covalent import CovalentGenerator
    gen = CovalentGenerator(config)
    structures = gen.generate()
    # Main block returns C2
    assert str(structures[0].symbols) == "C2"

def test_molecule_generator_mocked(mock_rdkit):
    config = MoleculeSystemConfig(elements=["H", "O"], smiles="H2O", type="molecule")
    from nnp_gen.generators.molecule import MoleculeGenerator
    gen = MoleculeGenerator(config)
    structures = gen.generate()
    # Main block returns H2O
    assert str(structures[0].symbols) == "H2O"

def test_fallback_when_modules_missing(mocker):
    # Ensure modules are treated as missing by setting them to None in sys.modules
    mocker.patch.dict("sys.modules", {
        "icet": None,
        "pymatgen": None,
        "pymatgen.core": None,
        "pyxtal": None,
        "rdkit": None,
        "rdkit.Chem": None
    })

    # Alloy
    config = AlloySystemConfig(elements=["Ag"], type="alloy")
    gen = AlloyGenerator(config)
    structures = gen.generate()
    # Fallback returns 1 atom of element[0] ("Ag")
    assert len(structures) == 1
    assert str(structures[0].symbols) == "Ag"
