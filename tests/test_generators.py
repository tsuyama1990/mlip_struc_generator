import pytest
import sys
from unittest.mock import MagicMock
from nnp_gen.core.config import AlloySystemConfig, IonicSystemConfig, CovalentSystemConfig, MoleculeSystemConfig, PhysicsConstraints
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
    """Test that AlloyGenerator imports icet and returns structure."""
    config = AlloySystemConfig(
        elements=["Cu", "Au"],
        type="alloy",
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=0.1)
    )
    gen = AlloyGenerator(config)

    structures = gen.generate()
    # Mock path returns Cu4. With no expansion, it stays Cu4
    assert str(structures[0].symbols) == "Cu4"

def test_ionic_generator_mocked(mock_pymatgen):
    config = IonicSystemConfig(
        elements=["Li", "F"],
        oxidation_states={"Li":1, "F":-1},
        type="ionic",
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=0.1)
    )
    gen = IonicGenerator(config)
    structures = gen.generate()
    assert str(structures[0].symbols) == "NaCl"

def test_covalent_generator_mocked(mock_pyxtal):
    config = CovalentSystemConfig(
        elements=["C"],
        type="covalent",
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=0.1)
    )
    from nnp_gen.generators.covalent import CovalentGenerator
    gen = CovalentGenerator(config)
    structures = gen.generate()
    assert str(structures[0].symbols) == "C2"

def test_molecule_generator_mocked(mock_rdkit):
    config = MoleculeSystemConfig(
        elements=["H", "O"],
        smiles="H2O",
        type="molecule",
        # Molecule usually pbc=False so supercell logic skipped anyway, but safe to set constraints
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=0.1)
    )
    from nnp_gen.generators.molecule import MoleculeGenerator
    gen = MoleculeGenerator(config)
    structures = gen.generate()
    assert str(structures[0].symbols) == "H2O"

def test_importerror_when_modules_missing(mocker):
    """Test that missing libraries raise ImportError or ModuleNotFoundError."""
    mocker.patch.dict("sys.modules", {
        "icet": None,
        "pymatgen": None,
        "pyxtal": None,
        "rdkit": None,
        "rdkit.Chem": None
    })

    # Alloy (needs icet)
    config = AlloySystemConfig(elements=["Ag"], type="alloy")
    gen = AlloyGenerator(config)
    with pytest.raises((ImportError, ModuleNotFoundError)):
        gen.generate()

    # Ionic (needs pymatgen)
    config_ionic = IonicSystemConfig(elements=["Na", "Cl"], oxidation_states={"Na":1, "Cl":-1}, type="ionic")
    gen_ionic = IonicGenerator(config_ionic)
    with pytest.raises((ImportError, ModuleNotFoundError)):
        gen_ionic.generate()
