import pytest
import sys
import numpy as np
from unittest.mock import MagicMock
from ase import Atoms
from nnp_gen.core.config import AlloySystemConfig, IonicSystemConfig, CovalentSystemConfig, MoleculeSystemConfig, PhysicsConstraints
from nnp_gen.generators.factory import GeneratorFactory
from nnp_gen.generators.alloy import AlloyGenerator
from nnp_gen.generators.ionic import IonicGenerator
from nnp_gen.generators.covalent import CovalentGenerator
from nnp_gen.generators.molecule import MoleculeGenerator

@pytest.fixture
def mock_pyxtal(mocker):
    mock = MagicMock()
    # mock is the module 'pyxtal'

    # We need to mock the class 'pyxtal' inside the module
    # When `from pyxtal import pyxtal` is called, it gets `mock.pyxtal`.
    # So we configure `mock.pyxtal`
    pyxtal_class = MagicMock()
    mock.pyxtal = pyxtal_class

    # Instance created by pyxtal()
    instance = pyxtal_class.return_value
    instance.valid = True
    # to_ase returns a real Atoms object for valid assertion
    instance.to_ase.return_value = Atoms('C2', positions=[[0,0,0], [1.5,0,0]], cell=[3,3,3], pbc=True)

    mocker.patch.dict("sys.modules", {"pyxtal": mock})
    return mock

@pytest.fixture
def mock_rdkit(mocker):
    mock = MagicMock()
    # Mock Chem
    mol = MagicMock()
    mock.Chem.MolFromSmiles.return_value = mol
    mock.Chem.AddHs.return_value = mol

    # Atoms in mol
    atom1 = MagicMock()
    atom1.GetSymbol.return_value = 'H'
    atom2 = MagicMock()
    atom2.GetSymbol.return_value = 'O'
    atom3 = MagicMock()
    atom3.GetSymbol.return_value = 'H'
    mol.GetAtoms.return_value = [atom1, atom2, atom3]

    # Embed
    mock.Chem.AllChem.EmbedMultipleConfs.return_value = [0]

    # Conformer
    conf = MagicMock()
    conf.GetPositions.return_value = np.array([[0.,0.,0.], [0.,0.,1.], [0.,1.,0.]])
    mol.GetConformer.return_value = conf

    mocker.patch.dict("sys.modules", {"rdkit": mock, "rdkit.Chem": mock.Chem, "rdkit.Chem.AllChem": mock.Chem.AllChem})
    return mock

def test_factory_creation():
    config = AlloySystemConfig(elements=["Cu", "Zr"], type="alloy")
    gen = GeneratorFactory.get_generator(config)
    assert isinstance(gen, AlloyGenerator)

    config_ionic = IonicSystemConfig(elements=["Na", "Cl"], oxidation_states={"Na":1, "Cl":-1}, type="ionic")
    gen_ionic = GeneratorFactory.get_generator(config_ionic)
    assert isinstance(gen_ionic, IonicGenerator)

def test_alloy_generator_real():
    # AlloyGenerator uses ASE now
    config = AlloySystemConfig(
        elements=["Cu", "Au"],
        type="alloy",
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=0.1)
    )
    gen = AlloyGenerator(config)
    structures = gen.generate()

    assert len(structures) > 0
    syms = structures[0].get_chemical_symbols()
    # Should contain Cu or Au (random)
    assert any(s in syms for s in ["Cu", "Au"])

def test_ionic_generator_real():
    # Ionic uses ASE now
    config = IonicSystemConfig(
        elements=["Li", "F"],
        oxidation_states={"Li":1, "F":-1},
        type="ionic",
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=0.1)
    )
    gen = IonicGenerator(config)
    structures = gen.generate()

    assert len(structures) > 0
    syms = structures[0].get_chemical_symbols()
    # Expect Li and F
    assert "Li" in syms and "F" in syms

def test_covalent_generator_mocked(mock_pyxtal):
    config = CovalentSystemConfig(
        elements=["C"],
        type="covalent",
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=0.1)
    )
    gen = CovalentGenerator(config)
    structures = gen.generate()

    assert len(structures) > 0
    # The mock returns C2
    assert "C" in str(structures[0].symbols)

def test_molecule_generator_mocked(mock_rdkit):
    config = MoleculeSystemConfig(
        elements=["H", "O"],
        smiles="H2O",
        type="molecule",
        constraints=PhysicsConstraints(min_cell_length_factor=0.0, r_cut=0.1)
    )
    gen = MoleculeGenerator(config)
    structures = gen.generate()

    assert len(structures) > 0
    assert len(structures[0]) == 3

def test_importerror_when_modules_missing(mocker):
    mocker.patch.dict("sys.modules", {
        "pyxtal": None,
        "rdkit": None,
        "rdkit.Chem": None
    })

    # Covalent
    config_cov = CovalentSystemConfig(elements=["C"], type="covalent")
    gen_cov = CovalentGenerator(config_cov)
    with pytest.raises(ImportError):
        gen_cov.generate()

    # Molecule
    config_mol = MoleculeSystemConfig(elements=["H"], smiles="H2", type="molecule")
    gen_mol = MoleculeGenerator(config_mol)
    with pytest.raises(ImportError):
        gen_mol.generate()
