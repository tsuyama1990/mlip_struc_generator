import logging
from typing import List, Optional
from ase import Atoms
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import MoleculeSystemConfig

logger = logging.getLogger(__name__)

class MoleculeGenerator(BaseGenerator):
    def __init__(self, config: MoleculeSystemConfig, seed: Optional[int] = None):
        super().__init__(config, seed=seed)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates molecular conformers using RDKit.
        """
        logger.info(f"Generating conformers for SMILES: {self.config.smiles}")

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            logger.error("rdkit not installed.")
            raise ImportError("rdkit is required for MoleculeGenerator")

        mol = Chem.MolFromSmiles(self.config.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {self.config.smiles}")

        mol = Chem.AddHs(mol)

        # Generate conformers
        n_confs = self.config.num_conformers
        # Try ETKDG first
        try:
            params = AllChem.ETKDG()
            params.randomSeed = 42 # For reproducibility if possible
            res = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        except Exception as e:
            logger.warning(f"ETKDG failed: {e}")
            res = []

        if not res:
            logger.warning("Could not generate conformers with ETKDG, trying random coords.")
            res = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, useRandomCoords=True)

        structures = []
        elements = [atom.GetSymbol() for atom in mol.GetAtoms()]

        for conf_id in res:
            conf = mol.GetConformer(conf_id)
            positions = conf.GetPositions()
            atoms = Atoms(symbols=elements, positions=positions)
            # Molecules generally don't have PBC, but BaseGenerator sets it from config.
            # Ensure pbc is set correctly in config (MoleculeSystemConfig defaults to False)
            structures.append(atoms)

        if not structures:
             logger.warning("No conformers generated.")

        return structures
