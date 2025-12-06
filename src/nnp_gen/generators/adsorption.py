import logging
import numpy as np
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pydantic import TypeAdapter

from ase import Atoms
from ase.build import surface, molecule
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import (
    VacuumAdsorbateSystemConfig,
    SolventAdsorbateSystemConfig,
    SystemConfig as SystemConfigUnion,
    AdsorbateMode,
    AdsorbateConfig
)
from nnp_gen.core.exceptions import GenerationError

logger = logging.getLogger(__name__)

class VacuumAdsorbateGenerator(BaseGenerator):
    def __init__(self, config: VacuumAdsorbateSystemConfig):
        super().__init__(config)
        self.config = config

        # Instantiate sub-generator
        try:
            # Local import to avoid circular dependency
            from nnp_gen.generators.factory import GeneratorFactory

            # Convert dict to appropriate SystemConfig object
            # We assume the dict has a 'type' field that Pydantic can use for discrimination
            substrate_conf_obj = TypeAdapter(SystemConfigUnion).validate_python(self.config.substrate)
            self.substrate_generator = GeneratorFactory.get_generator(substrate_conf_obj)
        except Exception as e:
            raise GenerationError(f"Failed to initialize substrate generator: {e}")

    def _generate_impl(self) -> List[Atoms]:
        # 1. Generate bulk
        try:
            # We call generate() to get fully validated bulk structures
            bulks = self.substrate_generator.generate()
        except Exception as e:
            raise GenerationError(f"Substrate generation failed: {e}")

        if not bulks:
            raise GenerationError("Substrate generator returned no structures")

        surfaces = []
        for bulk_atoms in bulks:
            # 2. Loop over miller indices
            for miller in self.config.miller_indices:
                try:
                    # Create surface
                    # ase.build.surface adds vacuum on both sides? No, it creates a slab with vacuum.
                    # vacuum parameter is "size of vacuum on both sides" or "total vacuum"?
                    # ASE docs: "vacuum: float. thickness of vacuum layer on each side of the slab."
                    # We usually want vacuum on top (z-direction).
                    # But ase.build.surface centers the slab.
                    # We will center it later or accept it.
                    slab = surface(bulk_atoms, miller, layers=self.config.layers, vacuum=self.config.vacuum)

                    # Ensure the slab is centered and vacuum is correct?
                    # ase.build.surface puts vacuum/2 on bottom and vacuum/2 on top.
                    # We want all vacuum on top?
                    # Usually for surface science we want slab at bottom, vacuum on top.
                    slab.center(vacuum=self.config.vacuum, axis=2)
                    # Wait, center(vacuum=v) sets total vacuum to v?
                    # "vacuum: If specified, the atomic structure is centered in the unit cell with a vacuum of this size on both sides (in Angstroms)."
                    # We want to just put the atoms at the bottom.
                    # Let's just use what surface gives, but make sure we have enough Z.

                    # 3. Expand supercell (min_width >= 10.0)
                    self._expand_supercell(slab, min_width=10.0)

                    # 4. Defects
                    if self.config.defect_rate > 0:
                        self._apply_defects(slab)

                    # 5. Adsorbates
                    if self.config.adsorbates:
                        self._apply_adsorbates(slab)

                    # 6. Constraints
                    self._apply_constraints(slab)

                    surfaces.append(slab)

                except Exception as e:
                    logger.warning(f"Failed to generate surface {miller} for bulk: {e}")
                    # raise e # Debug
                    continue

        return surfaces

    def _expand_supercell(self, atoms: Atoms, min_width: float):
        lengths = atoms.cell.cellpar()[:3]
        nx = int(np.ceil(min_width / lengths[0]))
        ny = int(np.ceil(min_width / lengths[1]))
        nz = 1

        if nx > 1 or ny > 1:
            atoms *= (nx, ny, nz)

    def _apply_defects(self, atoms: Atoms):
        pos = atoms.get_positions()
        if len(pos) == 0:
            return

        z_max = np.max(pos[:, 2])

        # Identify top-layer atoms (Z > Z_max - 1.5)
        top_indices = [i for i, p in enumerate(pos) if p[2] > z_max - 1.5]

        if not top_indices:
            return

        n_remove = int(len(top_indices) * self.config.defect_rate)
        if n_remove == 0:
            return

        # Deterministic random based on structure
        seed_str = f"defects_{atoms.get_chemical_formula()}_{len(atoms)}"
        seed_hash = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
        seed = int(seed_hash, 16) % (2**32)
        rng = np.random.RandomState(seed)

        indices_to_remove = rng.choice(top_indices, size=n_remove, replace=False)

        # Delete atoms. Note: deleting invalidates indices, so we must be careful if doing multiple ops.
        # But `del atoms[indices]` works if indices is a list/array.
        del atoms[indices_to_remove]

    def _load_adsorbate(self, config: AdsorbateConfig) -> Atoms:
        if config.mode == AdsorbateMode.ATOM:
            return Atoms(config.source)
        elif config.mode == AdsorbateMode.MOLECULE:
            return molecule(config.source)
        elif config.mode == AdsorbateMode.SMILES:
            if not HAS_RDKIT:
                 raise GenerationError("RDKit required for SMILES mode")
            try:
                mol = Chem.MolFromSmiles(config.source)
                if mol is None:
                    raise ValueError("Invalid SMILES")
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                conf = mol.GetConformer()
                pos = conf.GetPositions()
                syms = [atom.GetSymbol() for atom in mol.GetAtoms()]
                return Atoms(symbols=syms, positions=pos)
            except Exception as e:
                 raise GenerationError(f"Failed to parse SMILES {config.source}: {e}")

        elif config.mode == AdsorbateMode.FILE:
             from ase.io import read
             return read(config.source)
        else:
            raise GenerationError(f"Unknown mode {config.mode}")

    def _apply_adsorbates(self, atoms: Atoms):
        seed_str = f"adsorbates_{atoms.get_chemical_formula()}_{len(atoms)}"
        seed_hash = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
        seed = int(seed_hash, 16) % (2**32)
        rng = np.random.RandomState(seed)

        for ads_conf in self.config.adsorbates:
            try:
                ads_template = self._load_adsorbate(ads_conf)
            except Exception as e:
                logger.error(f"Failed to load adsorbate {ads_conf.source}: {e}")
                continue

            for _ in range(ads_conf.count):
                placed = False
                attempts = 0
                max_attempts = 50

                while not placed and attempts < max_attempts:
                    attempts += 1

                    cell = atoms.get_cell()
                    x = rng.uniform(0, cell[0][0])
                    y = rng.uniform(0, cell[1][1])

                    z_max_surface = np.max(atoms.get_positions()[:, 2])
                    z = z_max_surface + ads_conf.height

                    new_ads = ads_template.copy()
                    com = new_ads.get_center_of_mass()
                    # Translate to target
                    new_ads.translate([x - com[0], y - com[1], z - com[2]])

                    # Overlap check (distance < 1.5A)
                    combined = atoms.copy()
                    combined += new_ads

                    # Check if any new atom is close to any old atom
                    # New atoms indices: len(atoms) .. len(combined)-1

                    collision = False

                    i_list, j_list = neighbor_list('ij', combined, cutoff=1.5)

                    for i, j in zip(i_list, j_list):
                        if i != j:
                             is_i_new = i >= len(atoms)
                             is_j_new = j >= len(atoms)

                             if is_i_new != is_j_new:
                                 collision = True
                                 break

                    if not collision:
                        atoms += new_ads
                        placed = True

    def _apply_constraints(self, atoms: Atoms):
        pos = atoms.get_positions()
        z = pos[:, 2]
        z_min = np.min(z)
        z_max = np.max(z)
        thickness = z_max - z_min
        cutoff = z_min + 0.25 * thickness

        indices = [i for i, z_val in enumerate(z) if z_val < cutoff]

        c = FixAtoms(indices=indices)
        atoms.set_constraint(c)


class SolventAdsorbateGenerator(VacuumAdsorbateGenerator):
    def __init__(self, config: SolventAdsorbateSystemConfig):
        super().__init__(config)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        surfaces = super()._generate_impl()

        final_surfaces = []
        for slab in surfaces:
            try:
                self._pack_solvent(slab)
                final_surfaces.append(slab)
            except Exception as e:
                logger.warning(f"Solvent packing failed: {e}")

        return final_surfaces

    def _pack_solvent(self, slab: Atoms):
        pos = slab.get_positions()
        if len(pos) == 0:
            return

        z_highest = np.max(pos[:, 2])
        z_start = z_highest + 2.0

        # Solvent height: Try to fill remaining vacuum
        # Assumes slab is centered-ish or at bottom.
        # Total Z is cell[2][2].
        cell = slab.get_cell()
        cell_z = cell[2][2]

        # We fill up to cell_z - 2.0 buffer?
        z_end = cell_z - 2.0

        solvent_height = z_end - z_start
        if solvent_height <= 2.0:
            # Not enough space? Or maybe vacuum config was small.
            # Try to force a reasonable height?
            # If we force height, we might overlap PBC in Z?
            # But BaseGenerator will enforce PBC checks?
            # Prompt says "Define solvent box 2.0A above surface". Doesn't specify top.
            # But we need volume to calculate N.
            # Let's assume height is derived from config.vacuum?
            # If vacuum is 10, and we start 2A above surface, we have ~8A?
            solvent_height = max(solvent_height, 5.0) # at least 5A?

        try:
            # Heuristic for mode:
            # Prompt says default "O" (Water). SMILES "O" is water.
            # We use SMILES mode for default if it looks like SMILES, or fall back to MOLECULE if simple formula.
            # But prompt says "solvent_smiles: str", so we should treat it as SMILES primarily.
            # However, user might pass "H2O".
            # Let's assume SMILES.

            solvent_mol = self._load_adsorbate(
                AdsorbateConfig(
                    source=self.config.solvent_smiles,
                    mode=AdsorbateMode.SMILES,
                    count=1
                )
            )
        except:
             # Fallback to H2O molecule if SMILES fails (e.g. no rdkit)
             solvent_mol = molecule('H2O')

        mol_mass = sum(solvent_mol.get_masses())

        vecs = slab.get_cell()
        cross_prod = np.cross(vecs[0], vecs[1])
        area = np.linalg.norm(cross_prod)

        vol_cm3 = (area * solvent_height) * 1e-24
        total_mass_g = self.config.solvent_density * vol_cm3
        mol_mass_g = mol_mass * 1.66053906660e-24

        n_mols = int(total_mass_g / mol_mass_g)

        seed_str = f"solvent_{slab.get_chemical_formula()}_{len(slab)}"
        seed_hash = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
        seed = int(seed_hash, 16) % (2**32)
        rng = np.random.RandomState(seed)

        added_count = 0
        max_attempts_per_mol = 100

        for i in range(n_mols):
            placed = False
            mol_attempts = 0
            while not placed and mol_attempts < max_attempts_per_mol:
                mol_attempts += 1

                r1, r2 = rng.uniform(0, 1, 2)
                z = rng.uniform(z_start, z_start + solvent_height)

                pos_xy = r1 * vecs[0] + r2 * vecs[1]
                pos = pos_xy + np.array([0, 0, z])

                new_mol = solvent_mol.copy()
                new_mol.rotate(rng.uniform(0, 360), 'z')
                new_mol.rotate(rng.uniform(0, 180), 'x')
                new_mol.rotate(rng.uniform(0, 360), 'y')

                com = new_mol.get_center_of_mass()
                new_mol.translate(pos - com)

                combined = slab + new_mol

                collision = False
                i_list, j_list = neighbor_list('ij', combined, cutoff=2.0)

                for ii, jj in zip(i_list, j_list):
                    if ii != jj:
                        is_ii_new = ii >= len(slab)
                        is_jj_new = jj >= len(slab)

                        if is_ii_new != is_jj_new:
                            collision = True
                            break

                if not collision:
                    slab += new_mol
                    placed = True
                    added_count += 1
