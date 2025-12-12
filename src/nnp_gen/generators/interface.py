import logging
import numpy as np
from typing import List, Optional
from ase import Atoms
from ase.build import make_supercell, stack
from ase.constraints import FixAtoms
from pydantic import TypeAdapter

from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import InterfaceSystemConfig, SystemConfig
from nnp_gen.core.exceptions import GenerationError

logger = logging.getLogger(__name__)

class InterfaceGenerator(BaseGenerator):
    def __init__(self, config: InterfaceSystemConfig, seed: int = 42):
        # We initialize BaseGenerator normally
        super().__init__(config, seed=seed)
        self.config = config

        # 1. Parse Sub-Configs
        # We need to convert the dictionaries back into Pydantic models
        # This allows the Factory to read them correctly.
        # We assume phase_a/b are raw dicts from YAML.
        # We use the Pydantic adapter to validate them into specific SystemConfigs.
        adapter = TypeAdapter(SystemConfig)
        try:
            self.config_a = adapter.validate_python(config.phase_a)
            self.config_b = adapter.validate_python(config.phase_b)
        except Exception as e:
            raise GenerationError(f"Failed to validate sub-configurations: {e}")

        # 2. Instantiate Sub-Generators
        # Local import to avoid circular dependency
        from nnp_gen.generators.factory import GeneratorFactory

        self.gen_a = GeneratorFactory.get_generator(self.config_a)
        self.gen_b = GeneratorFactory.get_generator(self.config_b)

    def _generate_impl(self) -> List[Atoms]:
        logger.info(f"Generating Interface: Mode={self.config.mode}")

        # A. Generate Candidates for both phases
        # We generate a batch of candidates for optimization
        # For interface generation, we typically want the base structures without random rattle/strain first?
        # But BaseGenerator.generate() applies rattle/strain.
        # Since gen_a and gen_b are BaseGenerators, calling generate() on them triggers their full pipeline
        # including validation.

        structs_a = self.gen_a.generate()
        structs_b = self.gen_b.generate()

        if not structs_a or not structs_b:
            raise GenerationError("Sub-generators produced no valid structures.")

        # B. Fuse them
        interfaces = []

        # For this example, we take the best/first candidate from each.
        # sophisticated implementation would loop through combinations.
        slab = structs_a[0]
        film = structs_b[0]

        if self.config.mode == "hetero_crystal":
            interfaces = self._build_hetero_junction(slab, film)
        elif self.config.mode == "solid_liquid":
            interfaces = self._build_solid_liquid(slab, film)

        return interfaces

    def _build_hetero_junction(self, slab: Atoms, film: Atoms) -> List[Atoms]:
        """
        Builds a commensurately matched solid-solid interface.
        """
        # 1. Lattice Matching Logic (Zur-McGill Algorithm simplified)
        # Calculate surface areas
        # Assumes slab and film are oriented with Z perpendicular to surface

        # Check if cell is defined
        if slab.cell is None or film.cell is None:
             raise GenerationError("Structures must have cells defined for hetero_crystal mode.")

        area_a = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
        area_b = np.linalg.norm(np.cross(film.cell[0], film.cell[1]))

        # Find integer supercells that match areas
        # (This is a heuristic placeholder for full matrix matching)
        if area_b < 1e-6:
             raise GenerationError("Film area is too small.")

        ratio = np.sqrt(area_a / area_b)
        N = int(np.round(ratio))
        if N < 1: N = 1

        # Create supercells
        # Slab is assumed to be the "substrate" so we keep it 1x1 or scale up if needed,
        # but here we scale film to match slab.
        # Wait, if ratio > 1, slab is larger than film. Film needs N.
        # If ratio < 1, film is larger than slab. Slab might need supercell.
        # The user example simplified this: slab [1,1], film [N, N].
        # Let's stick to the user's simplified logic for now but be aware of limitations.

        slab_super = make_supercell(slab, [[1,0,0],[0,1,0],[0,0,1]])
        film_super = make_supercell(film, [[N,0,0],[0,N,0],[0,0,1]])

        # 2. Strain Engineering
        # We strain the FILM to match the SLAB
        cell_target = slab_super.get_cell()
        cell_current = film_super.get_cell()

        # Calculate strain tensor (diagonal approximation)
        if cell_current[0,0] == 0 or cell_current[1,1] == 0:
             raise GenerationError("Film cell has zero dimension.")

        strain_x = cell_target[0,0] / cell_current[0,0]
        strain_y = cell_target[1,1] / cell_current[1,1]

        if abs(strain_x - 1.0) > self.config.max_mismatch:
            logger.warning(f"Skipping: Strain X {strain_x:.2f} exceeds limit {self.config.max_mismatch}")
            return []

        if abs(strain_y - 1.0) > self.config.max_mismatch:
             logger.warning(f"Skipping: Strain Y {strain_y:.2f} exceeds limit {self.config.max_mismatch}")
             return []

        # Apply affine transformation to film
        new_cell = np.array([
            [cell_current[0,0] * strain_x, 0, 0],
            [0, cell_current[1,1] * strain_y, 0],
            cell_current[2]
        ])
        film_super.set_cell(new_cell, scale_atoms=True)

        # 3. Stacking
        combined = stack(slab_super, film_super, axis=2, distance=self.config.interface_distance)

        # 4. Vacuum & Centering
        combined.center(vacuum=self.config.vacuum / 2, axis=2)

        # 5. Tagging for DFT
        # Important: Fix the bottom layers of the slab to simulate bulk
        # We fix the bottom 50% of the slab atoms
        z_positions = combined.positions[:, 2]
        min_z = np.min(z_positions)

        # Calculate slab height carefully.
        # The atoms from slab are the first len(slab_super) atoms?
        # stack() usually puts atoms1 then atoms2.
        # Yes, slab_super comes first.

        slab_indices = range(len(slab_super))
        slab_z = combined.positions[slab_indices, 2]
        slab_height = np.max(slab_z) - np.min(slab_z)

        fixed_indices = [atom.index for atom in combined
                         if atom.index < len(slab_super) and atom.position[2] < min_z + (slab_height * 0.5)]

        combined.set_constraint(FixAtoms(indices=fixed_indices))

        return [combined]

    def _build_solid_liquid(self, slab: Atoms, molecule: Atoms) -> List[Atoms]:
        """
        Grid-based packing of solvent molecules.
        """
        # 1. Expand Slab to a minimum size (e.g. 10x10 A) to avoid self-interaction of solvent
        min_dim = 10.0
        cell_norms = np.linalg.norm(slab.cell, axis=1)
        if cell_norms[0] < 1e-3 or cell_norms[1] < 1e-3:
             raise GenerationError("Slab cell dimensions are too small or zero.")

        n_x = int(np.ceil(min_dim / cell_norms[0]))
        n_y = int(np.ceil(min_dim / cell_norms[1]))

        # Make sure we have at least 1x1
        n_x = max(1, n_x)
        n_y = max(1, n_y)

        slab_super = make_supercell(slab, [[n_x,0,0], [0,n_y,0], [0,0,1]])

        # 2. Define Filling Region
        # We define a box above the surface
        cell = slab_super.get_cell()
        lx, ly = cell[0,0], cell[1,1]
        z_surface = np.max(slab_super.positions[:, 2])
        z_start = z_surface + self.config.interface_distance

        # Calculate number of molecules needed for target density
        mol_mass_amu = np.sum(molecule.get_masses())
        target_height = 12.0 # Angstroms of liquid layer
        vol_cm3 = (lx * ly * target_height) * 1e-24
        mass_g = self.config.solvent_density * vol_cm3

        if mol_mass_amu <= 0:
             raise GenerationError("Molecule has zero mass.")

        n_mols = int((mass_g / 1.66053906660e-24) / mol_mass_amu)

        logger.info(f"Packing {n_mols} solvent molecules into {lx:.1f}x{ly:.1f}x{target_height} box")

        # 3. Grid Insertion Strategy (More robust than random)
        # We create a 3D grid of potential sites
        combined = slab_super.copy()

        # Determine grid spacing based on molecule size (approx radius 2.0A)
        grid_spacing = 3.0
        x_sites = np.arange(0, lx, grid_spacing)
        y_sites = np.arange(0, ly, grid_spacing)
        z_sites = np.arange(z_start, z_start + target_height, grid_spacing)

        sites = []
        for z in z_sites:
            for y in y_sites:
                for x in x_sites:
                    sites.append([x, y, z])

        # Shuffle sites to avoid ordered artifacts
        rng = np.random.RandomState(42) # Use fixed seed or self.config.seed? BaseGenerator doesn't expose seed directly here
        rng.shuffle(sites)

        # Insert molecules
        added = 0

        # To optimize, we can build a list of atoms and extend once.
        molecules_to_add = []

        for site in sites:
            if added >= n_mols: break

            mol_copy = molecule.copy()
            # Random Rotation
            mol_copy.rotate(rng.rand() * 360, 'z')
            mol_copy.rotate(rng.rand() * 180, 'x')

            # Translate to site - center of mass?
            # molecule.center() centers it in its own cell usually, but here we just want to move positions
            mol_center = mol_copy.get_center_of_mass()
            translation = np.array(site) - mol_center
            mol_copy.translate(translation)

            # Simple check could be added here, but grid spacing gives basic assurance
            molecules_to_add.append(mol_copy)
            added += 1

        for mol in molecules_to_add:
            combined += mol

        # 4. Set final cell
        total_height = z_start + target_height + self.config.vacuum
        combined.set_cell([lx, ly, total_height])
        combined.set_pbc([True, True, True])

        return [combined]
