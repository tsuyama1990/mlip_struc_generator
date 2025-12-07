import logging
import numpy as np
from typing import List, Optional, Dict
from ase import Atoms
from ase.build import bulk
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import IonicSystemConfig
from nnp_gen.core.physics import apply_vacancies
from nnp_gen.core.exceptions import GenerationError

logger = logging.getLogger(__name__)

DefaultOxidationStates = {
    'Li': +1, 'Na': +1, 'K': +1, 'Rb': +1, 'Cs': +1,
    'Be': +2, 'Mg': +2, 'Ca': +2, 'Sr': +2, 'Ba': +2,
    'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
    'O': -2, 'S': -2, 'Se': -2, 'Te': -2,
    'N': -3, 'P': -3,
    'Al': +3, 'Ga': +3, 'In': +3,
    'Zn': +2, 'Cd': +2,
    'Ag': +1,
    'Sc': +3, 'Y': +3, 'La': +3
}

def validate_element(el: str) -> str:
    """
    Validates and sanitizes an element symbol.
    """
    if not isinstance(el, str):
         raise ValueError(f"Element must be a string, got {type(el)}")
    el_clean = el.strip().capitalize()
    if el_clean == 'X':
        raise ValueError(f"Invalid element symbol: {el} ('X' is a dummy element)")
    if el_clean not in chemical_symbols:
        # chemical_symbols contains 'X' as a dummy sometimes, but usually proper elements.
        # atomic_numbers is a safer check if we want real elements.
        # But prompt asked for chemical_symbols.
        raise ValueError(f"Invalid element symbol: {el}")
    return el_clean

# --- Radius Strategies ---

class RadiusStrategy:
    def get_radius(self, species: str, charge: Optional[int] = None) -> float:
        raise NotImplementedError

class PymatgenRadiusStrategy(RadiusStrategy):
    def __init__(self):
        try:
            import pymatgen.core as pmg
            from pymatgen.core import Species, Element
            self.Species = Species
            self.Element = Element
        except ImportError:
            raise ImportError("Pymatgen not installed")

    def get_radius(self, species: str, charge: Optional[int] = None) -> float:
        # Tier 1: Oxidation State Inference / Shannon Radii
        if charge is not None:
            try:
                sp = self.Species(species, oxidation_state=charge)
                radius = sp.get_shannon_radius(cn="VI")
                if radius:
                    return radius
            except Exception:
                pass 

        # Tier 2: Fallback to Atomic Radius
        el = self.Element(species)
        if el.average_ionic_radius:
            return el.average_ionic_radius
        
        if el.atomic_radius:
            return el.atomic_radius
            
        raise GenerationError(f"Could not determine radius for species {species}")

class FallbackRadiusStrategy(RadiusStrategy):
    def get_radius(self, species: str, charge: Optional[int] = None) -> float:
        # Simple heuristic using ASE covalent radii
        # This is not physically accurate for ions but allows code to run without pymatgen
        try:
            return covalent_radii[atomic_numbers[species]]
        except KeyError:
             raise GenerationError(f"Unknown element: {species}")


class IonicGenerator(BaseGenerator):
    def __init__(self, config: IonicSystemConfig, seed: Optional[int] = None):
        super().__init__(config, seed=seed)
        self.config = config

        # Validate elements
        clean_elements = []
        for el in self.config.elements:
             try:
                 clean_elements.append(validate_element(el))
             except ValueError as e:
                 logger.error(str(e))
                 raise GenerationError(str(e))
        self.config.elements = clean_elements

        # Select Radius Strategy
        self.strategy: RadiusStrategy
        self.has_pmg = False
        
        if not getattr(self.config, 'strict_mode', True):
             # Try to import but allow failure
             try:
                 self.strategy = PymatgenRadiusStrategy()
                 self.has_pmg = True
                 self._setup_pmg() # Helper to setup pmg internal refs if needed for prototype generation
             except ImportError:
                 self.strategy = FallbackRadiusStrategy()
                 logger.warning("Pymatgen not installed. Using fallback radius strategy (Covalent Radii). Accuracy reduced.")
        else:
             # Strict mode: Force pymatgen
             try:
                 self.strategy = PymatgenRadiusStrategy()
                 self.has_pmg = True
                 self._setup_pmg()
             except ImportError:
                 raise ImportError("Pymatgen required for Ionic Generation in strict mode. Install it or disable strict_mode.")

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates ionic structures.
        Supports binary compounds with various prototypes using pymatgen if available.
        """
        logger.info(f"Generating ionic structures for {self.config.elements}")

        # Smart Oxidation State Inference
        for el in self.config.elements:
            if el not in self.config.oxidation_states:
                if el in DefaultOxidationStates:
                    logger.info(f"Using default oxidation state for {el}: {DefaultOxidationStates[el]}")
                    self.config.oxidation_states[el] = DefaultOxidationStates[el]
                else:
                    logger.error(f"Oxidation state for {el} is ambiguous. Please specify in config.")
                    raise GenerationError(f"Oxidation state for {el} missing and no default available.")

        structures = []

        if self.has_pmg:
            structures.extend(self._generate_with_pymatgen())
        else:
            # Fallback to simple ASE generation for specific cases or raise error
            structures.extend(self._generate_fallback())

        if not structures:
             raise GenerationError("No structures could be generated with the current configuration.")

        # Apply Post-Processing: Vacancies and Charges
        seed_val = self.seed if self.seed is not None else 42
        rng = np.random.RandomState(seed_val)

        final_structures = []
        for atoms in structures:
            # 1. Set Initial Charges
            self._set_initial_charges(atoms)

            # 2. Apply Vacancies
            if self.config.vacancy_concentration > 0.0:
                atoms = apply_vacancies(atoms, self.config.vacancy_concentration, rng)

            final_structures.append(atoms)

        return final_structures

    def _set_initial_charges(self, atoms: Atoms):
        """
        Set initial charges on atoms based on oxidation states.
        """
        charges = []
        oxi_states = self.config.oxidation_states

        for atom in atoms:
            sym = atom.symbol
            q = oxi_states.get(sym, 0.0)
            charges.append(q)

        atoms.set_initial_charges(charges)

    def _setup_pmg(self):
        import pymatgen.core as pmg
        from pymatgen.core import Lattice
        self.pmg = pmg
        self.Lattice = Lattice

    def _get_radius(self, species: str, charge: Optional[int] = None) -> float:
        return self.strategy.get_radius(species, charge)

    def _generate_with_pymatgen(self) -> List[Atoms]:
        """
        Use pymatgen to replace species in prototype structures.
        """
        generated = []
        elements = self.config.elements
        oxidation_states = self.config.oxidation_states

        # Get radii to estimate lattice constant
        radii_map = {}
        for el_str in elements:
            charge = oxidation_states.get(el_str)
            radii_map[el_str] = self._get_radius(el_str, charge)

        logger.info(f"Using radii for generation: {radii_map}")

        if len(elements) == 2:
            el1, el2 = elements
            q1 = oxidation_states[el1]
            q2 = oxidation_states[el2]

            radii_sum = radii_map[el1] + radii_map[el2]

            # 1. Rocksalt / CsCl / Zincblende (1:1)
            if abs(q1) == abs(q2):
                generated.extend(self._create_binary_prototypes(el1, el2, radii_sum, ["rocksalt", "cscl", "zincblende"]))

            # 2. Fluorite / Antifluorite (1:2 or 2:1)
            elif abs(q1) == 2 * abs(q2):
                # AB2
                generated.extend(self._create_binary_prototypes(el1, el2, radii_sum, ["fluorite"]))
            elif 2 * abs(q1) == abs(q2):
                # A2B (Anti-fluorite structure is fluorite with swapped positions)
                generated.extend(self._create_binary_prototypes(el2, el1, radii_sum, ["fluorite"]))

        return generated

    def _create_binary_prototypes(self, species_a: str, species_b: str, radii_sum: float, types: List[str]) -> List[Atoms]:
        structures = []

        # Tier 3: Geometric Debug / Correct Pre-factors
        # Ensure correct a = f(r) math

        for t in types:
            try:
                struct = None
                if t == "rocksalt":
                    # For Rocksalt (face-centered cubic, Fm-3m)
                    # Nearest neighbor distance d = r_A + r_B = radii_sum
                    # Lattice constant a = 2 * d
                    a = 2.0 * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("Fm-3m", self.Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.5,0.5,0.5]])

                elif t == "cscl":
                    # CsCl (Simple Cubic based, Pm-3m)
                    # Nearest neighbor (body diagonal) d = sqrt(3)/2 * a
                    # => a = 2/sqrt(3) * d
                    a = (2.0 / np.sqrt(3.0)) * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("Pm-3m", self.Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.5,0.5,0.5]])

                elif t == "zincblende":
                    # Zincblende (F-43m)
                    # Nearest neighbor (quarter body diagonal) d = sqrt(3)/4 * a
                    # => a = 4/sqrt(3) * d
                    a = (4.0 / np.sqrt(3.0)) * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("F-43m", self.Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.25,0.25,0.25]])

                elif t == "fluorite":
                    # Fluorite CaF2 (Fm-3m)
                    # Nearest neighbor distance d (Ca-F) is sqrt(3)/4 * a
                    # => a = 4/sqrt(3) * d
                    a = (4.0 / np.sqrt(3.0)) * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("Fm-3m", self.Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.25,0.25,0.25]])

                if struct:
                    atoms = self._pmg_to_ase(struct)

                    # Note on Supercells:
                    # We generate the unit cell above with physically correct 'a'.
                    # However, MLIP training often requires a minimum simulation box size (r_cut).
                    # The BaseGenerator pipeline will apply `ensure_supercell_size` later,
                    # which may expand this unit cell. This is intended behavior.

                    # We also apply explicit supercell expansion from config here if requested,
                    # though arguably this should also be handled by the pipeline or configured carefully.
                    # Current logic: apply config-based expansion.
                    atoms = atoms * self.config.supercell_size
                    structures.append(atoms)
            except Exception as e:
                logger.warning(f"Failed to generate {t} for {species_a}-{species_b}: {e}")

        return structures

    def _pmg_to_ase(self, struct) -> Atoms:
        """Convert pymatgen Structure to ASE Atoms"""
        symbols = [str(s) for s in struct.species]
        positions = struct.cart_coords
        cell = struct.lattice.matrix
        pbc = [True, True, True]
        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

    def _generate_fallback(self) -> List[Atoms]:
        """
        Fallback implementation using ASE bulk if pymatgen is missing.
        """
        logger.info("Using ASE fallback generation.")
        structures = []
        elements = self.config.elements

        if len(elements) == 2:
            cation, anion = elements

            # Use ASE covalent radii
            r1 = covalent_radii[atomic_numbers[cation]]
            r2 = covalent_radii[atomic_numbers[anion]]

            # Estimate lattice constant for Rocksalt
            # a = 2 * (r1 + r2)
            a = 2.0 * (r1 + r2)

            logger.info(f"Fallback: Generating Rocksalt for {cation}-{anion} with a={a:.2f}")

            try:
                # 'NaCl' is just a prototype in ase.build.bulk
                prim = bulk('NaCl', 'rocksalt', a=a)
                new_symbols = []
                for s in prim.get_chemical_symbols():
                    if s == 'Na':
                        new_symbols.append(cation)
                    else:
                        new_symbols.append(anion)
                prim.set_chemical_symbols(new_symbols)
                atoms = prim * self.config.supercell_size
                structures.append(atoms)
            except Exception as e:
                 logger.error(f"Fallback generation failed: {e}")
                 raise GenerationError(f"Fallback generation failed for {elements}: {e}")

        return structures
