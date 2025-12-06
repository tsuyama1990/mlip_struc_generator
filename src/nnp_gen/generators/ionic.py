import logging
import numpy as np
from typing import List, Optional, Dict
from ase import Atoms
from ase.build import bulk
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import IonicSystemConfig
from nnp_gen.core.exceptions import GenerationError

logger = logging.getLogger(__name__)

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

class IonicGenerator(BaseGenerator):
    def __init__(self, config: IonicSystemConfig):
        super().__init__(config)
        self.config = config

        # Validate elements
        clean_elements = []
        for el in self.config.elements:
             try:
                 clean_elements.append(validate_element(el))
             except ValueError as e:
                 logger.error(str(e))
                 raise GenerationError(str(e))
        # Update config with sanitized elements (assuming mutable)
        self.config.elements = clean_elements

        # Check for optional dependencies
        try:
            import pymatgen.core as pmg
            from pymatgen.symmetry.structure import SymmetrizedStructure
            from pymatgen.core import Lattice, Species, Element
            self.pmg = pmg
            self.Species = Species
            self.Lattice = Lattice
            self.Element = Element
            self.has_pmg = True
        except ImportError:
            self.has_pmg = False
            self.pmg = None
            self.Lattice = None
            self.Species = None
            self.Element = None
            logger.warning("pymatgen not installed. IonicGenerator functionality will be severely limited.")

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates ionic structures.
        Supports binary compounds with various prototypes using pymatgen if available.
        """
        logger.info(f"Generating ionic structures for {self.config.elements}")

        for el in self.config.elements:
            if el not in self.config.oxidation_states:
                logger.error(f"Oxidation state for {el} not defined.")
                raise GenerationError(f"Oxidation state for {el} missing")

        structures = []

        if self.has_pmg:
            structures.extend(self._generate_with_pymatgen())
        else:
            # Fallback to simple ASE generation for specific cases or raise error
            structures.extend(self._generate_fallback())

        if not structures:
             raise GenerationError("No structures could be generated with the current configuration.")

        return structures

    def _get_heuristic_radius(self, species_str: str, charge: Optional[int] = None) -> float:
        """
        Get the radius of a species for lattice constant estimation using a tiered heuristic.

        Tier 1: Shannon effective ionic radii (via pymatgen).
        Tier 2: Electronegativity-scaled fallback (TODO: refined scaling), currently standard atomic/ionic.
        Tier 3: Geometric considerations are handled in _create_binary_prototypes.
        """
        if not self.has_pmg:
             # Tier 2 Fallback (No pymatgen): ASE covalent radii
             # Simple heuristic: for ions, covalent radii are often too large for cations
             # and too small for anions, but we lack data to correct it properly without a table.
             return covalent_radii[atomic_numbers[species_str]]

        # Tier 1: Oxidation State Inference / Shannon Radii
        if charge is not None:
            try:
                # Use Species class to get Shannon radius
                # Using coordination number 6 ("VI") as a standard reference for rocksalt
                sp = self.Species(species_str, oxidation_state=charge)
                radius = sp.get_shannon_radius(cn="VI")
                if radius:
                    return radius
            except Exception:
                pass # Fall through to next tier

        el = self.Element(species_str)

        # Tier 2: Fallbacks

        # 2a. Average ionic radius (if available in pymatgen data)
        if el.average_ionic_radius:
            return el.average_ionic_radius

        # 2b. Atomic radius (often covalent)
        r_atomic = el.atomic_radius
        if r_atomic:
            # Electronegativity correction could go here if we had a target structure context
            # For now, we warn if we are deep in fallback territory for an ionic generator
            logger.debug(f"Using atomic radius for {species_str} (fallback)")
            return r_atomic

        # 2c. Ultimate fallback
        return 1.5

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
            radii_map[el_str] = self._get_heuristic_radius(el_str, charge)

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
