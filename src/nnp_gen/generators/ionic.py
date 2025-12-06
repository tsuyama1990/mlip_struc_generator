import logging
import numpy as np
from typing import List, Optional
from ase import Atoms
from ase.build import bulk
from ase.data import covalent_radii, atomic_numbers
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import IonicSystemConfig
from nnp_gen.core.exceptions import GenerationError

logger = logging.getLogger(__name__)

class IonicGenerator(BaseGenerator):
    def __init__(self, config: IonicSystemConfig):
        super().__init__(config)
        self.config = config

        # Check for optional dependencies
        try:
            import pymatgen.core as pmg
            from pymatgen.symmetry.structure import SymmetrizedStructure
            from pymatgen.core import Lattice, Species
            self.pmg = pmg
            self.Species = Species
            self.Lattice = Lattice
            self.has_pmg = True
        except ImportError:
            self.has_pmg = False
            self.pmg = None
            self.Lattice = None
            self.Species = None
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

    def _get_radius(self, species_str: str, charge: Optional[int] = None) -> float:
        """
        Get the radius of a species for lattice constant estimation.
        Prioritizes Shannon effective ionic radii if charge is provided/guessable,
        falls back to atomic radius.
        """
        if not self.has_pmg:
             # ASE fallback
             return covalent_radii[atomic_numbers[species_str]]

        try:
            # 1. Try Shannon radius if charge is known
            if charge is not None:
                try:
                    # Use Species class to get Shannon radius
                    sp = self.Species(species_str, oxidation_state=charge)
                    # Use coord number 6 ("VI") as a standard for rocksalt/general estimation
                    return sp.get_shannon_radius(cn="VI")
                except Exception:
                    pass

            el = self.pmg.Element(species_str)

            # 2. Try average ionic radius if no charge provided or lookup failed
            if el.average_ionic_radius:
                return el.average_ionic_radius

            # 3. Fallback: Use atomic radius
            # Note: atomic_radius might be None for some elements
            r = el.atomic_radius
            if r is not None:
                return r

            # 4. Ultimate fallback (e.g. for noble gases if atomic_radius is missing)
            return 1.5

        except Exception as e:
            logger.warning(f"Could not retrieve radius for {species_str}: {e}")
            return 1.5

    def _generate_with_pymatgen(self) -> List[Atoms]:
        """
        Use pymatgen to replace species in prototype structures.
        """
        generated = []
        elements = self.config.elements
        oxidation_states = self.config.oxidation_states

        # Get radii to estimate lattice constant
        # We compute radii based on individual oxidation states
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

        # Heuristics for lattice constants based on radii sum

        for t in types:
            try:
                struct = None
                if t == "rocksalt":
                    # For Rocksalt (face-centered cubic)
                    # a = 2 * radii_sum
                    a = 2.0 * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("Fm-3m", self.Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.5,0.5,0.5]])

                elif t == "cscl":
                    # CsCl (Body centered cubic-like).
                    # a = 2/sqrt(3) * radii_sum
                    a = (2.0 / np.sqrt(3.0)) * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("Pm-3m", self.Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.5,0.5,0.5]])

                elif t == "zincblende":
                    # Zincblende (Diamond-like)
                    # a = 4/sqrt(3) * radii_sum
                    a = (4.0 / np.sqrt(3.0)) * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("F-43m", self.Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.25,0.25,0.25]])

                elif t == "fluorite":
                    # Fluorite CaF2.
                    # a = 4/sqrt(3) * radii_sum
                    a = (4.0 / np.sqrt(3.0)) * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("Fm-3m", self.Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.25,0.25,0.25]])

                if struct:
                    atoms = self._pmg_to_ase(struct)
                    # Expand supercell
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
                 # Raise GenerationError to be consistent with requirements
                 raise GenerationError(f"Fallback generation failed for {elements}: {e}")

        return structures
