import logging
import numpy as np
from typing import List, Optional
from ase import Atoms
from ase.build import bulk
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
            self.pmg = pmg
            self.has_pmg = True
        except ImportError:
            self.has_pmg = False
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

    def _generate_with_pymatgen(self) -> List[Atoms]:
        """
        Use pymatgen to replace species in prototype structures.
        """
        generated = []

        # Define some common binary prototypes (spacegroup number, symbol)
        # 225: Rocksalt (Fm-3m)
        # 221: CsCl (Pm-3m)
        # 216: Zincblende (F-43m)
        # 225: Fluorite (Fm-3m) - AB2

        # We can create dummy structures and replace species
        # But we need appropriate lattice constants.
        # Pymatgen's Structure.from_spacegroup requires lattice parameters.
        # We can estimate bond lengths from ionic radii.

        elements = self.config.elements
        oxidation_states = self.config.oxidation_states

        # Get ionic radii to estimate lattice constant
        # This is a bit complex without a database query.
        # As a heuristic, we can use sum of atomic radii or look up tables if we had them.
        # pymatgen Element class has atomic_radius.

        radii_sum = 0.0
        for el_str in elements:
            el = self.pmg.Element(el_str)
            # Use atomic radius as approximation if ionic not available easily without knowing coord number
            r = el.atomic_radius
            if r is None:
                r = 1.5 # fallback
            radii_sum += r

        # Simplified logic: Generate a few common structure types if stoichiometry matches

        # Determine likely stoichiometry from oxidation states
        # This is strictly combinatorics if we don't know the compound.
        # But the user provides a list of elements.
        # If 2 elements, we assume binary.

        if len(elements) == 2:
            el1, el2 = elements
            q1 = oxidation_states[el1]
            q2 = oxidation_states[el2]

            # Simple charge balance check: n1*q1 + n2*q2 = 0
            # Rocksalt: 1:1 -> q1 = -q2
            # Fluorite: 1:2 -> q1 = -2*q2 or 2*q1 = -q2

            # 1. Rocksalt / CsCl / Zincblende (1:1)
            if abs(q1) == abs(q2):
                generated.extend(self._create_binary_prototypes(el1, el2, radii_sum, ["rocksalt", "cscl", "zincblende"]))

            # 2. Fluorite / Antifluorite (1:2 or 2:1)
            elif abs(q1) == 2 * abs(q2):
                # el1 is +4 or +2, el2 is -2 or -1. Formula AB2
                generated.extend(self._create_binary_prototypes(el1, el2, radii_sum, ["fluorite"]))
            elif 2 * abs(q1) == abs(q2):
                # Formula A2B
                generated.extend(self._create_binary_prototypes(el2, el1, radii_sum, ["fluorite"]))

        # If no specific logic matched or just as a catch-all, we might try generic substitution
        # but that's risky without valid physics.

        return generated

    def _create_binary_prototypes(self, species_a: str, species_b: str, radii_sum: float, types: List[str]) -> List[Atoms]:
        structures = []

        # Estimate lattice parameter 'a'
        # Rocksalt: a = 2 * (r_cation + r_anion) roughly?
        # For NaCl: rNa=1.02, rCl=1.81 (ionic). Sum=2.83. a_exp=5.64. So 2*sum is good estimate.
        # CsCl: 2*r = sqrt(3)/2 * a  => a = 4/sqrt(3) * r ~ 2.3 * r
        # Zincblende: 2*r = sqrt(3)/4 * a => a = 8/sqrt(3) * r ~ 4.6 * r

        # We'll use a rough heuristic and then maybe let MD/optimization fix it later?
        # The prompt asks to remove hardcoded a=5.0 and use pymatgen/radii.

        for t in types:
            try:
                struct = None
                if t == "rocksalt":
                    a = 2.0 * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.5,0.5,0.5]])
                elif t == "cscl":
                    # CsCl (B2): 8-coord. d = a * sqrt(3)/2 => a = 2/sqrt(3) * d ~ 1.15 * d
                    a = (2.0 / np.sqrt(3.0)) * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("Pm-3m", Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.5,0.5,0.5]])
                elif t == "zincblende":
                    # Zincblende (B3): 4-coord. d = a * sqrt(3)/4 => a = 4/sqrt(3) * d ~ 2.31 * d
                    a = (4.0 / np.sqrt(3.0)) * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("F-43m", Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.25,0.25,0.25]])
                elif t == "fluorite":
                    # CaF2: Ca at 0,0,0; F at 0.25,0.25,0.25 and 0.75,0.75,0.75
                    # Bond length d = sqrt(3)/4 * a. d ~ sum_radii.
                    a = 4.0/np.sqrt(3.0) * radii_sum
                    struct = self.pmg.Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [species_a, species_b], [[0,0,0], [0.25,0.25,0.25]])

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
        # We can use pymatgen.io.ase.AseAtomsAdaptor but avoiding extra imports if possible
        # Or just manual conversion
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

        # Only support binary rocksalt as fallback for now (improving on original code slightly by guessing lattice)
        if len(elements) == 2:
            cation, anion = elements

            # Rough estimate of lattice constant without radii data (hard to do accurately without data)
            # We will use a slightly better heuristic or default if we must.
            # But really we should just fail if accuracy is required.
            # However, I'll keep the logic but maybe just warn.

            # If we don't have radii, we can't guess 'a' well.
            # The user explicitly asked to remove hardcoded a=5.0.
            # But without pymatgen, we don't have easy access to radii.
            # I will check if ASE has data.
            from ase.data import covalent_radii, atomic_numbers

            r1 = covalent_radii[atomic_numbers[cation]]
            r2 = covalent_radii[atomic_numbers[anion]]
            # Ionic radii are usually different but covalent sum is a better guess than 5.0
            # Rocksalt: a = 2 * d. d ~ r1+r2 (very roughly)
            a = 2.0 * (r1 + r2)

            try:
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

        return structures

# Need to import Lattice for _create_binary_prototypes if using pymatgen
try:
    from pymatgen.core import Lattice
except ImportError:
    pass
