import logging
from typing import List, Optional, Dict, Any, Tuple
import random
import warnings
from ase import Atoms
from ase.data import atomic_numbers
from ase.units import Bohr
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Composition, Structure, Element
from pymatgen.ext.cod import COD
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

from nnp_gen.core.config import KnowledgeSystemConfig
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.exceptions import GenerationError

logger = logging.getLogger(__name__)

class KnowledgeBasedGenerator(BaseGenerator):
    """
    Generator that creates structures using "Knowledge-Based" strategies:
    1. Query COD for exact formula matches (handling disorder).
    2. Query COD for anonymous prototypes and dope them ("Smart Doping").
    3. Fallback to random symmetry generation (Pyxtal).
    """

    def __init__(self, config: KnowledgeSystemConfig):
        super().__init__(config)
        self.config: KnowledgeSystemConfig = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Orchestrates the cascading generation logic.
        """
        structures = []

        # Step 1: Exact Match (COD)
        if self.config.use_cod:
            logger.info(f"Attempting COD query for {self.config.formula}...")
            try:
                structures = self._query_cod(self.config.formula)
                if structures:
                    logger.info(f"Found {len(structures)} structures via exact COD match.")
                    return structures
            except Exception as e:
                logger.warning(f"COD Exact Match failed: {e}")

        # Step 2: Prototype Substitution (Smart Doping)
        if self.config.use_prototypes:
            logger.info(f"Attempting Prototype Substitution for {self.config.formula}...")
            try:
                structures = self._generate_from_prototype(self.config.formula)
                if structures:
                    logger.info(f"Found {len(structures)} structures via Prototype Substitution.")
                    return structures
            except Exception as e:
                logger.warning(f"Prototype Substitution failed: {e}")

        # Step 3: Symmetry Generation (Pyxtal Fallback)
        if self.config.use_symmetry_generation:
            logger.info("Falling back to Pyxtal symmetry generation...")
            try:
                structures = self._generate_with_pyxtal(self.config.formula)
                if structures:
                    logger.info(f"Generated {len(structures)} structures via Pyxtal.")
                    return structures
            except Exception as e:
                logger.warning(f"Pyxtal generation failed: {e}")

        if not structures:
            raise GenerationError(f"All knowledge-based strategies failed for {self.config.formula}")

        return structures

    def _query_cod(self, formula: str) -> List[Atoms]:
        """
        Queries COD for structures matching the formula.
        Handles disordered structures by ordering them.
        """
        try:
            cod = COD()
            structures: List[Structure] = cod.get_structure_by_formula(formula)
        except Exception as e:
            logger.warning(f"COD API call failed: {e}")
            return []

        valid_atoms = []
        for i, struct in enumerate(structures):
            try:
                if struct.is_ordered:
                    atoms = AseAtomsAdaptor.get_atoms(struct)
                    valid_atoms.append(atoms)
                else:
                    logger.info(f"Structure {i} from COD is disordered. Attempting to order...")
                    ordered = self._order_disordered_structure(struct)
                    if ordered:
                        valid_atoms.append(ordered)
            except Exception as e:
                logger.warning(f"Failed to process COD structure {i}: {e}")

        return valid_atoms

    def _order_disordered_structure(self, struct: Structure) -> Optional[Atoms]:
        """
        Orders a disordered structure using OrderDisorderedStructureTransformation.
        """
        try:
            # Enforce max atoms limit immediately to prevent explosions
            # However, we don't know the size until we transform.
            # We can start with a primitive cell if possible.
            try:
                struct = struct.get_primitive_structure()
            except Exception:
                pass # Already primitive or failed

            trans = OrderDisorderedStructureTransformation(
                no_oxi_states=True,
                symmetrize=False # Keep it simple
            )

            # transformations return a list of dicts [{'structure': ...}, ...]
            ordered_list = trans.apply_transformation(struct, return_ranked_list=1)

            if not ordered_list:
                return None

            best_struct = ordered_list[0]['structure']

            if len(best_struct) > self.config.max_supercell_atoms:
                logger.warning(f"Ordered structure too large ({len(best_struct)} atoms). Skipping.")
                return None

            return AseAtomsAdaptor.get_atoms(best_struct)

        except Exception as e:
            logger.warning(f"Ordering logic failed: {e}")
            return None

    def _generate_from_prototype(self, formula: str) -> List[Atoms]:
        """
        Implements 'Smart Doping':
        1. Identify anonymous formula (e.g., A B0.5 C0.5 O2 -> ABC2)
        2. Query COD for ordered prototypes matching the anonymous formula.
        3. Substitute atoms based on electronegativity matching.
        4. Apply Vegard's law for volume scaling.
        """
        target_comp = Composition(formula)

        end_member_formula = self._guess_end_member(target_comp)
        logger.info(f"Guessed end-member prototype formula: {end_member_formula}")

        try:
            cod = COD()
            prototypes = cod.get_structure_by_formula(end_member_formula)
        except Exception as e:
             logger.warning(f"COD query for prototype {end_member_formula} failed: {e}")
             return []

        ordered_prototypes = [p for p in prototypes if p.is_ordered]

        # If no ordered ones, try to order them first
        if not ordered_prototypes:
             for p in prototypes:
                 ordered_atoms = self._order_disordered_structure(p)
                 if ordered_atoms:
                     # Adapt back to Structure for processing
                     ordered_prototypes.append(AseAtomsAdaptor.get_structure(ordered_atoms))

        if not ordered_prototypes:
            logger.warning(f"No ordered prototype found for {end_member_formula}")
            return []

        # Pick the smallest/simplest prototype to start with?
        # Sort by number of atoms to be efficient
        ordered_prototypes.sort(key=lambda x: len(x))
        prototype = ordered_prototypes[0]

        # Now apply the substitution
        final_structure = self._apply_doping(target_comp, prototype)
        return [final_structure]

    def _guess_end_member(self, comp: Composition) -> str:
        """
        Constructs a likely end-member formula.
        E.g. LiFe0.5Co0.5O2 -> LiCoO2 (picking one dopant)
        """
        import collections

        # Group by stoichiometry logic similar to _group_by_stoichiometry
        groups = self._group_by_stoichiometry(comp)

        final_composition = {}

        for group in groups:
            # group is {'amount': ..., 'avg_en': ..., 'elements': {El: amt}}

            # Pick representative element for this site/group
            # We pick the one with Highest EN? Or random?
            # Let's pick Highest EN (most electronegative) as it's often the anion or stable cation.
            # But wait, for (Fe, Co), they are metals.
            # If we pick Oxygen, it's fine.

            # Architect said: "Sort groups ... secondarily by Average Electronegativity".
            # This implies EN is a good discriminator.

            els = list(group['elements'].keys())
            # Sort by EN descending (highest first)
            els.sort(key=lambda x: x.X, reverse=True)
            representative = els[0]

            final_composition[representative] = group['amount']

        return Composition.from_dict(final_composition).formula

    def _apply_doping(self, target_comp: Composition, prototype: Structure) -> Atoms:
        """
        Maps Target Composition to Prototype Sites and performs substitution.
        """
        # 1. Analyze Target Groups
        target_groups = self._group_by_stoichiometry(target_comp)

        # 2. Analyze Prototype Groups
        proto_comp = prototype.composition
        proto_groups = self._group_by_stoichiometry(proto_comp)

        # 3. Sort Groups (Mole Fraction Descending, then EN Ascending)
        def sort_key(group):
            return (-group['amount'], group['avg_en'])

        target_groups.sort(key=sort_key)
        proto_groups.sort(key=sort_key)

        if len(target_groups) != len(proto_groups):
            # Fallback: Just try to map by index up to min length?
            # Or fail?
            logger.warning(f"Mismatch in site groups: Target {len(target_groups)} vs Proto {len(proto_groups)}. Attempting best effort map.")

        # 4. Map and Replace
        species_map = {}
        vol_proto = 0.0
        vol_target = 0.0

        # Only iterate up to the matching count
        limit = min(len(target_groups), len(proto_groups))

        for i in range(limit):
            t_group = target_groups[i]
            p_group = proto_groups[i]

            # Map every element in p_group to the distribution in t_group
            # Assume prototype site is occupied by the single representative element (or all elements in that group)
            # Since we selected the prototype based on end-member logic, p_group should likely be single element.

            # But if prototype has multiple elements in a group (e.g. it was complex), we map ALL of them.

            p_elements = list(p_group['elements'].keys())

            # Target distribution
            total_t = t_group['amount']
            t_dist = {k: v/total_t for k,v in t_group['elements'].items()}

            for p_el in p_elements:
                species_map[p_el] = t_dist

                # Vegard's: Proto Contribution
                r_p = getattr(p_el, 'atomic_radius', 1.5) or 1.5
                vol_proto += p_group['elements'][p_el] * (r_p**3)

            # Vegard's: Target Contribution
            # The target volume for this group is proportional to total_t
            # But we are replacing p_group sites.
            # So we scale by p_group total amount?
            # Yes, if we replace 1 atom of Co with (0.5 Fe, 0.5 Co), we preserve site count.

            p_total = p_group['amount']

            for t_el, t_frac in t_dist.items():
                r_t = getattr(t_el, 'atomic_radius', 1.5) or 1.5
                # Weighted by the amount of sites we are replacing
                vol_target += (p_total * t_frac) * (r_t**3)

        # 5. Apply Substitution
        prototype.replace_species(species_map)

        # 6. Apply Vegard's Scaling
        if vol_proto > 0:
            scale_factor = (vol_target / vol_proto) ** (1/3)
            if not (0.8 < scale_factor < 1.2):
                logger.warning(f"Large Vegard scaling factor {scale_factor:.2f}. Clamping to [0.8, 1.2].")
                scale_factor = max(0.8, min(scale_factor, 1.2))

            prototype.scale_lattice(prototype.volume * (scale_factor**3))

        # 7. Convert to Ordered Supercell
        ordered = self._order_disordered_structure(prototype)
        if not ordered:
             raise GenerationError("Failed to order the doped structure.")

        return ordered

    def _group_by_stoichiometry(self, comp: Composition) -> List[Dict]:
        """
        Groups elements by stoichiometry to identify sites.
        Returns list of dicts: {'amount': float, 'avg_en': float, 'elements': {Element: amt}}
        """
        import collections
        groups = []
        els_dict = {el: comp.get_el_amt_dict()[el.symbol] for el in comp.elements}

        # Sort by electronegativity for deterministic processing
        els = sorted(comp.elements, key=lambda x: x.X)
        used = set()

        for el in els:
            if el in used: continue

            amt = els_dict[el]
            current_group = {el: amt}
            current_sum = amt
            used.add(el)

            # Try to complete integer sums
            if abs(current_sum - round(current_sum)) > 0.01:
                for peer in els:
                    if peer in used: continue
                    peer_amt = els_dict[peer]
                    new_sum = current_sum + peer_amt

                    if abs(new_sum - round(new_sum)) < 0.01:
                         current_group[peer] = peer_amt
                         used.add(peer)
                         current_sum = new_sum
                         break
                    elif abs(current_sum - 0.5) < 0.01 and abs(peer_amt - 0.5) < 0.01:
                         current_group[peer] = peer_amt
                         used.add(peer)
                         current_sum = new_sum
                         break

            avg_en = sum(e.X * a for e,a in current_group.items()) / current_sum

            groups.append({
                'amount': current_sum,
                'avg_en': avg_en,
                'elements': current_group
            })

        return groups

    def _generate_with_pyxtal(self, formula: str) -> List[Atoms]:
        """
        Uses Pyxtal to generate random symmetric structures.
        """
        try:
            from pyxtal import pyxtal
        except ImportError:
            logger.warning("Pyxtal not installed. Skipping symmetry generation.")
            return []

        comp = Composition(formula)
        # Get integer formula for Pyxtal
        int_comp, scale = comp.get_integer_formula_and_factor()
        species = [str(el) for el in int_comp.elements]
        numIons = [int(int_comp.get_el_amt_dict()[el.symbol]) for el in int_comp.elements]

        structures = []
        for _ in range(5):
            try:
                sg = random.randint(2, 230)
                crystal = pyxtal()
                crystal.from_random(3, sg, species, numIons)
                if crystal.valid:
                    structures.append(crystal.to_ase())
            except Exception:
                continue

        return structures
