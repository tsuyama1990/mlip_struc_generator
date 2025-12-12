
import logging
from typing import List, Optional
from ase import Atoms
from ase.build import bulk
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import DesignatedSystemConfig
from nnp_gen.core.physics import apply_vacancies

logger = logging.getLogger(__name__)

class DesignatedGenerator(BaseGenerator):
    def __init__(self, config: DesignatedSystemConfig, seed: Optional[int] = None):
        super().__init__(config, seed=seed)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates a specific designated structure.
        """
        seed_val = self.seed if self.seed is not None else 42
        # Use random state even if strict? Valid for vacancy randomness? Yes.
        # But for 'designated', maybe no randomness?
        # User said: "designated by space idenx etc ... : get user to key in space group and supercell size etc."
        # This implies standard crystal builder.
        
        # Determine crystal type
        sg = self.config.spacegroup
        struc_type = 'fcc'
        if sg == 229:
            struc_type = 'bcc'
        elif sg == 221:
            struc_type = 'sc'
        elif sg == 194:
            struc_type = 'hcp'
        
        # We need to construct based on `elements`. 
        # Designated implies we might want specific ordering?
        # If Elements = [Fe, Pt], and we use bulk(), how do we order?
        # If user wants ordered L1_0, they need correct basis.
        # Simple bulk() doesn't support complex basis easily with just args.
        # However, for the scope of "Designated by space index", user likely expects
        # a Unit Cell of that spacegroup with those elements.
        
        # If spacegroup is provided, we can try `ase.spacegroup.crystal`?
        # But we need Wyckoff positions.
        # The prompt implies simple high-level inputs "space group and supercell size".
        # So we probably stick to randomly filling a lattice of that SG?
        # That overlaps with AlloyGenerator but with fixed params.
        
        try:
            # Create primitive lattice
            # Use First Element?
            prim = bulk(self.config.elements[0], struc_type, a=self.config.lattice_constant)
        except Exception:
             # Fallback for non-cubic/common
             logger.warning(f"Simple bulk builder failed for SG {sg}, falling back to FCC/Cu template.")
             prim = bulk('Cu', 'fcc', a=self.config.lattice_constant)
        
        atoms = prim * self.config.supercell_size
        
        # Fill elements
        # For Designated, if they provide multiple elements e.g. [Fe, Pt], do they want ordered?
        # Without Wyckoff info, we can only do Random Alloy filling or ordered filling if we had a pattern.
        # Let's assume random filling of the correct lattice for now (Alloy-like but explicit params).
        # OR: "designated" might imply we just use the elements list cyclically?
        # e.g. [Fe, Pt] -> Atom 1 Fe, Atom 2 Pt...
        # That creates "some" order.
        
        rng = np.random.RandomState(seed_val)
        n_atoms = len(atoms)
        
        # Check if length of elements matches atoms?
        # If user provided exactly N elements in config `elements`, use them?
        # BaseSystemConfig elements is usually unique symbols.
        # But if DesignatedSystemConfig allows repeated? 
        # Actually `BaseSystemConfig` validates unique symbols in `validate_elements`?
        # Let's check logic:
        # `valid_symbols = set(chemical_symbols)` ... `for el in v: if el not in ...`
        # It does NOT enforce uniqueness. So user CAN provide [Fe, Fe, Pt, Pt].
        
        if len(self.config.elements) == n_atoms:
            # EXACT MAPPING
            atoms.set_chemical_symbols(self.config.elements)
        else:
            # Inferred Composition
            # Just fill randomly? Or cyclically?
            # Let's use cyclic for deterministic "designated" feel
            # unless it's just meant to be "Explicit Params Alloy".
            # "Designated" usually opposes "Random".
            # Let's cycle.
            symbols = []
            for i in range(n_atoms):
                symbols.append(self.config.elements[i % len(self.config.elements)])
            atoms.set_chemical_symbols(symbols)
            
        if self.config.vacancy_concentration > 0.0:
            atoms = apply_vacancies(atoms, self.config.vacancy_concentration, rng)
            
        return [atoms]
