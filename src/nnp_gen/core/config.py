from typing import Literal, Union, List, Dict, Optional, Tuple, Any
import numpy as np
import logging
from enum import Enum
from pydantic import BaseModel, Field, model_validator, field_validator

logger = logging.getLogger(__name__)

# --- Enums ---

class EnsembleType(str, Enum):
    AUTO = "AUTO"
    NVT = "NVT"
    NPT = "NPT"

class MCStrategy(str, Enum):
    SWAP = "SWAP"
    VACANCY_HOP = "VACANCY_HOP"

# --- Sub-Configurations ---

class MonteCarloConfig(BaseModel):
    enabled: bool = Field(False, description="Enable Monte Carlo moves")
    strategy: List[MCStrategy] = Field([MCStrategy.SWAP], description="List of MC strategies to use")
    swap_interval: int = Field(100, ge=1, description="Number of MD steps between MC moves")
    swap_pairs: Optional[List[Tuple[str, str]]] = Field(None, description="List of element pairs to swap")
    allow_charge_mismatch: bool = Field(False, description="Allow swapping ions with different charges")
    temp: Optional[float] = Field(None, gt=0, description="Temperature for MC acceptance (defaults to MD temp)")

    @field_validator('swap_pairs', mode='before')
    @classmethod
    def validate_swap_pairs(cls, v: Any) -> Optional[List[Tuple[str, str]]]:
        if v is None:
            return None
        # Handle List[List[str]] -> List[Tuple[str, str]]
        if isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, list):
                    if len(item) != 2:
                         raise ValueError(f"Swap pair must have exactly 2 elements: {item}")
                    new_list.append(tuple(item))
                elif isinstance(item, tuple):
                     if len(item) != 2:
                         raise ValueError(f"Swap pair must have exactly 2 elements: {item}")
                     new_list.append(item)
                else:
                    raise ValueError(f"Invalid swap pair format: {item}")
            return new_list
        return v

class PhysicsConstraints(BaseModel):
    max_atoms: int = Field(200, description="Hard limit for total number of atoms")
    min_density: float = Field(0.0, description="Minimum density (g/cm^3)")
    min_distance: float = Field(0.5, description="Minimum distance between atoms (Angstrom)")
    min_cell_length_factor: float = Field(1.0, description="Minimum cell length relative to r_cut")
    r_cut: float = Field(5.0, description="Cutoff radius of the potential model")

# --- System Configuration ---

class BaseSystemConfig(BaseModel):
    elements: List[str] = Field(..., min_length=1, description="List of elements in the system")
    constraints: PhysicsConstraints = Field(default_factory=PhysicsConstraints)
    pbc: List[bool] = Field([True, True, True], description="Periodic Boundary Conditions")
    rattle_std: float = Field(0.01, description="Standard deviation for Gaussian rattle in Angstrom")
    vol_scale_range: List[float] = Field([0.95, 1.05], min_length=2, max_length=2, description="Min/Max scaling factors for volume augmentation")
    strict_mode: bool = Field(True, description="Enforce strict dependency and physics checks")

    @field_validator('elements')
    @classmethod
    def validate_elements(cls, v: List[str]) -> List[str]:
        from ase.data import chemical_symbols
        valid_symbols = set(chemical_symbols)
        for el in v:
            if el not in valid_symbols:
                raise ValueError(f"Invalid element symbol: {el}")
        return v

    @field_validator('rattle_std')
    @classmethod
    def validate_rattle_std(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"rattle_std must be between 0.0 and 1.0 Angstrom, got {v}")
        if v > 0.5:
            logger.warning(f"High rattle_std ({v} A) detected. This may break bonds in fragile molecules.")
        return v

    @field_validator('vol_scale_range')
    @classmethod
    def validate_vol_scale_range(cls, v: List[float]) -> List[float]:
        if v[0] > v[1]:
            raise ValueError("vol_scale_range min must be less than or equal to max")
        if any(x <= 0 for x in v):
            raise ValueError("vol_scale_range values must be positive")
        return v

class IonicSystemConfig(BaseSystemConfig):
    type: Literal["ionic"] = "ionic"
    oxidation_states: Dict[str, int] = Field(..., description="Oxidation states for each element")
    charge_balance_tolerance: float = Field(0.0, description="Tolerance for charge balance")
    supercell_size: List[int] = Field([1, 1, 1], min_length=3, max_length=3, description="Supercell expansion factors")
    default_magmoms: Optional[Dict[str, float]] = Field(None, description="Initial magnetic moments per element")
    vacancy_concentration: float = Field(0.0, description="Fraction of atoms to remove as vacancies")

    @field_validator('vacancy_concentration')
    @classmethod
    def validate_vacancy_concentration(cls, v: float) -> float:
        if not (0.0 <= v <= 0.25):
            raise ValueError("vacancy_concentration must be between 0.0 and 0.25")
        return v

class AlloySystemConfig(BaseSystemConfig):
    type: Literal["alloy"] = "alloy"
    lattice_constant: Optional[float] = Field(None, description="Approximate lattice constant")
    spacegroup: Optional[int] = Field(None, description="Target spacegroup number (1-230)")
    supercell_size: List[int] = Field([1, 1, 1], min_length=3, max_length=3, description="Supercell expansion factors")
    default_magmoms: Optional[Dict[str, float]] = Field(None, description="Initial magnetic moments per element")
    vacancy_concentration: float = Field(0.0, description="Fraction of atoms to remove as vacancies")

    @field_validator('vacancy_concentration')
    @classmethod
    def validate_vacancy_concentration(cls, v: float) -> float:
        if not (0.0 <= v <= 0.25):
            raise ValueError("vacancy_concentration must be between 0.0 and 0.25")
        return v

class CovalentSystemConfig(BaseSystemConfig):
    type: Literal["covalent"] = "covalent"
    dimensionality: Literal[0, 1, 2, 3] = Field(3, description="Dimensionality of the system")
    min_volume: Optional[float] = Field(None, description="Minimum volume per atom")
    vacancy_concentration: float = Field(0.0, description="Fraction of atoms to remove as vacancies")

    @field_validator('vacancy_concentration')
    @classmethod
    def validate_vacancy_concentration(cls, v: float) -> float:
        if not (0.0 <= v <= 0.25):
            raise ValueError("vacancy_concentration must be between 0.0 and 0.25")
        return v

class MoleculeSystemConfig(BaseSystemConfig):
    type: Literal["molecule"] = "molecule"
    smiles: str = Field(..., description="SMILES string of the molecule")
    num_conformers: int = Field(10, description="Number of conformers to generate")
    # Molecules usually don't have supercell expansion in the same way, or pbc is False
    pbc: List[bool] = Field([False, False, False], description="Periodic Boundary Conditions for Molecules")

class InterfaceMode(str, Enum):
    HETERO_CRYSTAL = "hetero_crystal"  # Solid on Solid (e.g., Cu on Au)
    SOLID_LIQUID = "solid_liquid"      # Liquid on Solid (e.g., Water on TiO2)

class InterfaceSystemConfig(BaseSystemConfig):
    type: Literal["interface"] = "interface"
    mode: InterfaceMode

    # RECURSIVE CONFIGURATION:
    # We re-use the specific configs for the two phases.
    phase_a: Dict[str, Any] = Field(..., description="Config dict for the substrate (e.g. Alloy)")
    phase_b: Dict[str, Any] = Field(..., description="Config dict for the film/liquid")

    # Interface Physics
    vacuum: float = Field(15.0, description="Vacuum padding in Angstrom")
    interface_distance: float = Field(2.5, description="Initial distance between phases")
    max_mismatch: float = Field(0.05, description="Max allowed lattice mismatch (5%)")

    # For Solid-Liquid
    solvent_density: float = Field(1.0, description="Target liquid density in g/cm^3")

    elements: List[str] = Field(default=[], description="List of elements in the system")

    @model_validator(mode='before')
    @classmethod
    def extract_elements(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If elements are not explicitly provided, try to extract from phases
            if 'elements' not in data or not data['elements']:
                elems = set()
                for phase_key in ['phase_a', 'phase_b']:
                    if phase_key in data and isinstance(data[phase_key], dict):
                        phase_elements = data[phase_key].get('elements', [])
                        if phase_elements:
                            elems.update(phase_elements)

                # If we found elements, set them
                if elems:
                    data['elements'] = list(elems)
        return data

class AdsorbateMode(str, Enum):
    ATOM = "atom"
    MOLECULE = "molecule"
    SMILES = "smiles"
    FILE = "file"

class AdsorbateConfig(BaseModel):
    source: str = Field(..., description="Element symbol, Formula, SMILES, or Filepath")
    mode: AdsorbateMode = Field(..., description="Interpretation of the source string")
    count: int = Field(1, ge=1, description="Number of adsorbates to place")
    height: float = Field(2.0, description="Height above the surface")

    @field_validator('count')
    @classmethod
    def validate_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Count must be at least 1")
        return v

class VacuumAdsorbateSystemConfig(BaseSystemConfig):
    type: Literal["vacuum_adsorbate"] = "vacuum_adsorbate"
    substrate: Dict[str, Any] = Field(..., description="Configuration for the bulk substrate")
    miller_indices: List[Tuple[int, int, int]] = Field(..., description="List of Miller indices to cleave")
    layers: int = Field(4, description="Number of atomic layers in the slab")
    vacuum: float = Field(10.0, description="Vacuum size on top of the surface")
    defect_rate: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of top-layer atoms to remove")
    adsorbates: List[AdsorbateConfig] = Field(default_factory=list, description="List of adsorbates to place")

    @field_validator('defect_rate')
    @classmethod
    def validate_defect_rate(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
             raise ValueError("defect_rate must be between 0.0 and 1.0")
        return v

    @model_validator(mode='before')
    @classmethod
    def extract_elements(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'elements' not in data or not data['elements']:
                elems = set()
                # Substrate elements
                if 'substrate' in data and isinstance(data['substrate'], dict):
                     elems.update(data['substrate'].get('elements', []))
                if elems:
                    data['elements'] = list(elems)
        return data

class SolventAdsorbateSystemConfig(VacuumAdsorbateSystemConfig):
    type: Literal["solvent_adsorbate"] = "solvent_adsorbate"
    solvent_density: float = Field(1.0, description="Target solvent density in g/cm^3")
    solvent_smiles: str = Field("O", description="SMILES string for the solvent (default Water)")

class KnowledgeSystemConfig(BaseSystemConfig):
    type: Literal["knowledge"] = "knowledge"
    formula: str = Field(..., description="Chemical formula (e.g., 'LiFe0.5Co0.5O2')")
    use_cod: bool = Field(True, description="Attempt to query COD for exact matches")
    use_materials_project: bool = Field(False, description="Attempt to query Materials Project (requires API key)")
    mp_api_key: Optional[str] = Field(None, description="Materials Project API Key")
    use_prototypes: bool = Field(True, description="Attempt to use anonymous prototypes if exact match fails")
    use_symmetry_generation: bool = Field(True, description="Fallback to random symmetry generation (Pyxtal)")
    max_supercell_atoms: int = Field(200, description="Max atoms in supercell for disordered structures")

    @field_validator('formula')
    @classmethod
    def validate_formula(cls, v: str) -> str:
        from pymatgen.core import Composition
        try:
            Composition(v)
        except Exception:
            raise ValueError(f"Invalid chemical formula: {v}")
        return v

    @model_validator(mode='before')
    @classmethod
    def extract_elements(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'elements' not in data or not data['elements']:
                if 'formula' in data:
                    from pymatgen.core import Composition
                    try:
                        comp = Composition(data['formula'])
                        data['elements'] = [str(el) for el in comp.elements]
                    except Exception:
                        pass # Let field validator handle it
        return data

class UserFileSystemConfig(BaseSystemConfig):
    type: Literal["user_file"] = "user_file"
    path: str = Field(..., description="Path to the structure file")
    format: Optional[str] = Field(None, description="File format (e.g., 'cif', 'xyz'). If None, inferred from extension.")
    repeat: int = Field(1, description="Number of times to duplicate the structures")
    vacancy_concentration: float = Field(0.0, description="Fraction of atoms to remove as vacancies")

    @field_validator('repeat')
    @classmethod
    def validate_repeat(cls, v: int) -> int:
        if v < 1:
            raise ValueError("repeat must be at least 1")
        return v

    @field_validator('vacancy_concentration')
    @classmethod
    def validate_vacancy_concentration(cls, v: float) -> float:
        if not (0.0 <= v <= 0.25):
            raise ValueError("vacancy_concentration must be between 0.0 and 0.25")
        return v

# Discriminated Union for System Config
SystemConfig = Union[
    IonicSystemConfig,
    AlloySystemConfig,
    CovalentSystemConfig,
    MoleculeSystemConfig,
    InterfaceSystemConfig,
    VacuumAdsorbateSystemConfig,
    SolventAdsorbateSystemConfig,
    UserFileSystemConfig,
    KnowledgeSystemConfig
]

# --- Exploration Configuration ---

class ExplorationConfig(BaseModel):
    method: Literal["md", "mc", "hybrid_mc_md", "melt_quench", "normal_mode"] = Field("md", description="Exploration method")
    model_name: Literal["mace", "sevenn", "emt"] = Field("mace", description="Calculator model name")
    
    # Temperature Settings
    temperature_mode: Literal["constant", "gradient"] = Field("constant", description="Temperature control mode")
    temperature: float = Field(300.0, description="Temperature in Kelvin (Constant mode)")
    temp_start: Optional[float] = Field(None, description="Start temperature for gradient mode")
    temp_end: Optional[float] = Field(None, description="End temperature for gradient mode")
    
    pressure: Optional[float] = Field(None, description="Pressure in GPa (None for NVT)")
    ttime: float = Field(100.0, description="Thermostat time constant for NPT in fs")
    steps: int = Field(1000, description="Number of steps per exploration run")
    timestep: float = Field(1.0, description="Timestep in fs")

    ensemble: EnsembleType = Field(EnsembleType.AUTO, description="MD Ensemble (AUTO, NVT, NPT)")
    mc_config: Optional[MonteCarloConfig] = Field(default_factory=lambda: MonteCarloConfig(enabled=False), description="Monte Carlo Configuration")

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Temperature must be positive")
        if v >= 10000:
            raise ValueError("Temperature must be less than 10000 K")
        return v
    
    @model_validator(mode='after')
    def validate_temperature_settings(self) -> 'ExplorationConfig':
        if self.temperature_mode == "gradient":
            if self.temp_start is None or self.temp_end is None:
                raise ValueError("temp_start and temp_end must be provided for gradient mode")
            if self.temp_start <= 0 or self.temp_end <= 0:
                raise ValueError("Temperatures must be positive")
        return self

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Steps must be positive")
        return v

    @field_validator('timestep')
    @classmethod
    def validate_timestep(cls, v: float) -> float:
        if v <= 0.1:
            raise ValueError("Timestep must be greater than 0.1 fs")
        return v

# --- Sampling Configuration ---

class SamplingConfig(BaseModel):
    strategy: Literal["fps", "random", "manual"] = Field("fps", description="Sampling strategy")
    n_samples: int = Field(100, description="Number of structures to sample")
    descriptor_type: Literal["soap", "ace"] = Field("soap", description="Descriptor for FPS")
    min_distance: float = Field(1.5, description="Minimum atomic distance for pre-filtering")

# --- App Configuration ---

class AppConfig(BaseModel):
    system: SystemConfig = Field(..., discriminator="type")
    exploration: ExplorationConfig
    sampling: SamplingConfig
    output_dir: str = Field("output", description="Directory to save results")
    seed: int = Field(42, description="Random seed")
