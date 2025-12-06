from typing import Literal, Union, List, Dict, Optional, Tuple, Any
import numpy as np
from enum import Enum
from pydantic import BaseModel, Field, model_validator, field_validator

# --- System Configuration ---

class PhysicsConstraints(BaseModel):
    max_atoms: int = Field(200, description="Hard limit for total number of atoms")
    min_density: float = Field(0.0, description="Minimum density (g/cm^3)")
    min_distance: float = Field(0.5, description="Minimum distance between atoms (Angstrom)")
    min_cell_length_factor: float = Field(1.0, description="Minimum cell length relative to r_cut")
    r_cut: float = Field(5.0, description="Cutoff radius of the potential model")

class BaseSystemConfig(BaseModel):
    elements: List[str] = Field(..., min_length=1, description="List of elements in the system")
    constraints: PhysicsConstraints = Field(default_factory=PhysicsConstraints)
    pbc: List[bool] = Field([True, True, True], description="Periodic Boundary Conditions")
    rattle_std: float = Field(0.01, description="Standard deviation for Gaussian rattle in Angstrom")
    vol_scale_range: List[float] = Field([0.95, 1.05], min_length=2, max_length=2, description="Min/Max scaling factors for volume augmentation")

    @field_validator('rattle_std')
    @classmethod
    def validate_rattle_std(cls, v: float) -> float:
        if not (0.0 <= v <= 0.5):
            raise ValueError(f"rattle_std must be between 0.0 and 0.5 Angstrom, got {v}")
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

class AlloySystemConfig(BaseSystemConfig):
    type: Literal["alloy"] = "alloy"
    lattice_constant: Optional[float] = Field(None, description="Approximate lattice constant")
    spacegroup: Optional[int] = Field(None, description="Target spacegroup number (1-230)")
    supercell_size: List[int] = Field([1, 1, 1], min_length=3, max_length=3, description="Supercell expansion factors")
    default_magmoms: Optional[Dict[str, float]] = Field(None, description="Initial magnetic moments per element")

class CovalentSystemConfig(BaseSystemConfig):
    type: Literal["covalent"] = "covalent"
    dimensionality: Literal[0, 1, 2, 3] = Field(3, description="Dimensionality of the system")
    min_volume: Optional[float] = Field(None, description="Minimum volume per atom")

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

# Discriminated Union for System Config
SystemConfig = Union[
    IonicSystemConfig,
    AlloySystemConfig,
    CovalentSystemConfig,
    MoleculeSystemConfig,
    InterfaceSystemConfig
]

# --- Exploration Configuration ---

class ExplorationConfig(BaseModel):
    method: Literal["md", "mc", "hybrid_mc_md", "melt_quench", "normal_mode"] = Field("md", description="Exploration method")
    model_name: Literal["mace", "sevenn", "emt"] = Field("mace", description="Calculator model name")
    temperature: float = Field(300.0, description="Temperature in Kelvin")
    pressure: Optional[float] = Field(None, description="Pressure in GPa (None for NVT)")
    steps: int = Field(1000, description="Number of steps per exploration run")
    timestep: float = Field(1.0, description="Timestep in fs")

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Temperature must be positive")
        return v

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Steps must be positive")
        return v

    @field_validator('timestep')
    @classmethod
    def validate_timestep(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Timestep must be positive")
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
