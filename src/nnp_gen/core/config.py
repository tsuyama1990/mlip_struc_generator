from typing import Literal, Union, List, Dict, Optional
from pydantic import BaseModel, Field

# --- System Configuration ---

class BaseSystemConfig(BaseModel):
    elements: List[str] = Field(..., min_length=1, description="List of elements in the system")

class IonicSystemConfig(BaseSystemConfig):
    type: Literal["ionic"] = "ionic"
    oxidation_states: Dict[str, int] = Field(..., description="Oxidation states for each element")
    charge_balance_tolerance: float = Field(0.0, description="Tolerance for charge balance")

class AlloySystemConfig(BaseSystemConfig):
    type: Literal["alloy"] = "alloy"
    lattice_constant: Optional[float] = Field(None, description="Approximate lattice constant")
    spacegroup: Optional[int] = Field(None, description="Target spacegroup number (1-230)")

class CovalentSystemConfig(BaseSystemConfig):
    type: Literal["covalent"] = "covalent"
    dimensionality: Literal[0, 1, 2, 3] = Field(3, description="Dimensionality of the system")
    min_volume: Optional[float] = Field(None, description="Minimum volume per atom")

class MoleculeSystemConfig(BaseSystemConfig):
    type: Literal["molecule"] = "molecule"
    smiles: str = Field(..., description="SMILES string of the molecule")
    num_conformers: int = Field(10, description="Number of conformers to generate")

# Discriminated Union for System Config
SystemConfig = Union[
    IonicSystemConfig,
    AlloySystemConfig,
    CovalentSystemConfig,
    MoleculeSystemConfig
]

# --- Exploration Configuration ---

class ExplorationConfig(BaseModel):
    method: Literal["md", "mc", "hybrid_mc_md", "melt_quench", "normal_mode"] = Field("md", description="Exploration method")
    temperature: float = Field(300.0, description="Temperature in Kelvin")
    pressure: Optional[float] = Field(None, description="Pressure in GPa (None for NVT)")
    steps: int = Field(1000, description="Number of steps per exploration run")
    timestep: float = Field(1.0, description="Timestep in fs")

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
