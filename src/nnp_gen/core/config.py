from typing import Literal, Union, List, Dict, Optional
import numpy as np
from pydantic import BaseModel, Field, model_validator

# --- System Configuration ---

class PhysicsConstraints(BaseModel):
    max_atoms: int = Field(200, description="Hard limit for total number of atoms")
    min_density: float = Field(0.0, description="Minimum density (g/cm^3)")
    min_distance: float = Field(0.5, description="Minimum distance between atoms (Angstrom)")

class BaseSystemConfig(BaseModel):
    elements: List[str] = Field(..., min_length=1, description="List of elements in the system")
    constraints: PhysicsConstraints = Field(default_factory=PhysicsConstraints)
    pbc: List[bool] = Field([True, True, True], description="Periodic Boundary Conditions")

class IonicSystemConfig(BaseSystemConfig):
    type: Literal["ionic"] = "ionic"
    oxidation_states: Dict[str, int] = Field(..., description="Oxidation states for each element")
    charge_balance_tolerance: float = Field(0.0, description="Tolerance for charge balance")
    supercell_size: List[int] = Field([1, 1, 1], min_length=3, max_length=3, description="Supercell expansion factors")

    @model_validator(mode='after')
    def check_max_atoms(self):
        # Rough estimate: assumption of at least 1 atom per unit cell per element?
        # Or just checking expansion factor scaling?
        # The prompt says: "If unit_atoms * 5*5*5 > max_atoms".
        # We don't know unit_atoms exactly without structure.
        # But we can assume minimal valid cell has sum(stoichiometry) atoms.
        # If we don't know stoichiometry, assume 1 atom.
        # Let's assume minimum 1 atom per unit cell for safety if just checking supercell scaling vs max isn't enough.
        # However, 5*5*5 = 125. If max_atoms=200, it fits (125 < 200).
        # If user gives 10*10*10 = 1000, it fails.

        vol_factor = self.supercell_size[0] * self.supercell_size[1] * self.supercell_size[2]
        if vol_factor > self.constraints.max_atoms:
            # Even with 1 atom/cell, this exceeds max_atoms
            raise ValueError(f"Supercell expansion {self.supercell_size} results in at least {vol_factor} atoms, exceeding max_atoms={self.constraints.max_atoms}")
        return self

class AlloySystemConfig(BaseSystemConfig):
    type: Literal["alloy"] = "alloy"
    lattice_constant: Optional[float] = Field(None, description="Approximate lattice constant")
    spacegroup: Optional[int] = Field(None, description="Target spacegroup number (1-230)")
    supercell_size: List[int] = Field([1, 1, 1], min_length=3, max_length=3, description="Supercell expansion factors")

    @model_validator(mode='after')
    def check_max_atoms(self):
        vol_factor = self.supercell_size[0] * self.supercell_size[1] * self.supercell_size[2]
        if vol_factor > self.constraints.max_atoms:
            raise ValueError(f"Supercell expansion {self.supercell_size} results in at least {vol_factor} atoms, exceeding max_atoms={self.constraints.max_atoms}")
        return self

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
