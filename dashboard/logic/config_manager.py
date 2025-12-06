import yaml
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

# Define Pydantic Model for Configuration
class SimulationConfig(BaseModel):
    composition: str = Field(..., min_length=1, description="Chemical composition, e.g., FeNi")
    system_type: Literal["Alloy", "Covalent Crystal", "Ionic", "Molecule"] = "Alloy"
    temperature: float = Field(300.0, ge=0.0, le=5000.0, description="Temperature in Kelvin")
    atom_limit: int = Field(100, gt=0, le=10000, description="Maximum number of atoms")

class ConfigManager:
    """
    Handles loading and saving of configuration.
    """
    def __init__(self, config_path: str = "dashboard/config.yaml"):
        self.config_path = Path(config_path)

    def load_config(self) -> SimulationConfig:
        """
        Loads the configuration from YAML file and validates it.
        If file doesn't exist, returns default config.
        """
        if not self.config_path.exists():
            return SimulationConfig(composition="FeNi")

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            return SimulationConfig(**(data or {}))
        except (yaml.YAMLError, ValidationError) as e:
            # Fallback or re-raise?
            # For now, let's print error and return default to avoid crashing UI entirely,
            # but ideally we should propagate error.
            # Sticking to robustness: raise error so UI can show it, or return default?
            # User requirement: "validation logic (ensuring invalid configs raise errors)"
            # So I will let ValidationError propagate or handle it explicitly.
            raise e

    def save_config(self, config: SimulationConfig) -> None:
        """
        Saves the configuration model to YAML file.
        """
        with open(self.config_path, "w") as f:
            yaml.dump(config.model_dump(), f)
