from typing import Literal, Optional
from pydantic import BaseModel, Field

class StructureMetadata(BaseModel):
    """
    Metadata for tracking structure provenance and status.
    """
    source: Literal["seed", "md", "relaxed"] = Field(..., description="Source process of the structure")
    config_hash: str = Field(..., description="Hash of the configuration used to generate this structure")
    parent_id: Optional[int] = Field(None, description="ID of the parent structure (e.g. initial structure for MD)")
    is_sampled: bool = Field(False, description="Whether this structure has been selected by sampling")
    sampling_method: Optional[str] = Field(None, description="Method used for sampling (e.g. FPS)")
    energy: Optional[float] = Field(None, description="Potential energy of the structure")
