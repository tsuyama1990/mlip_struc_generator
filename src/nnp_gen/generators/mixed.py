
import logging
from typing import List, Optional
from ase import Atoms
from pydantic import TypeAdapter

from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import MixedSystemConfig, SystemConfig
from nnp_gen.core.exceptions import GenerationError

logger = logging.getLogger(__name__)

class MixedGenerator(BaseGenerator):
    def __init__(self, config: MixedSystemConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Iterates over sub-system configurations, instantiates their generators,
        and aggregates the results.
        """
        logger.info(f"Generating mixed structures from {len(self.config.systems)} sub-systems.")
        
        all_structures = []
        
        # Avoid circular imports usually, but we need Factory here.
        # Import inside method or at top? Top might cycle.
        # But Factory imports us probably. So inside method is safer.
        from nnp_gen.generators.factory import GeneratorFactory
        
        for i, sub_conf_dict in enumerate(self.config.systems):
            try:
                # Convert dict to Pydantic Model using the Union
                # This ensures we get the specific config class (e.g. AlloySystemConfig)
                sub_config = TypeAdapter(SystemConfig).validate_python(sub_conf_dict)
                
                # Derive a seed for this sub-generator to ensure reproducibility
                # but distinct from others.
                sub_seed = (self.seed + i * 113) if self.seed is not None else None
                
                # Create generator
                gen = GeneratorFactory.get_generator(sub_config, seed=sub_seed)
                
                # Generate
                structs = gen.generate()
                logger.info(f"Sub-system {i+1} ({sub_config.type}) produced {len(structs)} structures.")
                
                all_structures.extend(structs)
                
            except Exception as e:
                logger.error(f"Failed to generate for sub-system {i}: {e}")
                raise GenerationError(f"Mixed generation failed at index {i}: {e}")
                
        return all_structures
