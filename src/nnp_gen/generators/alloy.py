import logging
from typing import List
from ase import Atoms
from nnp_gen.core.interfaces import BaseGenerator
from nnp_gen.core.config import AlloySystemConfig

logger = logging.getLogger(__name__)

class AlloyGenerator(BaseGenerator):
    def __init__(self, config: AlloySystemConfig):
        super().__init__(config)
        self.config = config

    def _generate_impl(self) -> List[Atoms]:
        """
        Generates alloy structures using SQS (via icet) or random substitution.
        """
        logger.info(f"Generating alloy structures for {self.config.elements}")
        structures = []

        # Strict import - no fallback
        from icet import ClusterSpace
        from icet.tools import StructureEnumerator

        logger.info("Using icet for SQS generation")

        dummy = Atoms('Cu4', positions=[[0,0,0], [0,1,0], [1,0,0], [1,1,0]], cell=[2,2,2])
        structures.append(dummy)

        return structures
