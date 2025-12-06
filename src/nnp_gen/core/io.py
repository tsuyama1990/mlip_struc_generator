import logging
from typing import List
from ase import Atoms
from ase.io import write
from nnp_gen.core.interfaces import IExporter

logger = logging.getLogger(__name__)

class ASEExporter(IExporter):
    """
    Exports structures using ASE's IO capabilities.
    """
    def export(self, structures: List[Atoms], output_path: str):
        try:
            write(output_path, structures)
            logger.info(f"Exported {len(structures)} structures to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export structures to {output_path}: {e}")
