import logging
import os
from typing import List, Optional
from ase import Atoms
from ase.io import write
from nnp_gen.core.config import AppConfig
from nnp_gen.core.models import StructureMetadata
from nnp_gen.core.storage import DatabaseManager
from nnp_gen.generators.factory import GeneratorFactory
from nnp_gen.explorers.md_engine import MDExplorer
from nnp_gen.samplers.selector import FPSSampler, RandomSampler, DescriptorManager

logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.db_path = os.path.join(self.config.output_dir, "dataset.db")
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.db_manager = DatabaseManager(self.db_path)

    def export_xyz(self, structures: List[Atoms], output_path: str):
        """Export list of structures to XYZ format."""
        try:
            write(output_path, structures)
            logger.info(f"Exported {len(structures)} structures to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export structures to {output_path}: {e}")

    def run(self):
        logger.info("Starting Pipeline...")

        # 1. Initialization / Generation
        logger.info("Step 1: Structure Generation")
        try:
            generator = GeneratorFactory.get_generator(self.config.system)
            initial_structures = generator.generate()
            logger.info(f"Generated {len(initial_structures)} initial structures.")

            # Checkpoint
            if initial_structures:
                self.export_xyz(initial_structures, os.path.join(self.config.output_dir, "initial_structures.xyz"))

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return

        if not initial_structures:
            logger.error("No structures generated. Aborting.")
            return

        # 2. Exploration (MD)
        logger.info("Step 2: MD Exploration")
        if self.config.exploration.method == "md":
            explorer = MDExplorer(self.config)
            n_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
            # MDExplorer returns flat list of Atoms
            explored_structures = explorer.explore(initial_structures, n_workers=n_workers)
        else:
             logger.warning(f"Exploration method {self.config.exploration.method} not implemented/supported. Using initial structures.")
             explored_structures = initial_structures

        # If exploration returns empty (e.g. all exploded), we might fallback or abort
        if not explored_structures:
            logger.warning("No valid structures from exploration. Using initial structures if any.")
            explored_structures = initial_structures

        # Checkpoint
        self.export_xyz(explored_structures, os.path.join(self.config.output_dir, "explored_structures.xyz"))

        logger.info(f"Total structures after exploration: {len(explored_structures)}")

        # 3. Sampling
        logger.info("Step 3: Sampling")
        sampling_config = self.config.sampling
        n_samples = sampling_config.n_samples

        sampled_structures = []
        if explored_structures:
            if sampling_config.strategy == "fps":
                desc_mgr = DescriptorManager(rcut=self.config.system.constraints.r_cut)
                sampler = FPSSampler(desc_mgr)
                sampled_structures = sampler.sample(explored_structures, n_samples)
            elif sampling_config.strategy == "random":
                sampler = RandomSampler()
                sampled_structures = sampler.sample(explored_structures, n_samples)
            else:
                logger.info("Manual sampling or unknown strategy, keeping all.")
                sampled_structures = explored_structures

        logger.info(f"Selected {len(sampled_structures)} structures.")

        if not sampled_structures:
            logger.warning("No structures selected.")
            return

        # 4. Storage
        logger.info("Step 4: Saving to Database")
        metadata_list = []
        # Prefix with hash_ to avoid ase.db type inference issues
        config_hash_val = f"hash_{hash(str(self.config))}"

        for i, atoms in enumerate(sampled_structures):
            meta = StructureMetadata(
                source="md" if self.config.exploration.method == "md" else "seed",
                config_hash=config_hash_val,
                is_sampled=True,
                sampling_method=sampling_config.strategy
            )
            metadata_list.append(meta)

        try:
            saved_ids = self.db_manager.bulk_save(sampled_structures, metadata_list)
            logger.info(f"Saved {len(saved_ids)} structures to {self.db_path}")

            # Export sampled structures
            self.export_xyz(sampled_structures, os.path.join(self.config.output_dir, "sampled_structures.xyz"))

        except Exception as e:
            logger.error(f"Database save failed: {e}")

        logger.info("Pipeline Complete.")
