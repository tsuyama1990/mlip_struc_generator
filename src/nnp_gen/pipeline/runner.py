import logging
import os
import pickle
from typing import List, Optional, Any
from ase import Atoms
from ase.io import read, write

from nnp_gen.core.config import AppConfig
from nnp_gen.core.models import StructureMetadata
from nnp_gen.core.interfaces import IExplorer, ISampler, IStorage, IExporter, BaseGenerator
from nnp_gen.generators.factory import GeneratorFactory
from nnp_gen.explorers.md_engine import MDExplorer
from nnp_gen.samplers.selector import FPSSampler, RandomSampler, DescriptorManager
from nnp_gen.core.storage import DatabaseManager
from nnp_gen.core.io import ASEExporter
from nnp_gen.core.exceptions import PipelineError, GenerationError, ExplorationError

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, name: str, data: Any):
        """
        Save data to a pickle file.
        Using .pkl extension as requested for full state preservation.
        """
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Checkpoint saved: {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {name}: {e}")

    def load(self, name: str) -> Optional[Any]:
        """
        Load data from a pickle file.
        """
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Checkpoint loaded: {path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {name}: {e}")
                return None
        return None

    def exists(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.checkpoint_dir, f"{name}.pkl"))

class PipelineRunner:
    def __init__(
        self,
        config: AppConfig,
        generator: Optional[BaseGenerator] = None,
        explorer: Optional[IExplorer] = None,
        sampler: Optional[ISampler] = None,
        storage: Optional[IStorage] = None,
        exporter: Optional[IExporter] = None
    ):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup Checkpoint Manager
        self.ckpt_mgr = CheckpointManager(os.path.join(self.config.output_dir, "checkpoints"))

        # Dependency Injection with Defaults
        self.generator = generator or GeneratorFactory.get_generator(self.config.system)

        if explorer:
            self.explorer = explorer
        else:
             if self.config.exploration.method == "md":
                self.explorer = MDExplorer(self.config)
             else:
                self.explorer = None # Or a NullExplorer

        if sampler:
             self.sampler = sampler
        else:
            # Factory logic for sampler could be extracted too
            if self.config.sampling.strategy == "fps":
                desc_mgr = DescriptorManager(rcut=self.config.system.constraints.r_cut)
                self.sampler = FPSSampler(desc_mgr)
            elif self.config.sampling.strategy == "random":
                self.sampler = RandomSampler()
            else:
                self.sampler = None

        if storage:
            self.storage = storage
        else:
             db_path = os.path.join(self.config.output_dir, "dataset.db")
             self.storage = DatabaseManager(db_path)

        self.exporter = exporter or ASEExporter()


    def run(self):
        logger.info("Starting Pipeline...")

        # 1. Initialization / Generation
        logger.info("Step 1: Structure Generation")

        # Checkpoint name: checkpoint_generated
        initial_structures = self.ckpt_mgr.load("checkpoint_generated")
        if not initial_structures:
            try:
                initial_structures = self.generator.generate()
                logger.info(f"Generated {len(initial_structures)} initial structures.")

                if initial_structures:
                    self.ckpt_mgr.save("checkpoint_generated", initial_structures)
                    # Also export for user
                    self.exporter.export(initial_structures, os.path.join(self.config.output_dir, "initial_structures.xyz"))
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise GenerationError(str(e))

        if not initial_structures:
            logger.error("No structures generated. Aborting.")
            return

        # 2. Exploration (MD)
        logger.info("Step 2: Exploration")

        # Checkpoint name: checkpoint_explored
        explored_structures = self.ckpt_mgr.load("checkpoint_explored")
        if not explored_structures:
            if self.explorer:
                # Use psutil-based dynamic workers inside MDExplorer (handled by default)
                # Or pass explicitly if needed.
                try:
                    explored_structures = self.explorer.explore(initial_structures)
                except Exception as e:
                    logger.error(f"Exploration failed: {e}")
                    raise ExplorationError(str(e))
            else:
                 logger.warning(f"Exploration method {self.config.exploration.method} not implemented/supported or disabled. Using initial structures.")
                 explored_structures = initial_structures

            # If exploration returns empty (e.g. all exploded), we might fallback or abort
            if not explored_structures:
                logger.warning("No valid structures from exploration. Using initial structures if any.")
                explored_structures = initial_structures

            if explored_structures:
                self.ckpt_mgr.save("checkpoint_explored", explored_structures)
                self.exporter.export(explored_structures, os.path.join(self.config.output_dir, "explored_structures.xyz"))

        logger.info(f"Total structures after exploration: {len(explored_structures)}")

        # 3. Sampling
        logger.info("Step 3: Sampling")

        # Checkpoint name: checkpoint_sampled
        # Note: Previous plan used "step3_sampling", changing to "checkpoint_sampled" for consistency with user request (checkpoint_*.pkl)
        # However user specifically asked for checkpoint_generated and checkpoint_explored. I will use checkpoint_sampled for step 3.
        sampled_structures = self.ckpt_mgr.load("checkpoint_sampled")
        sampling_config = self.config.sampling

        if not sampled_structures:
            n_samples = sampling_config.n_samples

            sampled_structures = []
            if explored_structures:
                if self.sampler:
                    sampled_structures = self.sampler.sample(explored_structures, n_samples)
                else:
                    logger.info("Manual sampling or unknown strategy, keeping all.")
                    sampled_structures = explored_structures

            if sampled_structures:
                self.ckpt_mgr.save("checkpoint_sampled", sampled_structures)

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
            saved_ids = self.storage.bulk_save(sampled_structures, metadata_list)
            logger.info(f"Saved {len(saved_ids)} structures to database.")

            # Export sampled structures
            self.exporter.export(sampled_structures, os.path.join(self.config.output_dir, "sampled_structures.xyz"))

        except Exception as e:
            logger.error(f"Database save failed: {e}")
            # Raise generic PipelineError?
            raise PipelineError(f"Database save failed: {e}")

        logger.info("Pipeline Complete.")
