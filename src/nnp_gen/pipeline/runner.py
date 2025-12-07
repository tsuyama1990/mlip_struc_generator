import logging
import os
import shutil
from typing import List, Optional, Any
from ase import Atoms

from nnp_gen.core.config import AppConfig
from nnp_gen.core.models import StructureMetadata
from nnp_gen.core.interfaces import IExplorer, ISampler, IStorage, IExporter, BaseGenerator
from nnp_gen.generators.factory import GeneratorFactory
from nnp_gen.explorers.md_engine import MDExplorer
from nnp_gen.samplers.selector import FPSSampler, RandomSampler, DescriptorManager
from nnp_gen.core.storage import ASEDbStorage
from nnp_gen.core.io import ASEExporter
from nnp_gen.core.exceptions import (
    PipelineError, GenerationError, ExplorationError,
    PhysicsViolationError, TimeoutError, ConvergenceError, ConfigurationError
)

logger = logging.getLogger(__name__)

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
        self.debug_dir = os.path.join(self.config.output_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)

        # Migration Check: Legacy Pickle Files
        ckpt_dir = os.path.join(self.config.output_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            pkl_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pkl')]
            if pkl_files:
                msg = (
                    f"Found legacy pickle checkpoints in {ckpt_dir}: {pkl_files}. "
                    "These are no longer supported. Please delete the 'checkpoints' directory "
                    "and restart to migrate to the new database-based checkpoint system."
                )
                logger.error(msg)
                raise ConfigurationError(msg)

        # Setup Storage Backends
        # Checkpoints DB for intermediate steps
        ckpt_path = os.path.join(self.config.output_dir, "checkpoints.db")
        self.ckpt_storage = ASEDbStorage(ckpt_path)

        # Final Dataset DB
        # User supplied storage or default
        if storage:
            self.storage = storage
        else:
            db_path = os.path.join(self.config.output_dir, "dataset.db")
            self.storage = ASEDbStorage(db_path)

        # Dependency Injection with Defaults
        self.generator = generator or GeneratorFactory.get_generator(self.config.system)

        if explorer:
            self.explorer = explorer
        else:
             if self.config.exploration.method == "md":
                self.explorer = MDExplorer(self.config)
                # Pass debug_dir to explorer if possible?
                # Currently MDExplorer doesn't take debug_dir in init.
                # We will handle debug dumping in runner, but worker processes also dump locally.
                # To inform worker processes about debug dir, we might need to pass it in explore()
                # or via config hack. For now, MDExplorer will dump to CWD/debug or we assume explore()
                # has been updated. (Checked md_engine.py, it doesn't take debug_dir yet).
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

        self.exporter = exporter or ASEExporter()

    def _dump_debug_structure(self, atoms: Optional[Atoms], prefix: str, error_msg: str):
        """
        Dumps a failed structure to the debug directory.
        """
        if atoms is None:
            return

        try:
            filename = f"{prefix}_failed.xyz"
            path = os.path.join(self.debug_dir, filename)
            # Add error info to atoms
            atoms.info['error'] = str(error_msg)
            from ase.io import write
            write(path, atoms)
            logger.info(f"Dumped failed structure to {path}")
        except Exception as e:
            logger.error(f"Failed to dump debug structure: {e}")

    def run(self):
        logger.info("Starting Pipeline...")

        # 1. Initialization / Generation
        logger.info("Step 1: Structure Generation")

        # Config hash for metadata
        config_hash_val = f"hash_{hash(str(self.config))}"

        initial_structures = []
        try:
            initial_structures = self.generator.generate()
            logger.info(f"Generated {len(initial_structures)} initial structures.")

            if initial_structures:
                # Checkpoint immediately
                meta_list = [
                    {"source": "seed", "config_hash": config_hash_val, "stage": "generated"}
                    for _ in initial_structures
                ]
                self.ckpt_storage.bulk_save(initial_structures, meta_list)

                # Export for user (optional but nice)
                self.exporter.export(initial_structures, os.path.join(self.config.output_dir, "initial_structures.xyz"))

        except GenerationError as e:
            logger.error(f"Generation failed: {e}")
            raise e
        except Exception as e:
             logger.error(f"Unexpected error in generation: {e}")
             raise GenerationError(str(e))

        if not initial_structures:
            logger.error("No structures generated. Aborting.")
            return

        # 2. Exploration (MD)
        logger.info("Step 2: Exploration")

        explored_structures = []

        if self.explorer:
            # Batch Processing
            batch_size = 50

            # Simple chunking
            for i in range(0, len(initial_structures), batch_size):
                batch = initial_structures[i:i + batch_size]
                try:
                    logger.info(f"Exploring batch {i // batch_size + 1}...")

                    # Pass debug_dir to explore if we update the interface.
                    # Currently strict interface says explore(seeds, n_workers).
                    # We can't easily change strict IExplorer interface in this Epic without touching other files?
                    # IExplorer is in interfaces.py.
                    # Let's rely on MDExplorer to handle its own debug dumping (in workers)
                    # or update MDExplorer to use CWD/debug if not specified.
                    # Or we can pass debug_dir via config injection if ExplorerConfig allows.
                    # For now, runner will dump the *seed* which failed the batch.

                    explored_batch = self.explorer.explore(batch)

                    if explored_batch:
                        # Checkpoint immediately
                        meta_list = [
                            {"source": "md", "config_hash": config_hash_val, "stage": "explored"}
                            for _ in explored_batch
                        ]
                        self.ckpt_storage.bulk_save(explored_batch, meta_list)
                        explored_structures.extend(explored_batch)

                except (PhysicsViolationError, TimeoutError, ConvergenceError) as e:
                    logger.error(f"Exploration batch failed with specific error: {e}")
                    self._dump_debug_structure(batch[0] if batch else None, f"batch_{i}", str(e))
                except Exception as e:
                    logger.error(f"Exploration batch failed unexpectedly: {e}")
                    self._dump_debug_structure(batch[0] if batch else None, f"batch_{i}_crash", str(e))
                    continue
        else:
             logger.warning(f"Exploration disabled. Using initial structures.")
             explored_structures = initial_structures

             # Checkpoint "explored" (pass-through)
             meta_list = [
                {"source": "seed", "config_hash": config_hash_val, "stage": "explored_skipped"}
                for _ in explored_structures
             ]
             self.ckpt_storage.bulk_save(explored_structures, meta_list)

        if explored_structures:
             self.exporter.export(explored_structures, os.path.join(self.config.output_dir, "explored_structures.xyz"))

        logger.info(f"Total structures after exploration: {len(explored_structures)}")

        # 3. Sampling
        logger.info("Step 3: Sampling")

        sampled_structures = []
        sampling_config = self.config.sampling

        if explored_structures:
            n_samples = sampling_config.n_samples
            if self.sampler:
                sampled_structures = self.sampler.sample(explored_structures, n_samples)
            else:
                sampled_structures = explored_structures

        logger.info(f"Selected {len(sampled_structures)} structures.")

        if not sampled_structures:
            logger.warning("No structures selected.")
            return

        # 4. Storage (Final)
        logger.info("Step 4: Saving to Database")

        final_metadata_list = []
        for i, atoms in enumerate(sampled_structures):
            meta = StructureMetadata(
                source="md" if self.config.exploration.method == "md" else "seed",
                config_hash=config_hash_val,
                is_sampled=True,
                sampling_method=sampling_config.strategy
            )
            final_metadata_list.append(meta)

        try:
            saved_ids = self.storage.bulk_save(sampled_structures, final_metadata_list)
            logger.info(f"Saved {len(saved_ids)} structures to dataset database.")

            self.exporter.export(sampled_structures, os.path.join(self.config.output_dir, "sampled_structures.xyz"))

        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise PipelineError(f"Database save failed: {e}")

        logger.info("Pipeline Complete.")
