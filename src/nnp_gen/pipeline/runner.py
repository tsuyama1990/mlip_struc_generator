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
        self.generator = generator or GeneratorFactory.get_generator(self.config.system, seed=self.config.seed)

        if explorer:
            self.explorer = explorer
        else:
             if self.config.exploration.method in ["md", "hybrid_mc_md"]:
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

        initial_structures_path = os.path.join(self.config.output_dir, "initial_structures.xyz")
        config_hash_val = f"hash_{hash(str(self.config))}"

        # --- STEP 1: ACQUIRE STRUCTURES (Unified Workflow) ---
        # Strategy:
        # 1. Check if 'initial_structures.xyz' exists in output (Resume Mode).
        # 2. If not, Generate using self.generator (Generate Mode).
        # 3. If Generate Mode, Save to disk immediately.
        
        from ase.io import read
        
        if os.path.exists(initial_structures_path):
             logger.info(f"Found existing {initial_structures_path}. Resuming/Using existing structures.")
             # We rely on reloading in Step 2, so strictly we don't need to load 'initial_structures' variable here
             # unless we want to check count.
             # But for code clarity, let's just confirm it's valid.
             pass 
        else:
            logger.info("Generating initial structures...")
            try:
                # This works for both 'alloy', 'random', AND 'from_files' (FileLoaderGenerator)
                generated_structures = self.generator.generate()
                logger.info(f"Generated/Loaded {len(generated_structures)} structures.")

                if not generated_structures:
                    logger.error("No structures generated. Aborting.")
                    return

                # Checkpoint & Export
                meta_list = [
                    {"source": "seed", "config_hash": config_hash_val, "stage": "generated"}
                    for _ in generated_structures
                ]
                self.ckpt_storage.bulk_save(generated_structures, meta_list)
                
                # Save to disk (Canonical Source for Step 2)
                self.exporter.export(generated_structures, initial_structures_path)
                
                # Clear memory
                del generated_structures

            except GenerationError as e:
                logger.error(f"Generation failed: {e}")
                raise e
            except Exception as e:
                 logger.error(f"Unexpected error in generation: {e}")
                 raise GenerationError(str(e))

        # --- STEP 2: STATE ISOLATION (Reload) ---
        # Whether we generated or resumed, we now Load fresh from disk.
        logger.info("loading structures from disk for processing...")
        if not os.path.exists(initial_structures_path):
            raise PipelineError(f"Critical: {initial_structures_path} missing after generation step.")
            
        initial_structures = read(initial_structures_path, index=':')
        logger.info(f"Loaded {len(initial_structures)} structures for exploration.")

        # 2. Exploration (MD)
        logger.info("Step 2: Exploration")

        explored_structures = []
        
        # Cleanup old progressive file
        prog_path = os.path.join(self.config.output_dir, "explored_structures_progressive.xyz")
        if os.path.exists(prog_path):
            try:
                os.remove(prog_path)
            except OSError:
                pass

        if self.explorer:
            # Batch Processing
            # Segregation Mode: Run 1 by 1 as requested to ensure full isolation.
            batch_size = 1

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

                    # Define progressive callback
                    progressive_xyz_path = os.path.join(self.config.output_dir, "explored_structures_progressive.xyz")
                    
                    # Ensure file is clean for first batch (or append if later batches? we loop batches)
                    # If i==0, maybe clean? But explore loop handles one batch. 
                    # Actually, if we want a single file, we should appendMode.
                    # ASE write supports append=True.
                    
                    def progressive_save(new_structures: List[Atoms]):
                        from ase.io import write
                        # Append to XYZ
                        write(progressive_xyz_path, new_structures, append=True)
                        logger.debug(f"Progressively saved {len(new_structures)} structures to {progressive_xyz_path}")

                    explored_batch = self.explorer.explore(batch, callback=progressive_save)

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
                    raise e
                except Exception as e:
                    logger.error(f"Exploration batch failed unexpectedly: {e}")
                    self._dump_debug_structure(batch[0] if batch else None, f"batch_{i}_crash", str(e))
                    raise e
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
             
             # Cleanup: Remove progressive file as it's now redundant
             if os.path.exists(prog_path):
                 try:
                     os.remove(prog_path)
                     logger.debug(f"Removed redundant progressive file: {prog_path}")
                 except OSError:
                     pass

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
