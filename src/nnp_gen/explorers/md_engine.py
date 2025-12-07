import logging
import concurrent.futures
import numpy as np
import time
import random
import os
from typing import List, Optional, Dict, Any, Tuple
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from ase.data import covalent_radii
from tqdm import tqdm
import tempfile
from nnp_gen.core.interfaces import IExplorer
from nnp_gen.core.config import ExplorationConfig, MonteCarloConfig, EnsembleType, MCStrategy
from nnp_gen.core.physics import detect_vacuum
from nnp_gen.explorers.mc_moves import perform_mc_swap
from nnp_gen.core.exceptions import PhysicsViolationError, TimeoutError
from nnp_gen.core.calculators import CalculatorFactory
import psutil
import multiprocessing

logger = logging.getLogger(__name__)

def _get_calculator(model_name: str, device: str):
    """
    Factory for calculators.
    Sets torch threads to 1 for process parallelism.
    """
    try:
        import torch
        try:
            # Force single thread for process parallelism
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError as e:
            # This is expected if torch is already initialized
            pass

    except ImportError:
        pass

    return CalculatorFactory.get(model_name, device)

def _get_integrator(atoms: Atoms, expl_config: ExplorationConfig, current_temp: float):
    """
    Factory to create the appropriate MD integrator (NVT/NPT).
    """
    timestep = expl_config.timestep * units.fs
    ensemble = expl_config.ensemble

    is_vacuum = detect_vacuum(atoms)

    use_npt = False

    if ensemble == EnsembleType.AUTO:
        if is_vacuum:
            logger.debug("Vacuum/Slab detected: Using NVT (Langevin).")
            use_npt = False
        else:
            logger.debug("Bulk detected: Using NPT.")
            use_npt = True
    elif ensemble == EnsembleType.NPT:
        if is_vacuum:
            logger.warning("User forced NPT on a system with Vacuum. This may lead to infinite expansion.")
        use_npt = True
    else: # NVT
        use_npt = False

    if use_npt:
        pressure = expl_config.pressure if expl_config.pressure is not None else 0.0
        p_internal = pressure * units.GPa
        # Use configured ttime or safe default
        ttime_val = expl_config.ttime if hasattr(expl_config, 'ttime') else 100.0
        ttime = ttime_val * units.fs
        pfactor = 75**2 * units.fs**2 # Default-ish

        return NPT(atoms, timestep=timestep, temperature_K=current_temp, externalstress=p_internal, ttime=ttime, pfactor=pfactor)
    else:
        # NVT (Langevin)
        friction = 0.002
        return Langevin(atoms, timestep=timestep, temperature_K=current_temp, friction=friction)

def run_single_md_process(
    atoms: Atoms,
    expl_config: ExplorationConfig,
    model_name: str,
    device: str,
    timeout_seconds: int = 3600
) -> Optional[List[Atoms]]:
    """
    Run a single MD trajectory.
    Instantiates its own calculator (Late Binding).
    Designed for ProcessPoolExecutor.
    """
    start_time = time.time()

    # Unpack config
    steps = expl_config.steps

    # Temp
    if expl_config.temperature_mode == "gradient":
        temp_start = expl_config.temp_start
        temp_end = expl_config.temp_end
    else:
        temp_start = expl_config.temperature
        temp_end = expl_config.temperature

    mc_config = expl_config.mc_config

    try:
        # Instantiate Calculator Locally
        calc = _get_calculator(model_name, device)
        atoms.calc = calc

        # Initialize velocities at start temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_start)
        Stationary(atoms)

        # Setup Dynamics via Factory
        dyn = _get_integrator(atoms, expl_config, temp_start)

        trajectory = []

        # Check N>1 for Langevin stability
        if len(atoms) <= 1:
             logger.warning("MD simulation requested for single atom. Skipping MD stability check that requires N>1.")
             if len(atoms) == 1:
                  logger.warning("Cannot run Langevin dynamics on a single atom due to ASE limitations. Returning initial structure.")
                  atoms.calc = None
                  return [atoms]

        radii = covalent_radii[atoms.numbers]
        sum_radii = radii[:, None] + radii[None, :]
        overlap_threshold_ratio = 0.5
        thresholds = sum_radii * overlap_threshold_ratio
        np.fill_diagonal(thresholds, 0.0)

        def step_check():
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError("MD Simulation timed out.")
            try:
                # Basic overlap check
                dists = atoms.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)
                min_d = np.min(dists)
                if min_d < 0.5:
                    raise PhysicsViolationError(f"Structure Exploded: min_dist {min_d:.2f} < 0.5")

                # Element-specific check
                if np.any(dists < thresholds):
                     raise PhysicsViolationError("Structure Exploded: Atomic overlap detected")
            except ValueError:
                pass

        step_check()

        # --- HYBRID LOOP ---
        swap_interval = mc_config.swap_interval if (mc_config and mc_config.enabled) else steps
        snap_interval = max(1, steps // 50)

        total_steps_done = 0

        while total_steps_done < steps:
            # Determine chunk size
            stride = min(swap_interval, snap_interval)
            chunk = min(stride, steps - total_steps_done)
            if chunk < 1:
                break

            # Gradient Temp Update
            if abs(temp_end - temp_start) > 1e-6:
                current_t = temp_start + (temp_end - temp_start) * (total_steps_done / steps)
                dyn.set_temperature(temperature_K=current_t)
            else:
                current_t = temp_start

            # Run MD Chunk
            dyn.run(chunk)
            total_steps_done += chunk
            step_check()

            # Snapshot Logic
            if total_steps_done % snap_interval < chunk:
                 snap = atoms.copy()
                 snap.calc = None
                 trajectory.append(snap)

            # MC Logic
            if mc_config and mc_config.enabled:
                mc_temp = mc_config.temp if mc_config.temp else current_t
                perform_mc_swap(atoms, mc_config, mc_temp, calc)

        atoms.calc = None
        return trajectory

    except (PhysicsViolationError, TimeoutError, Exception) as e:
        # Catch all exceptions to dump debug structure locally
        # Since we are in a worker, we should write to a file that won't collide.
        try:
             # Dump to tempdir to prevent pollution
             debug_dir = tempfile.gettempdir()

             pid = os.getpid()
             filename = os.path.join(debug_dir, f"nnp_gen_worker_{pid}_failed.xyz")
             atoms.info['error'] = str(e)
             from ase.io import write
             write(filename, atoms)
             logger.error(f"Worker {pid} failed. Dumped state to {filename}. Error: {e}")
        except Exception as dump_err:
             logger.error(f"Worker {pid} failed and could not dump state: {dump_err}")

        # Re-raise to let caller know
        raise e

def _calculate_max_workers(n_atoms_estimate: int = 1000) -> int:
    """
    Calculate optimal number of workers based on system resources.
    """
    cpu_count = multiprocessing.cpu_count()
    try:
        mem = psutil.virtual_memory()
        available_ram_mb = mem.available / (1024 * 1024)
        mem_per_worker_mb = 300 + (n_atoms_estimate * 0.002) # Adjusted for Process overhead
        max_workers_mem = int(available_ram_mb / mem_per_worker_mb)
    except Exception:
        max_workers_mem = cpu_count

    max_workers = min(max_workers_mem, cpu_count, 16)
    return max(1, max_workers)

class MDExplorer(IExplorer):
    def __init__(self, config: Any):
        self.config = config

    def explore(self, seeds: List[Atoms], n_workers: Optional[int] = None) -> List[Atoms]:
        """
        Run parallel MD on seeds using ProcessPoolExecutor.
        """
        results = []

        # Prepare params
        expl = self.config.exploration
        model_name = expl.model_name

        # Determine device
        device = getattr(expl, 'device', 'cpu')

        # Enforce max_workers=1 for CUDA
        if "cuda" in device.lower():
            if n_workers is not None and n_workers > 1:
                logger.warning("CUDA detected: Forcing max_workers=1 to prevent multiprocessing errors.")
            n_workers = 1
        
        if n_workers is None:
            n_atoms = len(seeds[0]) if seeds else 1000
            n_workers = _calculate_max_workers(n_atoms)

        logger.info(f"Starting MD Exploration with {n_workers} processes. Method: {expl.method} on {device}")

        # Use ProcessPoolExecutor
        # We pass config and model string, not calculator objects.
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for seed in seeds:
                futures.append(
                    executor.submit(run_single_md_process, seed, expl, model_name, device, 3600)
                )

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(seeds), desc="MD Exploration"):
                try:
                    traj = future.result()
                    if traj is not None:
                        results.extend(traj)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
                    # We might want to re-raise specific errors if we want the runner to see them?
                    # The runner catches specific errors now.
                    raise e

        return results
