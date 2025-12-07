import logging
import concurrent.futures
import numpy as np
import time
import random
from typing import List, Optional, Dict, Any, Tuple
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from ase.data import covalent_radii
from tqdm import tqdm
from nnp_gen.core.interfaces import IExplorer
from nnp_gen.core.config import ExplorationConfig, MonteCarloConfig, EnsembleType, MCStrategy
from nnp_gen.core.physics import detect_vacuum
from nnp_gen.explorers.mc_moves import perform_mc_swap
import psutil
import multiprocessing
import queue
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MDTimeoutError(Exception):
    pass

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
            logger.debug(f"Could not set torch threads (already initialized): {e}")

    except ImportError:
        pass

    if model_name == "mace":
        try:
            from mace.calculators import mace_mp
            return mace_mp(model="small", device=device, default_dtype="float32")
        except ImportError:
             logger.warning("mace_mp not found in mace.calculators. Falling back to MACECalculator with direct path check (will likely fail if 'small' is not a file).")
             from mace.calculators import MACECalculator
             return MACECalculator(model_paths="small", device=device, default_dtype="float32")
        except Exception as e:
             logger.error(f"Error loading MACE model: {e}")
             raise e

    elif model_name == "sevenn":
        try:
            from sevenn.calculators import SevenNetCalculator
            return SevenNetCalculator(model="7net-0", device=device)
        except ImportError:
             raise ImportError("sevenn not found")

    elif model_name == "emt":
        from ase.calculators.emt import EMT
        return EMT()

    else:
        raise ImportError(f"Unknown model: {model_name}")

class CalculatorPool:
    """
    Thread-safe pool of calculators.
    """
    def __init__(self, model_name: str, device: str, size: int):
        self.queue = queue.Queue(maxsize=size)
        self.model_name = model_name
        self.device = device
        self.size = size

        logger.info(f"Initializing CalculatorPool with {size} calculators ({model_name})...")
        for _ in range(size):
             calc = _get_calculator(model_name, device)
             self.queue.put(calc)

    @contextmanager
    def get_calculator(self):
        calc = self.queue.get()
        try:
            yield calc
        finally:
            self.queue.put(calc)

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
            logger.info("Vacuum/Slab detected: Using NVT (Langevin).")
            use_npt = False
        else:
            logger.info("Bulk detected: Using NPT.")
            use_npt = True
    elif ensemble == EnsembleType.NPT:
        if is_vacuum:
            logger.warning("User forced NPT on a system with Vacuum. This may lead to infinite expansion.")
        use_npt = True
    else: # NVT
        use_npt = False

    if use_npt:
        # NPT requires stress support.
        # Pressure: default 0 (None in config implies 0 if NPT selected? Or we need pressure config)
        pressure = expl_config.pressure if expl_config.pressure is not None else 0.0
        # Convert GPa to internal units
        # ASE units: GPa is not direct, but units.GPa exists.
        # pressure * units.GPa
        p_internal = pressure * units.GPa

        # ttime: Characteristic time for thermostat (usually 20*dt)
        # pfactor: Bulk modulus estimate * time^2.
        # ASE NPT defaults are often sufficient but let's be explicit if needed.
        # We stick to standard defaults for robustness.
        ttime = 25 * units.fs
        pfactor = 75**2 * units.fs**2 # Default-ish

        # Check if calculator supports stress? ASE will raise error if not.
        return NPT(atoms, timestep=timestep, temperature_K=current_temp, externalstress=p_internal, ttime=ttime, pfactor=pfactor)
    else:
        # NVT (Langevin)
        friction = 0.002
        return Langevin(atoms, timestep=timestep, temperature_K=current_temp, friction=friction)

def run_single_md_thread(
    atoms: Atoms,
    expl_config: ExplorationConfig,
    calc_pool: CalculatorPool,
    timeout_seconds: int = 3600
) -> Optional[List[Atoms]]:
    """
    Run a single MD trajectory using a calculator from the pool.
    Designed for ThreadPoolExecutor.
    """
    start_time = time.time()

    # Unpack config
    steps = expl_config.steps
    timestep_fs = expl_config.timestep

    # Temp
    if expl_config.temperature_mode == "gradient":
        temp_start = expl_config.temp_start
        temp_end = expl_config.temp_end
    else:
        temp_start = expl_config.temperature
        temp_end = expl_config.temperature

    mc_config = expl_config.mc_config

    try:
        with calc_pool.get_calculator() as calc:
            atoms.calc = calc

            # Initialize velocities at start temperature
            MaxwellBoltzmannDistribution(atoms, temperature_K=temp_start)
            Stationary(atoms)

            # Setup Dynamics via Factory
            dyn = _get_integrator(atoms, expl_config, temp_start)

            trajectory = []

            # Precompute thresholds for overlap check
            radii = covalent_radii[atoms.numbers]
            sum_radii = radii[:, None] + radii[None, :]
            overlap_threshold_ratio = 0.5
            thresholds = sum_radii * overlap_threshold_ratio
            np.fill_diagonal(thresholds, 0.0)

            def step_check():
                if time.time() - start_time > timeout_seconds:
                    raise MDTimeoutError("MD Simulation timed out.")
                try:
                    dists = atoms.get_all_distances(mic=True)
                    np.fill_diagonal(dists, np.inf)
                    min_d = np.min(dists)
                    if min_d < 0.5:
                        raise RuntimeError(f"Structure Exploded: min_dist {min_d:.2f} < 0.5")
                    if np.any(dists < thresholds):
                         raise RuntimeError("Structure Exploded: Atomic overlap detected")
                except ValueError:
                    pass

            step_check()

            # --- HYBRID LOOP ---
            # Chunking Logic

            swap_interval = mc_config.swap_interval if (mc_config and mc_config.enabled) else steps
            # Snapshot interval for output
            snap_interval = max(1, steps // 50)

            total_steps_done = 0

            while total_steps_done < steps:
                # Determine chunk size
                chunk = min(swap_interval, steps - total_steps_done)
                if chunk < 1:
                    break

                # Gradient Temp Update (at start of chunk, approx)
                if abs(temp_end - temp_start) > 1e-6:
                    current_t = temp_start + (temp_end - temp_start) * (total_steps_done / steps)

                    # 1. Scale Velocities (Kinetic Energy)
                    dyn.set_temperature(temperature_K=current_t)

                    # 2. Update Thermostat Target (The Heat Bath)
                    # For ASE Langevin (uses 'temp')
                    if hasattr(dyn, 'temp'):
                         dyn.temp = current_t

                    # For ASE NPT (uses 'temperature')
                    elif hasattr(dyn, 'temperature'):
                        dyn.temperature = current_t

                    # For older ASE NPT/NVT versions
                    elif hasattr(dyn, 'temperature_K'):
                        dyn.temperature_K = current_t
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
                    # Trigger swap
                    mc_temp = mc_config.temp if mc_config.temp else current_t
                    perform_mc_swap(atoms, mc_config, mc_temp, calc)

                if steps >= 10 and total_steps_done % max(1, steps // 10) < chunk:
                     logger.info(f"MD Progress: {total_steps_done}/{steps}")

            atoms.calc = None
            return trajectory

    except MDTimeoutError:
        logger.warning("MD Simulation timed out.")
        return None
    except RuntimeError as e:
        logger.warning(f"MD Exploration Failed (RuntimeError): {e}")
        return None
    except Exception as e:
        logger.error(f"MD Exploration Error: {e}")
        return None

def _calculate_max_workers(n_atoms_estimate: int = 1000) -> int:
    """
    Calculate optimal number of workers based on system resources.
    """
    cpu_count = multiprocessing.cpu_count()
    try:
        mem = psutil.virtual_memory()
        available_ram_mb = mem.available / (1024 * 1024)
        mem_per_worker_mb = 200 + (n_atoms_estimate * 0.001)
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
        Run parallel MD on seeds.
        """
        results = []

        # Prepare params
        expl = self.config.exploration
        
        # Determine max workers
        if n_workers is None:
            n_atoms = len(seeds[0]) if seeds else 1000
            n_workers = _calculate_max_workers(n_atoms)

        logger.info(f"Starting MD Exploration with {n_workers} threads. Method: {expl.method}")

        model_name = expl.model_name
        device = "cpu"

        # Initialize Calculator Pool
        pool = CalculatorPool(model_name, device, size=n_workers)

        # Use ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for seed in seeds:
                futures.append(
                    executor.submit(run_single_md_thread, seed, expl, pool, 3600)
                )

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(seeds), desc="MD Exploration"):
                try:
                    traj = future.result()
                    if traj is not None:
                        results.extend(traj)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        return results
