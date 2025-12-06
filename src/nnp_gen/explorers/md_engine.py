import logging
import concurrent.futures
import numpy as np
from typing import List, Optional, Dict, Any
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from ase.data import covalent_radii
from tqdm import tqdm
from nnp_gen.core.interfaces import IExplorer
import psutil
import os
import signal
import multiprocessing
import queue
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MDTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise MDTimeoutError("MD Simulation timed out.")

def _get_calculator(model_name: str, device: str):
    """
    Factory for calculators.
    Sets torch threads to 1 for process parallelism.
    """
    try:
        import torch
        # Force single thread for process parallelism
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass

    if model_name == "mace":
        try:
            from mace.calculators import MACECalculator
            return MACECalculator(model_paths=None, model_type="mace_mp_0_small", device=device, default_dtype="float32")
        except ImportError:
             raise ImportError("mace not found")
        except Exception as e:
             # Fallback for different signature
             from mace.calculators import MACECalculator
             return MACECalculator(model_paths=None, device=device)

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

        # Pre-fill? Or lazy?
        # Lazy is safer to avoid long startup.
        # But we must ensure thread safety during creation if multiple threads hit empty queue.
        # Actually, best pattern is:
        # Workers try to get from queue. If empty and total created < size, create new.
        # But queue.Queue doesn't track "total created".
        # Simplest: Pre-fill.
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

def run_single_md_thread(
    atoms: Atoms,
    temp: float,
    steps: int,
    interval: int,
    calc_pool: CalculatorPool,
    timeout_seconds: int = 3600
) -> Optional[List[Atoms]]:
    """
    Run a single MD trajectory using a calculator from the pool.
    Designed for ThreadPoolExecutor.
    """
    # Signal handling for timeout works in main thread only.
    # For threads, we can't use signal.alarm effectively per thread.
    # We rely on total job timeout or manual checks?
    # Or strict step limits.
    # The prompt asked for timeout using signal. That implies Process based or single thread.
    # If we switch to Threads for Pool, we lose signal-based timeout per task.
    # However, "Issue: Loading ML models... for every thread is inefficient."
    # If we use Threads, we MUST share calculators.
    # If we use Processes, we can't share calculators via Queue easily.
    # So assuming ThreadPoolExecutor.
    # Timeout in threads: We can check time in the loop.

    import time
    start_time = time.time()

    try:
        with calc_pool.get_calculator() as calc:
            # Clone atoms to avoid mutating the seed in place if shared (seeds are usually unique copies)
            # atoms = atoms.copy() # Caller passes unique seed? Yes.

            atoms.calc = calc

            # Initialize velocities
            MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
            Stationary(atoms)

            # Setup Dynamics
            dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=temp, friction=0.002)

            trajectory = []

            # Precompute thresholds for overlap check
            radii = covalent_radii[atoms.numbers]
            sum_radii = radii[:, None] + radii[None, :]
            overlap_threshold_ratio = 0.5
            thresholds = sum_radii * overlap_threshold_ratio
            np.fill_diagonal(thresholds, 0.0)

            def step_check():
                # Timeout check
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

            # Initial check
            step_check()

            for i in range(steps):
                dyn.run(1)
                step_check()

                if (i + 1) % interval == 0:
                    snap = atoms.copy()
                    snap.calc = None # Detach calculator
                    trajectory.append(snap)

            atoms.calc = None # Detach before returning to pool logic (though 'calc' var is local)
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
        # Using ThreadPool, memory per worker is lower (model shared).
        # But we still have trajectories.
        # Let's keep conservative estimate.
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
        temp = self.config.exploration.temperature
        steps = self.config.exploration.steps
        interval = max(1, steps // 50)

        # Determine max workers
        if n_workers is None:
            n_atoms = len(seeds[0]) if seeds else 1000
            n_workers = _calculate_max_workers(n_atoms)

        logger.info(f"Starting MD Exploration with {n_workers} threads.")

        model_name = self.config.exploration.model_name
        device = "cpu" # or from config

        # Initialize Calculator Pool
        # Pool size = n_workers
        pool = CalculatorPool(model_name, device, size=n_workers)

        # Use ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for seed in seeds:
                futures.append(
                    executor.submit(run_single_md_thread, seed, temp, steps, interval, pool, 3600)
                )

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(seeds), desc="MD Exploration"):
                try:
                    traj = future.result()
                    if traj is not None:
                        results.extend(traj)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        return results
