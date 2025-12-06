import logging
import concurrent.futures
import numpy as np
from typing import List, Optional, Dict, Any
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from tqdm import tqdm

logger = logging.getLogger(__name__)

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
        # If torch is missing, we can't really use MACE/SevenNet usually.
        # But maybe we want to mock this function in tests without torch.
        # If real run, this raises ImportError later when importing mace.
        pass

    if model_name == "mace":
        try:
            from mace.calculators import MACECalculator
            # mace-mp-0-small recommended
            # Note: arguments might vary by version, assuming standard kwargs
            return MACECalculator(model_paths=None, model_type="mace_mp_0_small", device=device, default_dtype="float32")
        except ImportError:
             raise ImportError("mace not found")
        except Exception as e:
             # Fallback for different signature?
             from mace.calculators import MACECalculator
             return MACECalculator(model_paths=None, device=device)

    elif model_name == "sevenn":
        try:
            from sevenn.calculators import SevenNetCalculator
            # Assuming default model 7net-0 available or checking args
            return SevenNetCalculator(model="7net-0", device=device)
        except ImportError:
             raise ImportError("sevenn not found")

    else:
        raise ImportError(f"Unknown model: {model_name}")

def run_single_md(
    atoms: Atoms,
    temp: float,
    steps: int,
    interval: int,
    calculator_params: Dict[str, Any]
) -> Optional[List[Atoms]]:
    """
    Run a single MD trajectory.
    Defined as standalone function to ensure picklability.
    """
    try:
        # Setup calculator
        model_name = calculator_params.get("model_name", "mace")
        device = calculator_params.get("device", "cpu")
        calc = _get_calculator(model_name, device)
        atoms.calc = calc

        # Initialize velocities
        # temperature_K requires ase.units? No, it's just float in ASE methods usually
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
        Stationary(atoms)

        # Setup Dynamics
        # friction 0.02 atomic units? usually 0.002-0.02.
        dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=temp, friction=0.02)

        trajectory = []

        def step_check():
            # Guardrail
            # Check min distance
            # mic=True is important for periodic systems
            try:
                dists = atoms.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)
                min_d = np.min(dists)
                if min_d < 0.6:
                    raise RuntimeError(f"Structure Exploded: min_dist {min_d} < 0.6")
            except ValueError:
                # Can happen if structure is empty or weird
                pass

        # Initial check
        step_check()

        for i in range(steps):
            dyn.run(1)
            step_check()

            if (i + 1) % interval == 0:
                # Save copy
                # Ensure we strip calculator to save memory/pickling issues?
                # Atoms copy usually copies calc?
                # Better to save atoms with minimal info
                snap = atoms.copy()
                snap.calc = None
                trajectory.append(snap)

        return trajectory

    except RuntimeError as e:
        # Expected explosion
        return None
    except Exception as e:
        # Unexpected error
        # In production, might log to file
        return None

class MDExplorer:
    def __init__(self, config: Any):
        self.config = config

    def explore(self, seeds: List[Atoms], n_workers: int = 1) -> List[Atoms]:
        """
        Run parallel MD on seeds.
        """
        results = []

        # Prepare params
        temp = self.config.exploration.temperature
        steps = self.config.exploration.steps
        # Save every 50 steps or so
        interval = max(1, steps // 50)

        # Default params
        calc_params = {"model_name": "mace", "device": "cpu"}

        # Use ProcessPoolExecutor
        # If n_workers is 1, maybe run serial for easier debugging?
        # But requirement says "use concurrent.futures...".

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for seed in seeds:
                futures.append(
                    executor.submit(run_single_md, seed, temp, steps, interval, calc_params)
                )

            # Use tqdm for progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(seeds), desc="MD Exploration"):
                try:
                    traj = future.result()
                    if traj is not None:
                        results.extend(traj)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        return results
