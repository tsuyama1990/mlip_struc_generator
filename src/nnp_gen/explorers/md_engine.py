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
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
        Stationary(atoms)

        # Setup Dynamics
        # Friction 0.002 (approx 500fs decay time if units are fs)
        dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=temp, friction=0.002)

        trajectory = []

        # Precompute thresholds for overlap check
        radii = covalent_radii[atoms.numbers]
        sum_radii = radii[:, None] + radii[None, :]
        # Allow some overlap (e.g. 50% of sum of radii) before calling it an explosion
        overlap_threshold_ratio = 0.5
        thresholds = sum_radii * overlap_threshold_ratio

        # Set diagonal to zero for threshold or handle dists diagonal
        np.fill_diagonal(thresholds, 0.0)

        def step_check():
            # Guardrail
            try:
                # mic=True is important for periodic systems
                dists = atoms.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)

                # Check absolute minimum (nuclear fusion prevention)
                min_d = np.min(dists)
                if min_d < 0.5:
                    raise RuntimeError(f"Structure Exploded: min_dist {min_d:.2f} < 0.5")

                # Check element-specific overlap
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
                snap.calc = None # Fix memory leak
                trajectory.append(snap)

        return trajectory

    except RuntimeError as e:
        # Expected explosion
        logger.warning(f"MD Exploration Failed (RuntimeError): {e}")
        return None
    except Exception as e:
        # Unexpected error
        logger.error(f"MD Exploration Error: {e}")
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
        interval = max(1, steps // 50)

        # Default params (should be configurable, but keeping simple as per current code)
        calc_params = {"model_name": self.config.exploration.model_name, "device": "cpu"}

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for seed in seeds:
                futures.append(
                    executor.submit(run_single_md, seed, temp, steps, interval, calc_params)
                )

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(seeds), desc="MD Exploration"):
                try:
                    traj = future.result()
                    if traj is not None:
                        results.extend(traj)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        return results
