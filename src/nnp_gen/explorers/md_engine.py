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
from ase.optimize import FIRE
from ase.md.nvtberendsen import NVTBerendsen
from tqdm import tqdm
import tempfile
from nnp_gen.core.interfaces import IExplorer
from nnp_gen.core.config import ExplorationConfig, MonteCarloConfig, EnsembleType, MCStrategy, ThermostatType
from nnp_gen.core.physics import detect_vacuum
from nnp_gen.explorers.mc_moves import perform_mc_swap
from nnp_gen.core.exceptions import PhysicsViolationError, TimeoutError
from nnp_gen.core.calculators import CalculatorFactory
import psutil
import multiprocessing
import queue
from ase.calculators.calculator import Calculator, all_changes
from nnp_gen.core.zbl import ZBLCalculator

logger = logging.getLogger(__name__)



# Global Cache for Worker Process
_CACHED_CALC = None
_CACHED_MODEL_NAME = None
_CACHED_DEVICE = None

def _get_calculator(model_name: str, device: str, zbl_config: Optional[Any] = None):
    """
    Factory for calculators.
    Uses process-local caching to avoid reloading heavy models (like MACE) 
    if the worker process is reused.
    """
    global _CACHED_CALC, _CACHED_MODEL_NAME, _CACHED_DEVICE
    
    # Check if cached calculator is valid
    if _CACHED_CALC is not None:
        if _CACHED_MODEL_NAME == model_name and _CACHED_DEVICE == device:
            # Check if reset needed.
            # IMPORTANT: We must reset the calculator to clear history.
            # For MACE, reset() clears properties.
            if hasattr(_CACHED_CALC, 'reset'):
                 _CACHED_CALC.reset()

            # If ZBL is NOT used, we can return the cached one directly.
            # If ZBL IS used, we need to wrap the cached MACE calculator in a NEW SumCalculator
            # to avoid ZBL state pollution (as noted in audit).

            if not (zbl_config and zbl_config.enabled):
                return _CACHED_CALC
            else:
                # ZBL requested.
                # If cached calc is already a SumCalculator, we should probably unwrap it or just rebuild it.
                # To be safe and since SumCalculator is light, we assume _CACHED_CALC holds the BASE model (MACE).
                # Wait, if we return SumCalculator, next time _CACHED_CALC will be SumCalculator.
                # We should cache the BASE model separately?
                # Actually, simplest is: If ZBL is enabled, we cache the MACE part, and reconstruct the SumCalculator.
                # But _CACHED_CALC stores whatever we return.
                # So we need to store the heavy model in a separate variable if we want to wrap it?
                # Or just check instance type.
                pass
        else:
            logger.debug(f"Calculator mismatch (Req: {model_name}/{device}, Cached: {_CACHED_MODEL_NAME}/{_CACHED_DEVICE}). Reloading...")
            _CACHED_CALC = None # Clear old

    # If we are here, either cache is empty or invalid.
    # BUT, if we returned above, we are done.

    # Refined Logic:
    # 1. We want to cache the HEAVY model (MACE/SevenNet).
    # 2. ZBL/SumCalculator is lightweight glue.
    # 3. We should only cache the heavy model.

    # Let's introduce a specific cache for the heavy model.
    # We will reuse _CACHED_CALC for the heavy model.

    # If the user requested ZBL, we won't cache the Resulting SumCalculator in _CACHED_CALC.
    # Instead we cache the underlying model.

    # However, existing code structure:
    # _CACHED_CALC = calc
    # return calc

    # Let's change behavior: _CACHED_CALC always holds the BASE model.

    base_calc = _CACHED_CALC

    if base_calc is None or _CACHED_MODEL_NAME != model_name or _CACHED_DEVICE != device:
        logger.debug(f"Initializing calculator '{model_name}' on device '{device}'")
        try:
            import torch
            try:
                # Force single thread for process parallelism
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except RuntimeError as e:
                pass
        except ImportError:
            pass

        base_calc = CalculatorFactory.get(model_name, device)
        _CACHED_CALC = base_calc
        _CACHED_MODEL_NAME = model_name
        _CACHED_DEVICE = device
    else:
        # Reset the cached base calculator
        if hasattr(base_calc, 'reset'):
            base_calc.reset()
    
    # --- ZBL INTEGRATION ---
    if zbl_config and zbl_config.enabled:
        from ase.calculators.mixing import SumCalculator
        zbl_calc = ZBLCalculator(cutoff=zbl_config.cutoff)
        # Wrap the cached base calculator
        calc = SumCalculator([base_calc, zbl_calc])
        return calc
    else:
        return base_calc

def _get_integrator(atoms: Atoms, expl_config: ExplorationConfig, current_temp: float):
    """
    Factory to create the appropriate MD integrator (NVT/NPT).
    """
    timestep = expl_config.timestep * units.fs
    ensemble = expl_config.ensemble
    thermostat_type = expl_config.thermostat

    # Short-circuit vacuum detection if user specified ensemble
    if ensemble == EnsembleType.AUTO:
        is_vacuum = detect_vacuum(atoms)
        if is_vacuum:
            logger.debug("Vacuum/Slab detected (AUTO): Using NVT.")
            use_npt = False
        else:
            logger.debug("Bulk detected (AUTO): Using NPT.")
            use_npt = True
    elif ensemble == EnsembleType.NPT:
        use_npt = True
        # Optional: Warn if vacuum
        # if detect_vacuum(atoms): ... (Skipping to save time as requested)
    else: # NVT
        use_npt = False

    if use_npt:
        pressure = expl_config.pressure if expl_config.pressure is not None else 0.0
        p_internal = pressure * units.GPa
        # Use configured ttime or safe default
        ttime_val = expl_config.ttime if hasattr(expl_config, 'ttime') else 100.0
        ttime = ttime_val * units.fs
        pfactor = 75**2 * units.fs**2 # Default-ish

        # Note: ASE NPT uses Nose-Hoover chain by default.
        return NPT(atoms, timestep=timestep, temperature_K=current_temp, externalstress=p_internal, ttime=ttime, pfactor=pfactor)
    else:
        # NVT
        if thermostat_type == ThermostatType.BERENDSEN:
             taut = 100 * units.fs
             return NVTBerendsen(atoms, timestep=timestep, temperature_K=current_temp, taut=taut)

        elif thermostat_type == ThermostatType.NOSE_HOOVER:
             # Use NPT with zero compressibility/masking to act as NVT Nose-Hoover?
             # Or simply NPT with fixed cell?
             # ASE NPT documentation: "NPT dynamics... uses Nose-Hoover dynamics."
             # If we want NVT, we can use NPT with pfactor=None? No, pfactor is bulk modulus related.
             # NPT(..., mask=(0,0,0)) disables cell updates -> NVT.
             # ttime is the thermostat time constant.
             ttime_val = expl_config.ttime if hasattr(expl_config, 'ttime') else 100.0
             ttime = ttime_val * units.fs
             # pfactor is required but irrelevant if mask is (0,0,0)?
             # Let's provide a dummy pfactor.
             return NPT(atoms, timestep=timestep, temperature_K=current_temp,
                        externalstress=0.0, ttime=ttime, pfactor=None, mask=(0,0,0))

        else:
            # Default: Langevin
            if "cuda" in getattr(expl_config, 'device', 'cpu').lower():
                 logger.warning("Using Langevin dynamics on CUDA. If simulation hangs, consider switching 'exploration.thermostat' to 'berendsen'.")

            friction = 0.002 # Standard friction
            return Langevin(atoms, timestep=timestep, temperature_K=current_temp, friction=friction)

def run_single_md_process(
    atoms: Atoms,
    expl_config: ExplorationConfig,
    model_name: str,
    device: str,
    progress_queue: Optional[Any] = None,
    timeout_seconds: int = 3600,
    stop_event: Optional[Any] = None
) -> Optional[List[Atoms]]:
    """
    Run a single MD trajectory.
    Instantiates its own calculator (Late Binding).
    Designed for ProcessPoolExecutor.
    """
    # Use timeout from config if available (passed as arg default, but let's override from config if present)
    # The caller passes timeout_seconds based on config usually, but let's ensure we use config.execution_timeout
    # if the caller didn't pass specific one?
    # Actually, the caller (MDExplorer.explore) should pass it.
    # But let's check expl_config inside here too just in case.
    if hasattr(expl_config, 'execution_timeout'):
         timeout_seconds = expl_config.execution_timeout

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
        calc = _get_calculator(model_name, device, expl_config.zbl_config)
        atoms.calc = calc
        
        # DEBUG: Verify SoftSphere is attached
        if hasattr(calc, 'calculators'):
             calc_names = [c.__class__.__name__ for c in calc.calculators]
             logger.info(f"Worker Calculator Stack: {calc_names}")
        else:
             logger.info(f"Worker Calculator: {calc.__class__.__name__}")
        
        radii = covalent_radii[atoms.numbers]
        sum_radii = radii[:, None] + radii[None, :]
        # Threshold to detect unphysical fusion/explosion.
        # r_Fe ~ 1.32. Sum=2.64.
        # 0.35 -> 0.92 A limit.
        overlap_threshold_ratio = 0.35
        thresholds = sum_radii * overlap_threshold_ratio
        np.fill_diagonal(thresholds, 0.0)

        def step_check():
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Process stopped by event.")

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
                mask = dists < thresholds
                if np.any(mask):
                     # Find first failure
                     indices = np.argwhere(mask)
                     # picking first one
                     i, j = indices[0]
                     d = dists[i, j]
                     limit = thresholds[i, j]
                     msg = f"Structure Exploded: Atomic overlap detected. Atoms {i}({atoms.symbols[i]})-{j}({atoms.symbols[j]}) dist={d:.4f} < limit={limit:.4f}"
                     logger.error(msg)
                     raise PhysicsViolationError(msg)
            except ValueError:
                pass
        
        # --- STABILITY FIX: Energy Minimization ---
        # Relax the structure to remove bad contacts/overlaps before heating
        logger.debug(f"Minimizing structure to resolve initial overlaps...")
        try:
            # Use FIRE as it's robust
            # Increase steps to 500 to ensure relaxation for surfaces
            opt = FIRE(atoms, logfile=None)
            # Run
            opt.run(fmax=0.2, steps=500) 
            pe = atoms.get_potential_energy()
            fmax = np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())
            logger.info(f"Minimization complete. Steps={opt.get_number_of_steps()}, PE={pe:.3f} eV, Fmax={fmax:.3f} eV/A")
            
            # Post-Minimization Safety Check
            step_check() # Reuse the safety check logic
            
        except Exception as e:
            logger.warning(f"Minimization failed/interrupted: {e}. Proceeding with caution.")

        # Initialize velocities at start temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_start)
        Stationary(atoms)
        
        # CRITICAL FIX: Wrap atoms into the box before MACE sees them
        # MACE requires atoms to be strictly within the unit cell.
        if any(atoms.pbc):
            # Ensure cell is upper triangular (canonical orientation)
            # MACE/TorchGeometric often require this for periodic boundaries.
            # We rotate the entire system (cell + atoms) to align v1 with x-axis, v2 in xy-plane.
            from ase.geometry import cellpar_to_cell
            cellpar = atoms.cell.cellpar()
            new_cell = cellpar_to_cell(cellpar)
            
            # Check if rotation is needed to minimize numerical drift if already close
            if not np.allclose(atoms.get_cell(), new_cell, atol=1e-6):
                atoms.set_cell(new_cell, scale_atoms=True)
            
            atoms.wrap()

        # Setup Dynamics via Factory
        dyn = _get_integrator(atoms, expl_config, temp_start)

        trajectory = []
        
        # Attach observer to record trajectory
        def append_traj():
            # Wrap atoms for saving, but DO NOT modify the simulation state!
            # Copy first, then wrap the copy.
            snap = atoms.copy()
            if any(snap.pbc):
                snap.wrap()
            trajectory.append(snap)
            
        # Record every snapshot_interval
        snap_interval = max(1, expl_config.snapshot_interval)
        dyn.attach(append_traj, interval=snap_interval)

        # Check N>1 for Langevin stability
        if len(atoms) <= 1:
             logger.warning("MD simulation requested for single atom. Skipping MD stability check that requires N>1.")
             if len(atoms) == 1:
                  logger.warning("Cannot run Langevin dynamics on a single atom due to ASE limitations. Returning initial structure.")
                  atoms.calc = None
                  return [atoms]



        def step_check():
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Process stopped by event.")

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
                # Element-specific check
                mask = dists < thresholds
                if np.any(mask):
                     # Find first failure
                     indices = np.argwhere(mask)
                     # picking first one
                     i, j = indices[0]
                     d = dists[i, j]
                     limit = thresholds[i, j]
                     msg = f"Structure Exploded: Atomic overlap detected. Atoms {i}({atoms.symbols[i]})-{j}({atoms.symbols[j]}) dist={d:.4f} < limit={limit:.4f}"
                     logger.error(msg)
                     from ase.io import write
                     write("./debug_atoms.xyz", atoms)
                     raise PhysicsViolationError(msg)
            except ValueError:
                pass

        step_check()

        # --- HYBRID LOOP ---
        swap_interval = mc_config.swap_interval if (mc_config and mc_config.enabled) else steps
        snap_interval = max(1, expl_config.snapshot_interval)

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
                if total_steps_done % snap_interval == 0:
                    logger.debug(f"MD Step {total_steps_done}/{steps}: Updated thermostat to {current_t:.1f} K")
            else:
                current_t = temp_start

            # Run MD Chunk
            # logger.info(f"DEBUG: Starting MD chunk {total_steps_done}-{total_steps_done+chunk}")
            try:
                dyn.run(chunk)
                if progress_queue is not None:
                     progress_queue.put(chunk)
            except RuntimeError as e:
                # Handle MACE PBC error or similar: "Some input data are greater than the size of the periodic box"
                if "periodic box" in str(e):
                    logger.warning("MACE PBC Error detected. Wrapping atoms and retrying.")
                    atoms.wrap()
                    # Retry once
                    try:
                        dyn.run(chunk)
                        if progress_queue is not None:
                             progress_queue.put(chunk)
                    except Exception as retry_err:
                        # If retry fails, fail hard
                        logger.error(f"Retry failed: {retry_err}")
                        raise retry_err
                else:
                    raise e
            except Exception as e:
                # ... (existing error handling) ...
                raise e

            # ...
            
            # logger.info(f"DEBUG: MD chunk done. Checking MC...")

            # MC Logic
            if mc_config and mc_config.enabled:
                mc_temp = mc_config.temp if mc_config.temp else current_t
                # logger.info(f"DEBUG: Performing MC Swap at step {total_steps_done}")
                accepted = perform_mc_swap(atoms, mc_config, mc_temp, calc)
                if accepted:
                     # Reset velocities to canonical distribution to dissipate local heat/shock
                     MaxwellBoltzmannDistribution(atoms, temperature_K=current_t)
                     Stationary(atoms)
                # logger.info(f"DEBUG: MC Swap done.")
            
            # Increment and Check
            total_steps_done += chunk
            step_check()

        atoms.calc = None
        return trajectory

    except InterruptedError:
        # Graceful exit
        return None
    except (PhysicsViolationError, TimeoutError, Exception) as e:
        # Catch all exceptions to dump debug structure locally
        # Since we are in a worker, we should write to a file that won't collide.
        try:
             # Dump to tempdir to prevent pollution
             debug_dir = tempfile.gettempdir()

             pid = os.getpid()
             filename = os.path.join(debug_dir, f"nnp_gen_worker_{pid}_failed.xyz")
             atoms.info['error'] = str(e)
             
             # DEBUG: Verify distance right before write
             try:
                 dcheck = atoms.get_all_distances(mic=True)
                 np.fill_diagonal(dcheck, np.inf)
                 min_d_check = np.min(dcheck)
                 logger.error(f"DUMP VERIFICATION: Min Dist in dumped structure = {min_d_check:.4f} A")
             except:
                 pass

             from ase.io import write
             # Dump history + final state
             history = trajectory + [atoms]
             write(filename, history)
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

from typing import List, Optional, Dict, Any, Tuple, Callable
# ...
class MDExplorer(IExplorer):
    def __init__(self, config: Any):
        self.config = config

    def explore(self, seeds: List[Atoms], n_workers: Optional[int] = None, callback: Optional[Callable[[List[Atoms]], None]] = None) -> List[Atoms]:
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
        
        logger.info(f"DEBUG: device='{device}', n_workers={n_workers}")

        # Pre-flight check
        try:
            logger.info(f"Verifying calculator '{model_name}' availability...")
            _get_calculator(model_name, "cpu", getattr(expl, 'zbl_config', None)) 
        except Exception as e:
            logger.error(f"Calculator check failed: {e}")
            raise RuntimeError(f"Cannot start exploration: Calculator '{model_name}' failed to initialize. Error: {e}")

        # Use Manager for Queue and Stop Event
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        stop_event = manager.Event()

        logger.info(f"Starting MD Exploration with {n_workers} processes. Method: {expl.method} on {device}")
        
        total_steps = len(seeds) * expl.steps
        
        # Use ProcessPoolExecutor
        # Use ProcessPoolExecutor or Direct
        if n_workers == 1:
            logger.info("Running in Single Process Mode (Direct) to avoid CUDA/MP hangs.")
            # Direct Execution Loop
            # We still use the progress bar and structure
            with tqdm(total=total_steps, desc="MD Exploration Steps", unit="step") as pbar:
                for seed in seeds:
                    try:
                        # Create a simple queue wrapper or pass None to manual update?
                        # run_single_md_process puts into queue. 
                        # We can pass a dummy queue and poll it, or just not use queue for pbar in direct mode?
                        # Better: ProcessQueue is for MP. In direct mode, we can pass a callback or just update pbar after return?
                        # But we want progressive bars.
                        # Let's use a simple Queue and a thread to drain it? Or just pass pbar?
                        # run_single_md_process expects a queue.put(chunk).
                        
                        # Just use a standard queue
                        import queue
                        q = queue.Queue()
                        
                        # Issue: run_single is blocking. We can't drain queue while running unless we thread.
                        # BUT, we can just let it run and update pbar at the end? 
                        # No, user wants to see progress.
                        # Strategy: Pass a "Dummy" queue that updates pbar directly? 
                        # Queue needs .put().
                        
                        class PBarQueue:
                            def put(self, n):
                                pbar.update(n)
                        
                        # Use execution_timeout from config if available
                        timeout = expl.execution_timeout if hasattr(expl, 'execution_timeout') else 3600

                        traj = run_single_md_process(
                            seed, expl, model_name, device, 
                            progress_queue=PBarQueue(), 
                            timeout_seconds=timeout,
                            stop_event=stop_event
                        )
                        
                        if traj:
                            results.extend(traj)
                            if callback:
                                callback(traj)
                                
                    except Exception as e:
                        logger.error(f"Direct Execution failed: {e}")
                        raise e
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                # key: future, value: seed (for debug if needed)
                futures_map = {}
                # Use execution_timeout from config if available
                timeout = expl.execution_timeout if hasattr(expl, 'execution_timeout') else 3600

                for seed in seeds:
                    f = executor.submit(
                        run_single_md_process, 
                        seed, 
                        expl, 
                        model_name, 
                        device, 
                        progress_queue, 
                        timeout,
                        stop_event # Pass event
                    )
                    futures_map[f] = seed
    
                # Monitor Loop
                with tqdm(total=total_steps, desc="MD Exploration Steps", unit="step") as pbar:
                    unfinished = list(futures_map.keys())
                    
                    while unfinished:
                        # Drain Queue
                        try:
                            while not progress_queue.empty():
                                # get_nowait is nicer but .empty() check + get() is okay here since we control writers
                                n = progress_queue.get_nowait()
                                pbar.update(n)
                        except queue.Empty:
                            pass
                        except Exception:
                            pass
                        
                        # Check futures
                        done, not_done_sets = concurrent.futures.wait(
                            unfinished, 
                            timeout=0.1, 
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        
                        for f in done:
                            try:
                                traj = f.result()
                                if traj is not None:
                                    results.extend(traj)
                                    if callback:
                                        try:
                                            callback(traj)
                                        except Exception as cb_err:
                                            logger.error(f"Callback failed: {cb_err}")
                            except Exception as e:
                                logger.error(f"Worker failed: {e}")
                                # Stop others
                                stop_event.set()
                                # Cancel others (futures)
                                for uf in unfinished:
                                    uf.cancel()
                                raise e
                            
                            unfinished.remove(f)

        return results
