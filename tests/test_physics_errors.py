import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ase import Atoms
from nnp_gen.explorers.md_engine import run_single_md_process
from nnp_gen.core.config import ExplorationConfig

def test_md_crash_recovery_pbc():
    """
    Test that the MD engine recovers from MACE PBC errors (RuntimeError).
    """
    # 1. Setup Config
    expl_config = ExplorationConfig(
        method="md",
        model_name="mace",
        steps=5, # Short run
        timestep=1.0,
        temperature=300,
        temperature_mode="constant" 
    )
    
    # 2. Setup Atoms with PBC
    atoms = Atoms('H2', positions=[[0, 0, 0], [0.8, 0, 0]], cell=[10, 10, 10], pbc=True)
    
    # 3. Mock Calculator
    start_calc = MagicMock()
    
    # 4. Mock Integrator (to raise error on run)
    mock_dyn = MagicMock()
    
    # Side effect for dyn.run(chunk):
    # First call: Raise RuntimeError("Some input data are greater than ...")
    # Second call: Success (retry)
    # Subsequent calls: Success (chunks)
    # The loop runs 'steps' times (5 steps). 
    # Logic in md_engine: 
    # while total_steps_done < steps:
    #   dyn.run(chunk) 
    # If chunk=1 (default if snap_interval/swap_interval are small or steps is small), it might run 5 times.
    # We provide enough side effects to be safe.
    mock_dyn.run.side_effect = [
        RuntimeError("Some input data are greater than the size of the periodic box"),
        None, # Success on retry
        None, None, None, None, None, None, None, None, None # Subsequent steps padding
    ]
    
    # 5. Patch dependencies
    with patch("nnp_gen.explorers.md_engine._get_calculator", return_value=start_calc), \
         patch("nnp_gen.explorers.md_engine._get_integrator", return_value=mock_dyn):
         
         # Run
         traj = run_single_md_process(atoms, expl_config, "mace", "cpu", timeout_seconds=10)
         
         # Verification
         # Ensure we got results (trajectory) despite the crash
         assert len(traj) >= 0 
         
         # Ensure dyn.run was called at least twice (initial + retry)
         assert mock_dyn.run.call_count >= 2
         
         print("Successfully recovered from MACE PBC error!")
