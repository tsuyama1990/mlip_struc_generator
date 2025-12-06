import time
from typing import Dict, Any
from .simulation_interface import SimulationInterface

class MockSimulation(SimulationInterface):
    """
    Mock implementation of MD simulation.
    Simulates a delay but performs no actual physics.
    """

    def run(self, config: Dict[str, Any]) -> None:
        """
        Simulate a 5-second delay to mimic MD computation.
        """
        print(f"Starting mock simulation with config: {config}")
        # Mimic processing steps
        for i in range(5):
            time.sleep(0.2) # Sleep for 0.2 second 5 times (total 1s) to simulate work
            print(f"Simulation step {i+1}/5 complete...")
        print("Mock simulation finished.")
