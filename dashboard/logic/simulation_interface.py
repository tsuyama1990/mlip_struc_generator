from abc import ABC, abstractmethod
from typing import Dict, Any

class SimulationInterface(ABC):
    """
    Abstract interface for running MD simulations.
    """

    @abstractmethod
    def run(self, config: Dict[str, Any]) -> None:
        """
        Run the simulation with the given configuration.
        """
        pass
