import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Literal


class BaseVQEOptimizer(ABC):
    """Abstract base class for all VQE optimizers"""
    
    def __init__(self, config: dict):
        self.name = config.get("type", "adam").lower()
        self.gradient_method = config.get("gradient_method",None)
        self.max_iterations = config.get("max_iterations", 500)
        self.convergence_threshold = config.get("convergence_tolerance", 1e-4)
        self.plateau_window = config.get("plateau_window", 5)
        self.plateau_tolerance = config.get("plateau_tolerance", 1e-4)
        self.print_every = config.get("print_every", 20)

    def _check_plateau(self, history: dict) -> tuple:
        """Shared plateau detection logic."""
        energies = history.get('energies', [])

        if len(energies) < self.plateau_window:
            return False, float('inf'), float('inf'), float('inf')

        recent = energies[-self.plateau_window:]
        max_diff = max(recent) - min(recent)

        if max_diff >= self.convergence_threshold:
            return False, max_diff, np.std(recent), float('inf')

        x = np.arange(self.plateau_window)
        slope = np.polyfit(x, recent, 1)[0]

        std_dev = np.std(recent) 

        converged = abs(slope) < self.plateau_tolerance
    
        return converged, max_diff, std_dev, slope


    @abstractmethod
    def minimize(self, circuit, initial_params) -> Tuple[int, Dict[str, List[Any]], Any, Union[Any, Literal[False]]]:
        """Every subclass must implement this execution loop."""
        pass
