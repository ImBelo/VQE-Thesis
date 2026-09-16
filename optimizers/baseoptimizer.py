from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import numpy as np


class BaseVQEOptimizer(ABC):
    """Abstract base class for all VQE optimizers."""

    name: str
    gradient_method: str | None
    max_iterations: int
    convergence_threshold: float
    plateau_window: int
    plateau_tolerance: float
    print_every: int

    def __init__(self, config: dict[str, Any]) -> None:
        self.name: str = config.get("type", "adam").lower()
        self.gradient_method: str | None = config.get("gradient_method", None)
        self.max_iterations: int = config.get("max_iterations", 500)
        self.convergence_threshold: float = config.get("convergence_tolerance", 1e-4)
        self.plateau_window: int = config.get("plateau_window", 5)
        self.plateau_tolerance: float = config.get("plateau_tolerance", 1e-4)
        self.print_every: int = config.get("print_every", 20)

    def _check_plateau(self, history):
        window = history['energies'][-self.plateau_window:]
        if len(window) < self.plateau_window:
            return False, None, None, None
        diffs = [abs(window[i+1] - window[i]) for i in range(len(window)-1)]
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        converged = (max_diff < self.plateau_tolerance) and (mean_diff < self.plateau_tolerance / 2)
        return converged, max_diff, None, None

    @abstractmethod
    def minimize(
        self,
        circuit: Callable,
        initial_params: np.ndarray,
    ) -> Tuple[
        int,
        Dict[str, List[Any]],
        Any,
        Union[Any, Literal[False]],
    ]:
        """Every subclass must implement this execution loop."""
        ...
