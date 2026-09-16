from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pennylane as qml


class BaseAnsatz(ABC):
    """Abstract base class for all variational ansatzes."""

    n_qubits: int

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits: int = n_qubits

    @abstractmethod
    def __call__(
        self,
        params: np.ndarray,
        n_qubits: int,
        h_ref: Optional[np.ndarray] = None,
    ) -> None:
        """Apply the ansatz circuit to the active QNode."""
        ...

    @abstractmethod
    def get_num_params(self) -> int:
        """Return the number of trainable parameters in this ansatz."""
        ...


class AnsatzFactory:
    """Creates ansatz instances from a configuration dict."""

    @staticmethod
    def create(config: dict[str, Any]) -> BaseAnsatz:
        # Imported locally to avoid a circular import at module load:
        # he_ansatz and uccsd_ansatz both import BaseAnsatz from this module.
        from .he_ansatz import HardwareEfficientCircuit
        from .uccsd_ansatz import UCCSDCircuit

        ansatz_type: str = config.get("type", "hardware_efficient")
        n_qubits: int = config.get("n_qubits", 0)

        if ansatz_type == "uccsd":
            return UCCSDCircuit(
                n_layers=config.get("layers", 1),
                n_electrons=config.get("n_electrons"),
                n_qubits=n_qubits,
            )
        elif ansatz_type == "hardware_efficient":
            return HardwareEfficientCircuit(
                n_layers=config.get("layers", 2),
                entanglement=config.get("entanglement", "linear"),
                n_qubits=n_qubits,
            )
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type!r}")
