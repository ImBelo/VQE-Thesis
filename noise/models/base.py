from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

import pennylane as qml


class BaseNoiseModel(ABC):
    """Foundational abstract class for all VQE noise models."""

    qubits: Optional[Sequence[int]]

    def __init__(self, qubits: Optional[Sequence[int]] = None) -> None:
        self.qubits: Optional[Sequence[int]] = qubits

    def _get_base_condition(self, op: qml.operation.Operator) -> bool:
        """Return True if the operation acts on any of the target wires."""
        if self.qubits is not None:
            return qml.noise.wires_in(self.qubits)(op)
        return True

    @abstractmethod
    def wrap_qnode(self, base_qnode: Callable) -> Callable:
        """
        Wrap a QNode with a noise model.

        Subclasses must implement this to inject noise channels.
        """
        ...
