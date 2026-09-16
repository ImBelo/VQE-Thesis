from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pennylane as qml

from .base_ansatz import BaseAnsatz


Entanglement = Literal["full", "linear", "circular"]


class HardwareEfficientCircuit(BaseAnsatz):
    """Hardware-efficient ansatz, callable inside a @qml.qnode context."""

    n_layers: int
    entanglement: Entanglement

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        entanglement: Entanglement = "linear",
    ) -> None:
        super().__init__(n_qubits)
        self.n_layers: int = n_layers
        self.entanglement: Entanglement = entanglement

    def __call__(
        self,
        params: np.ndarray,
        h_ref: Optional[np.ndarray] = None,
    ) -> None:
        """Apply the ansatz circuit inside an active @qml.qnode."""
        params_2d: np.ndarray = params.reshape((self.n_layers, self.n_qubits))

        if h_ref is not None:
            qml.BasisState(h_ref, wires=range(self.n_qubits))

        for layer in range(self.n_layers):

            # Rotation layer
            for i in range(self.n_qubits):
                qml.RY(params_2d[layer, i], wires=i)

            # Entanglement layer
            if self.entanglement == "full":
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[i, j])

            elif self.entanglement == "linear":
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            elif self.entanglement == "circular":
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

    def get_num_params(self) -> int:
        """One rotation parameter per qubit per layer."""
        return self.n_qubits * self.n_layers
