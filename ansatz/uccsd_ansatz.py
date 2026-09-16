from __future__ import annotations

from typing import Optional

import numpy as np
import pennylane as qml
from pennylane.qchem import excitations, excitations_to_wires

from .base_ansatz import BaseAnsatz


class UCCSDCircuit(BaseAnsatz):
    """UCCSD ansatz, callable inside a @qml.qnode context."""

    n_layers: int
    n_electrons: int
    singles: list[tuple[int, int]]
    doubles: list[tuple[tuple[int, int], tuple[int, int]]]
    s_wires: list[list[int]]
    d_wires: list[list[list[int]]]

    def __init__(
        self,
        n_electrons: int,
        n_qubits: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__(n_qubits)
        self.n_layers: int = n_layers
        self.n_electrons: int = n_electrons

        (
            self.singles,
            self.doubles,
            self.s_wires,
            self.d_wires,
        ) = self.get_uccsd_excitations(n_qubits, electrons=self.n_electrons)

    def __call__(
        self,
        params: np.ndarray,
        h_ref: Optional[np.ndarray] = None,
    ) -> None:
        """Apply the UCCSD circuit inside an active @qml.qnode."""
        if h_ref is None:
            raise ValueError("UCCSDCircuit requires h_ref (initial state)")

        params_2d: np.ndarray = params.reshape(self.n_layers, -1)

        qml.UCCSD(
            weights=params_2d,
            wires=range(self.n_qubits),
            s_wires=self.s_wires,
            d_wires=self.d_wires,
            init_state=h_ref,
            n_repeats=self.n_layers,
        )

    def get_num_params(self) -> int:
        """Number of trainable parameters: (singles + doubles) * layers."""
        return (len(self.singles) + len(self.doubles)) * self.n_layers

    def get_uccsd_excitations(
        self,
        qubits: int,
        electrons: int,
    ) -> tuple[
        list[tuple[int, int]],
        list[tuple[tuple[int, int], tuple[int, int]]],
        list[list[int]],
        list[list[list[int]]],
    ]:
        """Compute singles and doubles for UCCSD and convert to wire format."""
        singles, doubles = excitations(electrons, qubits)
        s_wires, d_wires = excitations_to_wires(singles, doubles)
        return singles, doubles, s_wires, d_wires
