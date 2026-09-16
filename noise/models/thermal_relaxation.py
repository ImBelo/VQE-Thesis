from __future__ import annotations

from typing import Callable, Optional, Sequence

import pennylane as qml
from pennylane.operation import Operator

from .base import BaseNoiseModel


class ThermalNoiseModel(BaseNoiseModel):
    """Noise model for hardware-bound thermal relaxation errors."""

    t1: float
    t2: float
    tg: float

    def __init__(
        self,
        t1: float,
        t2: float,
        tg: float,
        qubits: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(qubits)
        self.t1: float = t1
        self.t2: float = t2
        self.tg: float = tg

    def wrap_qnode(self, base_qnode: Callable) -> Callable:
        @qml.BooleanFn
        def cond(op: Operator) -> bool:
            return len(op.wires) == 1 and self._get_base_condition(op)

        def _thermal_error(op: Operator, **metadata: object) -> None:
            qml.ThermalRelaxationError(
                pe=0.0,
                t1=self.t1,
                t2=self.t2,
                tg=self.tg,
                wires=op.wires,
            )

        return qml.add_noise(base_qnode, qml.NoiseModel({cond: _thermal_error}))
