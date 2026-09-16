from __future__ import annotations

from typing import Callable, Optional, Sequence

import pennylane as qml

from .base import BaseNoiseModel


class GenericNoiseModel(BaseNoiseModel):
    """Noise model for single-parameter channels (e.g. depolarizing)."""

    channel_func: Callable
    strength: float

    def __init__(
        self,
        channel_func: Callable,
        strength: float,
        qubits: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(qubits)
        self.channel_func: Callable = channel_func
        self.strength: float = strength

    def wrap_qnode(self, base_qnode: Callable) -> Callable:
        cond = qml.BooleanFn(self._get_base_condition)
        noise_fn = qml.noise.partial_wires(self.channel_func, self.strength)
        return qml.add_noise(base_qnode, qml.NoiseModel({cond: noise_fn}))
