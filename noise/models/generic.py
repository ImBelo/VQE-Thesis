# noise_models/generic.py
import pennylane as qml
from .base import BaseNoiseModel

class GenericNoiseModel(BaseNoiseModel):
    """Handles standard single-parameter channel insertions."""
    
    def __init__(self, channel_func, strength, qubits=None):
        super().__init__(qubits)
        self.channel_func = channel_func
        self.strength = strength

    def wrap_qnode(self, base_qnode):
        cond = qml.BooleanFn(self._get_base_condition)
        noise_fn = qml.noise.partial_wires(self.channel_func, self.strength)
        return qml.add_noise(base_qnode, qml.NoiseModel({cond: noise_fn}))
