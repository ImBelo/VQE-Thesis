# noise_models/thermal.py
import pennylane as qml
from .base import BaseNoiseModel

class ThermalNoiseModel(BaseNoiseModel):
    """Handles multi-parameter hardware-bound relaxation insertions."""
    
    def __init__(self, t1, t2, tg, qubits=None):
        super().__init__(qubits)
        self.t1 = t1
        self.t2 = t2
        self.tg = tg

    def wrap_qnode(self, base_qnode):
        @qml.BooleanFn
        def cond(op):
            return len(op.wires) == 1 and self._get_base_condition(op)

        def _thermal_error(op, **metadata):

            qml.ThermalRelaxationError(pe=0.0, t1=self.t1, t2=self.t2, tg=self.tg, wires=op.wires)
            
        return qml.add_noise(base_qnode, qml.NoiseModel({cond: _thermal_error}))
