import pennylane as qml

class BaseNoiseModel:
    """The foundational class for all VQE noise models."""
    
    def __init__(self, qubits=None):
        self.qubits = qubits

    def _get_base_condition(self, op) -> bool:
        """Evaluates if the operation matches target wires."""
        if self.qubits is not None:
            return qml.noise.wires_in(self.qubits)(op)
        return True

    def wrap_qnode(self, base_qnode):
        raise NotImplementedError
