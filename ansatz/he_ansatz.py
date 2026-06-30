from abc import ABC, abstractmethod
import pennylane
import pennylane as qml 
from .base_ansatz import BaseAnsatz

from typing import Optional
from typing import Optional, List
import numpy as np
class HardwareEfficientCircuit(BaseAnsatz):
    """Hardware efficient circuit - callable inside @qnode"""
    

    def __init__(self, n_qubits, n_layers=2, entanglement="full",):
        super().__init__(n_qubits)
        self.n_layers = n_layers
        self.entanglement = entanglement
    
    def __call__(self, params: np.ndarray, h_ref: Optional[np.ndarray] = None) -> None:
        """This runs INSIDE the @qnode context"""
        params_2d = params.reshape((self.n_layers, self.n_qubits))  
        if h_ref is not None:
            qml.BasisState(h_ref, wires=range(self.n_qubits))  

        for layer in range(self.n_layers):

            # Rotation Layer
            for i in range(self.n_qubits):
                qml.RY(params_2d[layer, i], wires=i)

            # Entanglement Layer
            if self.entanglement == "full":
                for i in range(self.n_qubits):
                    for j in range(i+1, self.n_qubits):
                        qml.CNOT(wires=[i, j])

            elif self.entanglement == "linear":
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])

            elif self.entanglement == "circular":
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i+1) % self.n_qubits])
    
    def get_num_params(self):
        """1 rotation parameters per qubit per layer"""
        return self.n_qubits * self.n_layers
