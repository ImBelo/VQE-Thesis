from abc import ABC, abstractmethod
import pennylane
import pennylane as qml 

from typing import Optional
import numpy as np 

class BaseAnsatz(ABC):
    """Abstract base class for all variational ansatzes"""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
    
    @abstractmethod
    def __call__(self, params: np.ndarray, n_qubits: int, h_ref: Optional[np.ndarray] = None) -> None:
        pass
    
    @abstractmethod
    def get_num_params(self) -> int:
        pass

class AnsatzFactory:
    """Creates ansatz functions that work inside @qnode"""
    
    @staticmethod
    def create(config: dict):
        ansatz_type = config.get("type", "hardware_efficient")
        n_qubits = config.get("n_qubits",0)
        
        from .he_ansatz import HardwareEfficientCircuit
        from .uccsd_ansatz import UCCSDCircuit
        if ansatz_type == "uccsd":
            return UCCSDCircuit(
                n_layers=config.get("layers", 1),
                n_electrons=config.get("n_electrons"),
                n_qubits = n_qubits,
                
            )
        elif ansatz_type == "hardware_efficient":
            return HardwareEfficientCircuit(
                n_layers=config.get("layers", 2),
                entanglement=config.get("entanglement", "linear"),
                n_qubits = n_qubits,
               
            )



