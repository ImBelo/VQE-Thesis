from abc import ABC, abstractmethod
import pennylane as qml 
from pennylane.qchem import excitations, excitations_to_wires
from .base_ansatz import BaseAnsatz
import numpy as np

from typing import Optional

class UCCSDCircuit(BaseAnsatz):
    """UCCSD circuit - callable inside @qnode"""
    
    def __init__(self, n_electrons, n_qubits,n_layers = 2):
        super().__init__(n_qubits)
        self.n_layers = n_layers
        self.n_electrons = n_electrons
        self.singles, self.doubles, self.s_wires, self.d_wires = self.get_uccsd_excitations(
            n_qubits, electrons=self.n_electrons
        )


    def __call__(self, params: np.ndarray, h_ref: Optional[np.ndarray] = None) -> None:
        """
        Returns a QNode that, given `params`, computes
        <UCCSD(params)| Hamiltonian |UCCSD(params)>.
        """
        
        if h_ref is None:
            raise ValueError("UCCSDAnsatz requires h_ref (initial state)")


        params_2d = params.reshape(self.n_layers, -1)
        qml.UCCSD(
            weights=params_2d,
            wires=range(self.n_qubits),
            s_wires=self.s_wires,
            d_wires=self.d_wires,
            init_state=h_ref,
            n_repeats=self.n_layers
        )

    
    def get_num_params(self) -> int:
        """Calculate number of parameters needed"""
        return (len(self.singles) + len(self.doubles)) * self.n_layers

    def get_uccsd_excitations(self,qubits, electrons):
        """Compute singles and doubles for UCCSD and convert to wire format."""
        singles, doubles = excitations(electrons, qubits)
        s_wires, d_wires = excitations_to_wires(singles, doubles)
        return singles, doubles, s_wires, d_wires


