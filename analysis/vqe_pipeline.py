from __future__ import annotations

from typing import Optional

import numpy as np
import pennylane as qml
import time                                       # added
from pennylane import numpy as pnp
from loguru import logger
from noise.models.factory import NoiseFactory
from ansatz.base_ansatz import AnsatzFactory, BaseAnsatz
from optimizers.baseoptimizer import BaseVQEOptimizer
from noise.models.base import BaseNoiseModel
from optimizers.optimizers import OptimizerFactory 
from core.molecules import (
    MoleculeConfig,
    MoleculeFactory,
    QubitMappedSystem,
)

class VQEPipeline:
    """Executes VQE with different components"""
    def __init__(
        self,
        hamiltonian: qml.Hamiltonian,
        ansatz: BaseAnsatz,
        optimizer: BaseVQEOptimizer,
        noise_model: Optional[BaseNoiseModel],
        n_qubits: int,
        n_electrons: int,
        h_ref: np.ndarray,
    ):
        self.hamiltonian: qml.Hamiltonian = hamiltonian
        self.ansatz: BaseAnsatz = ansatz
        self.optimizer: BaseVQEOptimizer = optimizer
        self.noise_model: Optional[BaseNoiseModel] = noise_model
        self.n_qubits: int = n_qubits
        self.n_electrons: int = n_electrons
        self.h_ref: np.ndarray = h_ref
        
        if self.noise_model is not None:
            # Noise models usually require density matrices (default.mixed)
            self.device = qml.device("default.mixed", wires=n_qubits)
            logger.info(f"Using CPU noise backend: default.mixed ({n_qubits} qubits)")
            
        elif n_qubits >= 10:
            try:
                self.device = qml.device("lightning.gpu", wires=n_qubits)
                logger.info(f"Using high-performance GPU backend: lightning.gpu ({n_qubits} qubits)")
            except Exception:
                self.device = qml.device("lightning.qubit", wires=n_qubits)
                logger.error("GPU device failed initialization. Falling back to CPU backend: lightning.qubit")
                
        else:
            self.device = qml.device("default.qubit", wires=n_qubits)
            logger.info(f"Using optimized CPU backend: lightning.qubit ({n_qubits} qubits)")    

    def create_qnode(self, ansatz_builder):
        """Create the QNode with the ansatz"""

        optimizer_name = ""
        if self.noise_model is not None:
            diff_method = "parameter-shift"
        else:
            diff_method = "adjoint"
            optimizer_name = getattr(self.optimizer, "name", str(self.optimizer)).lower()
        if "cobyla" in optimizer_name:
            diff_method = None

        dev = self.device

        # Calculate the expectation value of hamiltonian given our circuit U(theta)
        # <U^dagger(theta)|H_pauli|U(theta)>
        @qml.qnode(dev, diff_method=diff_method)
        def base_circuit(params,h_ref):
            ansatz_builder(params,h_ref)
            return qml.expval(self.hamiltonian)

        # If theres noise make each gate of the circuit noisy
        if self.noise_model is not None:
            base_circuit = self.noise_model.wrap_qnode(base_circuit)

        # Optimizers call base circuit(params) leaving h_ref out so wrap it into a lambda
        return lambda params: base_circuit(params, self.h_ref)
    
    def run(self):
        """Run VQE optimization"""
        # number of params of parametrized gates
        n_params = self.ansatz.get_num_params()
        # ansatz
        circuit = self.create_qnode(self.ansatz)
        
        raw_params = np.random.default_rng().random(n_params) * 0.01 
        params = pnp.array(raw_params, requires_grad=True)
        
        start_time = time.perf_counter()

        step, history, final_params, converged = self.optimizer.minimize(circuit, params)

        for key in history:
            history[key] = np.array(history[key])
            
        execution_time = time.perf_counter() - start_time    
            
        return step, history, final_params, converged, execution_time  

    @classmethod
    def from_config(
        cls,
        mol_config: MoleculeConfig,
        ansatz_config: dict,
        opt_config: dict,
        noise_config: Optional[dict],
    ) -> VQEPipeline:
        """Orchestrates the factories to build the pipeline from config"""


        physical_mol = MoleculeFactory.create_physical(
            name=mol_config["name"], 
            basis=mol_config["basis"]
        )
        qubit_mol = MoleculeFactory.create_mapped(
            physical=physical_mol,
            mapping=mol_config["mapping"]
        )

        ansatz_parameters = {
            **ansatz_config, 
            "n_electrons": physical_mol.n_electrons,  
            "n_qubits": qubit_mol.n_qubits
        }
        ansatz = AnsatzFactory.create(ansatz_parameters)

        optimizer = OptimizerFactory.create(opt_config)

        noise_model = NoiseFactory.create(noise_config)

        return cls(
            hamiltonian=qubit_mol.hamiltonian,
            ansatz=ansatz,
            optimizer=optimizer,
            noise_model = noise_model,
            n_qubits=qubit_mol.n_qubits,
            n_electrons=physical_mol.n_electrons,
            h_ref=qubit_mol.hf_state
        )
