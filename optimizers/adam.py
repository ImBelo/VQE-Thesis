from .optimizers import BaseVQEOptimizer
import pennylane as qml

from loguru import logger

class Adam(BaseVQEOptimizer):
    """Handles all native PennyLane step_and_cost optimizers."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.method = "adam"
        lr = config.get("lr", 0.01)
        beta1 = config.get("beta1", 0.9)
        beta2 = config.get("beta2", 0.999)
        self.opt = qml.AdamOptimizer(stepsize=lr, beta1=beta1, beta2=beta2)
        
    def minimize(self, circuit, initial_params):
        params = initial_params
        history = {'params': [], 'energies': [], 'steps': []}
        converged = False
        step = 0

        logger.info(f"Running optimization with PennyLane {self.opt.__class__.__name__}...")

        for iteration in range(self.max_iterations):
            step = iteration
            params, energy = self.opt.step_and_cost(circuit, params)
            
            history['params'].append(params.copy())
            history['energies'].append(float(energy))
            history['steps'].append(iteration)
            
            if iteration % self.print_every == 0:
                logger.info(f"  Iteration {iteration:4d}: Energy = {energy:.6f} Ha")
            if iteration >= self.plateau_window:
                converged, max_diff, std_dev, slope = self._check_plateau(history)
                if converged:
                    logger.success(f"CONVERGED at iteration {iteration}!")
                    break
                    
        return step, history, params, converged
