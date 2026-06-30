from operator import call
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from loguru import logger
from .baseoptimizer import BaseVQEOptimizer
class Cobyla(BaseVQEOptimizer):
    """Wrapper for the COBYLA optimizer with spam-proof convergence handles."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.method = "COBYLA" 
        self.rhobeg = config.get("lr", 0.5) 

    def minimize(self, circuit, initial_params):
        history = {'params': [], 'energies': [], 'steps': []}
        state = {'evals': 0, 'iterations': 0, 'converged': False}

        def cost_function(params):
            energy = float(circuit(params))
            history['params'].append(params.copy())
            history['energies'].append(energy)
            history['steps'].append(state['evals'])
            state['evals'] += 1
            return energy

        def callback(_current_best_params):
            state['iterations'] += 1
            iteration = state['iterations'] 
            latest_energy = history['energies'][-1] if history['energies'] else 0.0

            if iteration % self.print_every == 0:
                logger.info(f"Iterations {iteration:4d}: Energy = {latest_energy:.6f} Ha")

            if iteration >= self.plateau_window:
                is_converged, _, _, _ = self._check_plateau(history)
                if is_converged:
                    logger.success(f"CONVERGED at evaluation {iteration}!")
                    state['converged'] = True
                    raise RuntimeError("EarlyConvergenceTriggered")
            

        res = None
        try:
            res = scipy_minimize(
                        cost_function, 
                        initial_params, 
                        method=self.method,
                        callback=callback,
                        options={
                            "maxiter": self.max_iterations,
                            "rhobeg": self.rhobeg  
                        }
                    )
        except RuntimeError as e:
            print("EARLY")
            if str(e) != "EarlyConvergenceTriggered":
                raise e
        
        if res is not None and hasattr(res, 'x') and res.x is not None:
            final_params = res.x
            actual_convergence = True if res.status == 0 or state['converged'] else False
        else:
            final_params = history['params'][-1] if history['params'] else initial_params
            actual_convergence = state['converged']

        return state['iterations'], history, final_params, actual_convergence
