from __future__ import annotations

from typing import Any, Callable, Dict, List, Union

import numpy as np
from loguru import logger
from scipy.optimize import minimize as scipy_minimize

from .baseoptimizer import BaseVQEOptimizer


class Cobyla(BaseVQEOptimizer):
    """Wrapper for the COBYLA optimizer with plateau-based early stopping."""

    method: str
    rhobeg: float

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.method: str = "COBYLA"
        self.rhobeg: float = config.get("lr", 0.5)

    def minimize(
        self,
        circuit: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
    ) -> tuple[int, dict[str, list[Any]], np.ndarray, bool]:
        history: dict[str, list[Any]] = {"params": [], "energies": [], "steps": []}
        state: dict[str, Union[int, bool]] = {
            "evals": 0,
            "iterations": 0,
            "converged": False,
        }

        def cost_function(params: np.ndarray) -> float:
            energy = float(circuit(params))
            history["params"].append(params.copy())
            history["energies"].append(energy)
            history["steps"].append(state["evals"])
            state["evals"] = int(state["evals"]) + 1
            return energy

        def callback(_current_best_params: np.ndarray) -> None:
            state["iterations"] = int(state["iterations"]) + 1
            iteration = int(state["iterations"])
            latest_energy = (
                history["energies"][-1] if history["energies"] else 0.0
            )

            if iteration % self.print_every == 0:
                logger.info(
                    f"Iterations {iteration:4d}: Energy = {latest_energy:.6f} Ha"
                )

            if iteration >= self.plateau_window:
                is_converged, _, _, _ = self._check_plateau(history)
                if is_converged:
                    logger.success(f"CONVERGED at evaluation {iteration}!")
                    state["converged"] = True
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
                    "rhobeg": self.rhobeg,
                },
            )
        except RuntimeError as e:
            if str(e) != "EarlyConvergenceTriggered":
                raise

        if res is not None and hasattr(res, "x") and res.x is not None:
            final_params: np.ndarray = res.x
            actual_convergence: bool = (
                True if res.status == 0 or state["converged"] else False
            )
        else:
            final_params = (
                history["params"][-1] if history["params"] else initial_params
            )
            actual_convergence = bool(state["converged"])

        return int(state["iterations"]), history, final_params, actual_convergence
