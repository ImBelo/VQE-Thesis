from __future__ import annotations

import multiprocessing
from typing import Any, Dict, Iterable, List, Set, Tuple

from loguru import logger

from utils.config.config import EXPERIMENT_KEYS


Combination = Tuple[
    Dict[str, Any],   # molecule
    Dict[str, Any],   # ansatz
    Dict[str, Any],   # optimizer
    Dict[str, Any],   # noise
]


def build_execution_tasks(
    all_combinations: Iterable[Combination],
    completed_runs: Set[Tuple[Any, ...]],
    num_runs: int,
) -> List[Dict[str, Any]]:
    """
    Filter out combinations already stored in the database.

    Returns a list of experiment_config dicts ready to be dispatched to
    workers.
    """
    tasks_to_run: List[Dict[str, Any]] = []

    for current_mol, current_ansatz, current_opt, current_noise in all_combinations:
        for run_id in range(num_runs):
            experiment_config: Dict[str, Any] = {
                "run_id": run_id,
                "mol_name": current_mol["name"],
                "mol_basis": current_mol["basis"],
                "mapping": current_mol["mapping"],
                "ansatz_type": current_ansatz["type"],
                "ansatz_layers": current_ansatz["layers"],
                "ansatz_entanglement": current_ansatz["entanglement"],
                "opt_type": current_opt["type"],
                "opt_lr": current_opt["lr"],
                "gradient_method": current_opt["gradient_method"],
                "noise_model": current_noise["model"],
                "noise_strength": current_noise["strength"],

                "molecule": current_mol,
                "ansatz": current_ansatz,
                "optimizer": current_opt,
                "noise": current_noise,
            }

            experiment_fingerprint: Tuple[Any, ...] = tuple(
                experiment_config[key] for key in EXPERIMENT_KEYS
            )
            if experiment_fingerprint in completed_runs:
                continue

            tasks_to_run.append(experiment_config)

    return tasks_to_run


def allocate_compute_workers(use_gpu: bool = False) -> int:
    """Return the number of parallel workers to use based on the backend."""
    if use_gpu:
        logger.info("Running in GPU mode")
        return 1

    num_workers: int = max(1, multiprocessing.cpu_count() - 1)
    logger.info(
        f"Running in CPU mode: Utilizing {num_workers} parallel CPU processes."
    )
    return num_workers
