# utils/task_manager.py
import multiprocessing
from utils.config.config import EXPERIMENT_KEYS
from loguru import logger


def build_execution_tasks(all_combinations, completed_runs, num_runs):
    """Filters out combinations that have already been written to the storage layer."""
    tasks_to_run = []
    for current_mol, current_ansatz, current_opt, current_noise in all_combinations:
        for run_id in range(num_runs):
            experiment_config = {
                "run_id": run_id,
                "mol_name": current_mol["name"],
                "mol_basis": current_mol["basis"],
                "mapping": current_mol["mapping"],
                "ansatz_type": current_ansatz["type"],
                "ansatz_layers": current_ansatz["layers"],
                "opt_type": current_opt["type"],
                "opt_lr": current_opt["lr"],
                "gradient_method": current_opt["gradient_method"],
                "noise_model": current_noise["model"],         
                "noise_strength": current_noise["strength"],
                
                "molecule": current_mol,
                "ansatz": current_ansatz,
                "optimizer": current_opt,
                "noise": current_noise

            }

            experiment_fingerprint = tuple(experiment_config[key] for key in EXPERIMENT_KEYS)
            if experiment_fingerprint in completed_runs:
                continue
                
            tasks_to_run.append(experiment_config)
            
    return tasks_to_run

def allocate_compute_workers(use_gpu=False):
    """Determines parallel process based on config"""
    if use_gpu:
        logger.info("Running in GPU mode")
        return 1
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Running in CPU mode: Utilizing {num_workers} parallel CPU processes.")
    return num_workers
