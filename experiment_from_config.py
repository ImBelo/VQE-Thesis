import concurrent.futures
from loguru import logger
import multiprocessing as mp

from utils.config.config_shuffler import ConfigGenerator
from utils.config.config import DB_PATH,EXPERIMENT_KEYS

from utils.database import get_completed_runs, save_experiment_data
from utils.task_manager import build_execution_tasks, allocate_compute_workers
from core.runner import run_single_experiment
import concurrent.futures

logger.remove()

logger.add(
    "experiments.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)

logger.add(
    lambda msg: print(msg, end=""),  
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
    level="INFO",
    colorize=True
)

def run(config,use_gpu):
    # Create from config file the cartesian product of configs
    config_manager = ConfigGenerator()
    all_combinations = config_manager.cartesian_product(config)
    num_runs = config_manager.config_data.get("settings", {}).get("num_runs", 1)

    # Check database for completed runs
    completed_runs = get_completed_runs(DB_PATH)
    logger.info(f"Total experiment to run: {len(all_combinations)}")
    logger.info(f"Found {len(completed_runs)} already completed experiments in DB.\n")

    # Build the executers and runs task that have not been run num_runs times
    tasks_to_run = build_execution_tasks(all_combinations, completed_runs, num_runs)
    logger.info(f"Total tasks to compute: {len(tasks_to_run)}")
    
    if not tasks_to_run:
        logger.info("All experiments are already completed!")
        return

    # Allocate workers depending depending if using gpu
    num_workers = allocate_compute_workers(use_gpu)
    
    ctx = mp.get_context("spawn")

    # Create num_workers of processes that will run Tasks
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx
    ) as executor:
        # Dictionary {key: run_id value: Future[Task]}
        future_to_task = {
            executor.submit(run_single_experiment, task): task["run_id"] for task in tasks_to_run
        }
        
        # As the Future[Task] completes
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                # Safely extract Task
                result_payload, error = future.result()
                
                if error:
                    logger.error(f"FAILED: {error}")
                    continue
                    
                if result_payload:
                    logger.bind(**{k: result_payload.get(k) for k in EXPERIMENT_KEYS}, 
                    energy=result_payload.get('final_energy')).info("Experiment completed")
                    
                    # Save the identifier of the Task to not run it again
                    fingerprint = save_experiment_data(result_payload, DB_PATH)
                    completed_runs.add(fingerprint)
                    
            except Exception as exc:
                logger.error(f"Task generated an unhandled exception loop crash: {exc}")

    logger.info(f"\nAll tasks completed. Saved results to {DB_PATH}")


