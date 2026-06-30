from loguru import logger
from analysis.vqe_pipeline import VQEPipeline
from contextlib import nullcontext
def experiment_pretty_print(
    converged: bool, 
    final_energy: float, 
    iterations: int, 
    execution_time: float, 
    current_mol: str, 
    current_ansatz: str, 
    current_opt: str, 
    current_noise: str
):
    it_per_sec = iterations / execution_time if execution_time > 0 else 0.0

    metadata = f"[{current_mol} | {current_ansatz} | {current_opt} | Noise: {current_noise}]"

    if converged:
        msg = (
            "✓ VQE Optimization Converged!\n"
            "  ├── Config     : {}\n"
            "  ├── Energy     : {:.6f} Ha\n"
            "  ├── Iterations : {}\n"
            "  └── Speed      : {:.2f}s ({:.1f} it/s)"
        )
        logger.success(msg, metadata, final_energy, iterations, execution_time, it_per_sec)
    else:
        msg = (
            "⚠ VQE Optimization Terminated Early (Unconverged)\n"
            "  ├── Config     : {}\n"
            "  ├── Last Energy: {:.6f} Ha\n"
            "  ├── Steps Taken: {}\n"
            "  └── Time Spent : {:.2f}s"
        )
        logger.warning(msg, metadata, final_energy, iterations, execution_time)


def run_single_experiment(experiment_config):
    """ Runs a single experiment given an experiment_config dictionary. """
    try:
        current_mol = experiment_config.pop("molecule")
        current_ansatz = experiment_config.pop("ansatz")
        current_opt = experiment_config.pop("optimizer")
        current_noise = experiment_config.pop("noise")
        
        pipeline = VQEPipeline.from_config(current_mol, current_ansatz, current_opt, current_noise)
        
        n_qubits = pipeline.n_qubits
        n_params = pipeline.ansatz.get_num_params() 

        iterations, history, optimal_params, converged, execution_time = pipeline.run()  
        energies = history["energies"]
        final_energy = energies[-1]
        experiment_pretty_print(converged,final_energy,iterations,execution_time,current_mol,current_ansatz,current_opt,current_noise)

        experiment_result = {
            **experiment_config, 
            "iterations": iterations,
            "n_qubits": n_qubits,
            "n_params": n_params,
            "converged": converged,
            "final_energy": float(final_energy),
            "runtime_sec": round(execution_time, 2),
            "energy_history": [float(e) for e in energies]
        }
        return experiment_result, None

    except Exception as e:

        return None, f"Error {str(e)} in running: {experiment_config['run_id']}"
