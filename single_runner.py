import sys
from loguru import logger
from core.runner import run_single_experiment

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",

    level="INFO",
    colorize=True
)

def main():
    print("==================================================")
    logger.info("VQE Runner for single configs")
    print("==================================================")
    
    test_config = {
        "run_id": "manual_debug_test",
        
        "molecule": {
            "name": "H2",                       # "H2", "H2O"
            "basis": "sto-3g",                  # "sto-3g", "6-31g"
            "mapping": "bravyi_kitaev"          # "bravyi_kitaev", "jordan_wigner"
        },
        
        "ansatz": {
            "type": "hardware_efficient",                    # "uccsd", "hardware_efficient"
            "layers": 8                        # uccsd 1 is fine
        },
        
        "optimizer": {
            "type": "adam",                      # "adam", "cobyla"
            "lr": 0.1,                           # for COBYLA learning rate is rhobeg
            "gradient_method": "adjoint"         # the qnode automatically puts adjoint for noise free 
                                                 # and parameter shift for noise but put this for scalability
        },
        
        "noise": {
            "model": "none",                     # "none", "depolarizing", "phase_damping", "thermal_relaxation"
            "strength": "low"                    # "low","medium", "high", (or 0.0 if model is "none") see noise config
        }
    }

    ansatz_type = test_config["ansatz"]["type"]

    noise_model = test_config["noise"]["model"]
    logger.info(f"Target: {ansatz_type} | Noise: {noise_model}")
    print("==================================================")

    result_payload, error = run_single_experiment(test_config)

    print("\n==================================================")
    if error:
        logger.error(f"❌ Execution Failed: {error}")
    elif result_payload:
        logger.success("✅ Execution Finished Successfully!")
        print(f"  ├── Final Energy    : {result_payload.get('final_energy'):.6f} Ha")

        print(f"  ├── Total Iterations: {result_payload.get('iterations')}")
        print(f"  ├── Qubits Allocated: {result_payload.get('n_qubits')}")
        print(f"  └── Total Parameters: {result_payload.get('n_params')}")
    print("==================================================")

if __name__ == "__main__":
    main()
