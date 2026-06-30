import yaml
import itertools
from pathlib import Path
from loguru import logger

class ConfigGenerator:
    def __init__(self):
        self.config_data = {}

    def cartesian_product(self,file_path: Path):
        try:
            target_path = file_path.resolve()

            with target_path.open("r") as file:
                self.config_data = yaml.safe_load(file)

        except FileNotFoundError:
            logger.error(f"Error: Could not find configuration file at: {file_path}")
            logger.error(f"Attempted absolute path: {file_path.resolve()}")
            return []  

        # Cartesian product of molecules config
        mol = self.config_data["molecules"]
        mol_names = mol["names"]
        mol_basis_sets = mol["basis_sets"]
        mol_mappings = mol["mappings"] 
        molecule_list = [
            {"name": n, "basis": b, "mapping": m} 
            for n, b, m in itertools.product(mol_names, mol_basis_sets, mol_mappings)
        ]

        # Cartesian product of ansatz config
        ansatz_defs = self.config_data["ansatz_types"]
        ansatz_list = [
            {"type": ansatz_type, "layers": layers}
            for ansatz_type in ["hardware_efficient", "uccsd"]
            for layers in ansatz_defs[ansatz_type]["layers"]
        ]

        # Cartesian product of optimizers config
        optimizer_list = [
            {"type": opt_def["type"], "lr": lr, "gradient_method": gm}
            for opt_def in self.config_data["optimizers"]
            for lr in opt_def.get("lr", [0.1])
            for gm in opt_def["gradient_method"]
        ]

        noise_list = []
        
        # Safely grab the YAML section
        raw_noise_models = self.config_data.get("noise_models")
        
        # If the key is missing entirely, or the user left it blank (None) or empty ([]), force a default
        if not raw_noise_models:
            raw_noise_models = [{"model": "none", "strengths": [0.0]}]

        for noise_def in raw_noise_models:
            model = noise_def.get("model", "none")
            
            # Safely grab strengths. If accidentally left empty, fallback to [0.0]
            raw_strengths = noise_def.get("strengths")
            if not raw_strengths:
                raw_strengths = [0.0]
                
            for strength in raw_strengths:
                noise_config = {
                    "model": model,
                    "strength": strength
                }
                
                # Pass along any optional thermal/qubit parameters
                for key in ["qubits", "t1", "t2"]:
                    if key in noise_def:
                        noise_config[key] = noise_def[key]
                        
                noise_list.append(noise_config)
        all_combinations = list(
            itertools.product(molecule_list, ansatz_list, optimizer_list, noise_list)
        )

        return all_combinations
