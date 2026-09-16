from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from loguru import logger


class ConfigGenerator:
    """Generate Cartesian products of experiment configurations from a YAML file."""

    config_data: Dict[str, Any]

    def __init__(self) -> None:
        self.config_data: Dict[str, Any] = {}

    def cartesian_product(
        self,
        file_path: Path,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """
        Load a YAML config and return every combination of
        (molecule, ansatz, optimizer, noise) as a list of tuples.

        Returns an empty list if the file can't be read.
        """
        try:
            target_path = file_path.resolve()

            with target_path.open("r") as file:
                self.config_data = yaml.safe_load(file)

        except FileNotFoundError:
            logger.error(f"Error: Could not find configuration file at: {file_path}")
            logger.error(f"Attempted absolute path: {file_path.resolve()}")
            return []

        # --- molecules -------------------------------------------------------
        mol = self.config_data["molecules"]
        mol_names: List[str] = mol["names"]
        mol_basis_sets: List[str] = mol["basis_sets"]
        mol_mappings: List[str] = mol["mappings"]

        molecule_list: List[Dict[str, Any]] = [
            {"name": n, "basis": b, "mapping": m}
            for n, b, m in itertools.product(mol_names, mol_basis_sets, mol_mappings)
        ]

        # --- ansatz ----------------------------------------------------------
        ansatz_defs: Dict[str, Dict[str, Any]] = self.config_data["ansatz_types"]
        ansatz_list: List[Dict[str, Any]] = []

        for ansatz_type, config in ansatz_defs.items():
            param_names: List[str] = list(config.keys())
            param_values: List[List[Any]] = [
                config[name] if isinstance(config[name], list) else [config[name]]
                for name in param_names
            ]

            for values in itertools.product(*param_values):
                ansatz: Dict[str, Any] = {"type": ansatz_type}
                ansatz.update(dict(zip(param_names, values)))
                ansatz_list.append(ansatz)

        # --- optimizers ------------------------------------------------------
        optimizer_list: List[Dict[str, Any]] = [
            {"type": opt_def["type"], "lr": lr, "gradient_method": gm}
            for opt_def in self.config_data["optimizers"]
            for lr in opt_def.get("lr", [0.1])
            for gm in opt_def["gradient_method"]
        ]

        # --- noise -----------------------------------------------------------
        noise_list: List[Dict[str, Any]] = []

        raw_noise_models: List[Dict[str, Any]] = self.config_data.get(
            "noise_models"
        ) or [{"model": "none", "strengths": [0.0]}]

        for noise_def in raw_noise_models:
            model: str = noise_def.get("model", "none")

            # Thermal relaxation uses T1/T2/Tg, not a scalar strength
            if model == "thermal_relaxation":
                noise_config: Dict[str, Any] = {
                    "model": model,
                    "strength": "physical",
                    "t1": noise_def.get("t1", 50.0),
                    "t2": noise_def.get("t2", 70.0),
                    "tg": noise_def.get("tg", 0.1),
                }
                if "qubits" in noise_def:
                    noise_config["qubits"] = noise_def["qubits"]
                noise_list.append(noise_config)
                continue

            # All other noise models: one entry per strength
            strengths: List[Any] = noise_def.get("strengths", [0.0]) or [0.0]

            for strength in strengths:
                noise_config = {
                    "model": model,
                    "strength": strength,
                }
                if "qubits" in noise_def:
                    noise_config["qubits"] = noise_def["qubits"]
                noise_list.append(noise_config)

        # --- combined product --------------------------------------------------
        all_combinations: List[
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]
        ] = list(
            itertools.product(
                molecule_list,
                ansatz_list,
                optimizer_list,
                noise_list,
            )
        )
        return all_combinations
