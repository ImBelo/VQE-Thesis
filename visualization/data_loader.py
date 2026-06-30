import sqlite3
from pathlib import Path
import pandas as pd

def fetch_resource_accuracy_data(db_path: Path, mol_name: str, basis: str) -> pd.DataFrame:
    query = """
    SELECT 
        e.unique_run_id, e.ansatz_type, e.opt_type, e.mapping, e.noise_model, 
        e.noise_strength, e.ansatz_layers, h_max.step as total_iterations,
        h_max.energy as final_energy

    FROM experiments e
    INNER JOIN (
        SELECT unique_run_id, MAX(step) as max_step FROM optimization_history GROUP BY unique_run_id
    ) h_meta ON e.unique_run_id = h_meta.unique_run_id

    INNER JOIN optimization_history h_max 

        ON h_meta.unique_run_id = h_max.unique_run_id AND h_meta.max_step = h_max.step
    WHERE e.mol_name = ? AND e.mol_basis = ?;

    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name, basis.lower()))

def fetch_mapping_delta_data(db_path: Path, mol_name: str, basis: str) -> pd.DataFrame:
    query = """
    SELECT h.step, h.energy as energy, e.ansatz_type, e.noise_model, e.opt_type, e.mapping
    FROM optimization_history h
    INNER JOIN experiments e ON h.unique_run_id = e.unique_run_id
    WHERE e.mol_name = ? AND e.mol_basis = ?;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name, basis.lower()))


def fetch_trajectory_data(db_path: Path, mol_name: str, basis: str) -> pd.DataFrame:

    query = """
    SELECT h.unique_run_id, h.step, h.energy as energy, e.ansatz_type, e.noise_model, e.opt_type, e.ansatz_layers
    FROM optimization_history h
    INNER JOIN experiments e ON h.unique_run_id = e.unique_run_id
    WHERE e.mol_name = ? AND e.mol_basis = ?
      AND (e.ansatz_type = 'uccsd' OR e.ansatz_layers NOT IN (1, 2, 3, 7))
    ORDER BY h.unique_run_id, h.step ASC;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name, basis.lower()))

def fetch_noise_resilience_data(db_path: Path, mol_name: str) -> pd.DataFrame:
    query = """
    SELECT e.noise_model, e.noise_strength, e.ansatz_type, e.mapping, MIN(CAST(h.energy AS REAL)) as final_energy

    FROM experiments e
    INNER JOIN optimization_history h ON e.unique_run_id = h.unique_run_id
    WHERE e.mol_name = ?
    GROUP BY e.noise_model, e.ansatz_type, e.mapping;

    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name,))

def fetch_optimizer_efficiency_data(db_path: Path, mol_name: str) -> pd.DataFrame:
    query = """
    SELECT e.opt_type, e.gradient_method, e.noise_model, e.ansatz_layers,
           COUNT(h.step) as total_iterations, MIN(CAST(h.energy AS REAL)) as final_energy
    FROM experiments e
    INNER JOIN optimization_history h ON e.unique_run_id = h.unique_run_id
    WHERE e.mol_name = ? AND e.ansatz_layers NOT IN (1, 2, 3, 7)
    GROUP BY e.unique_run_id;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name,))

def fetch_rollercoaster_data(db_path: Path) -> pd.DataFrame:
    query = """
    WITH final_steps AS (
        SELECT unique_run_id, MAX(step) AS last_step FROM optimization_history GROUP BY unique_run_id
    )
    SELECT e.ansatz_layers, e.noise_model, AVG(h.energy) AS final_energy       
    FROM experiments e
    JOIN optimization_history h ON e.unique_run_id = h.unique_run_id
    JOIN final_steps f ON h.unique_run_id = f.unique_run_id AND h.step = f.last_step
    WHERE e.ansatz_type = 'hardware_efficient'
    GROUP BY e.noise_model, e.ansatz_layers
    ORDER BY e.noise_model, e.ansatz_layers;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
