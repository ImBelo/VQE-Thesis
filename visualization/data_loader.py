import sqlite3
from pathlib import Path
import pandas as pd


# Both tables now carry unique_run_id (TEXT). The join is an exact,
# indexable TEXT equality on both sides.
_JOIN = "h.unique_run_id = e.unique_run_id"


def fetch_resource_accuracy_data(db_path: Path, mol_name: str, basis: str) -> pd.DataFrame:
    query = f"""
    SELECT
        e.unique_run_id,
        e.ansatz_type,
        e.opt_type,
        e.mapping,
        e.noise_model,
        e.noise_strength,
        e.ansatz_layers,
        h_max.step   AS total_iterations,
        h_max.energy AS final_energy
    FROM experiments e
    INNER JOIN (
        SELECT unique_run_id, MAX(step) AS max_step
        FROM optimization_history
        GROUP BY unique_run_id
    ) h_meta
        ON h_meta.unique_run_id = e.unique_run_id
    INNER JOIN optimization_history h_max
        ON  h_max.unique_run_id = h_meta.unique_run_id
        AND h_max.step          = h_meta.max_step
    WHERE e.mol_name = ? AND e.mol_basis = ?;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name, basis.lower()))


def fetch_mapping_delta_data(db_path: Path, mol_name: str, basis: str) -> pd.DataFrame:
    query = f"""
    SELECT h.step, h.energy AS energy,
           e.ansatz_type, e.noise_model, e.opt_type, e.mapping
    FROM optimization_history h
    INNER JOIN experiments e ON {_JOIN}
    WHERE e.mol_name = ? AND e.mol_basis = ?;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name, basis.lower()))


def fetch_trajectory_data(db_path: Path, mol_name: str, basis: str) -> pd.DataFrame:
    query = f"""
    SELECT h.unique_run_id, h.step, h.energy AS energy,
           e.ansatz_type, e.noise_model, e.opt_type, e.ansatz_layers
    FROM optimization_history h
    INNER JOIN experiments e ON {_JOIN}
    WHERE e.mol_name = ? AND e.mol_basis = ?
      AND (e.ansatz_type = 'uccsd' OR e.ansatz_layers NOT IN (1, 2, 3, 7))
    ORDER BY h.unique_run_id, h.step ASC;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name, basis.lower()))


def fetch_noise_resilience_data(db_path: Path, mol_name: str) -> pd.DataFrame:
    query = """
    SELECT e.noise_model, e.noise_strength, e.ansatz_type, e.mapping,
           AVG(e.final_energy) AS final_energy,
           CASE
               WHEN COUNT(e.final_energy) > 1 THEN
                   SQRT(
                       (SUM(e.final_energy * e.final_energy)
                        - SUM(e.final_energy) * SUM(e.final_energy)
                          / COUNT(e.final_energy))
                       / (COUNT(e.final_energy) - 1)
                   )
               ELSE 0
           END AS std_energy,
           COUNT(*) AS n_runs
    FROM experiments e
    WHERE e.mol_name = ?
      AND e.final_energy < -0.9
      AND e.converged = 1
    GROUP BY e.noise_model, e.noise_strength, e.ansatz_type, e.mapping;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name,))


def fetch_optimizer_efficiency_data(db_path: Path, mol_name: str) -> pd.DataFrame:
    query = f"""
    SELECT e.opt_type, e.gradient_method, e.noise_model, e.ansatz_layers,
       AVG(e.iterations)     AS total_iterations,
       AVG(e.final_energy)   AS final_energy,
       COUNT(*)              AS n_runs
    FROM experiments e
    WHERE e.mol_name = ? AND e.ansatz_layers NOT IN (1, 2, 3, 7)
    GROUP BY e.opt_type, e.gradient_method, e.noise_model, e.ansatz_layers;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(mol_name,))


def fetch_rollercoaster_data(db_path: Path) -> pd.DataFrame:
    query = f"""
    WITH final_steps AS (
        SELECT unique_run_id, MAX(step) AS last_step
        FROM optimization_history
        GROUP BY unique_run_id
    )
    SELECT e.ansatz_layers,
           e.ansatz_entanglement,
           e.noise_model,
           AVG(e.final_energy)   AS final_energy,
           COUNT(*)              AS n_runs
    FROM experiments e
    WHERE e.ansatz_type = 'hardware_efficient'
      AND e.converged = 1
      AND e.final_energy < -0.9
    GROUP BY e.ansatz_layers, e.ansatz_entanglement, e.noise_model
    ORDER BY e.noise_model, e.ansatz_layers, e.ansatz_entanglement;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
