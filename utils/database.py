import sqlite3
import pandas as pd
from loguru import logger


from utils.config.config import DB_PATH, EXPERIMENT_KEYS


def get_completed_runs(db_path=DB_PATH):
    """Reads the database and returns a set of completed experiment fingerprints."""
    try:
        db_path.resolve()
    except FileNotFoundError:
        logger.error(f"Error: Could not find DB file at: {db_path}")
        return set()      
        
    try:
        with sqlite3.connect(db_path) as engine:
            columns_str = ", ".join(EXPERIMENT_KEYS)
            query = f"SELECT {columns_str} FROM experiments"

            df = pd.read_sql(query, engine)
        return set(tuple(row) for row in df.to_numpy())
        
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return set()    


def save_experiment_data(result_payload, db_name=DB_PATH):
    """Saves experiment data in database"""
    energy_trajectory = result_payload.pop("energy_history", [])

    # Experiment identifier
    unique_run_id = "_".join([
        f"run{result_payload[k]}" if k == "run_id" else
        f"L{result_payload[k]}" if k == "ansatz_layers" else
        f"lr{result_payload[k]}" if k == "opt_lr" else
        str(result_payload[k])
        for k in EXPERIMENT_KEYS
    ])
    result_payload["unique_run_id"] = unique_run_id

    experiment_fingerprint = tuple(result_payload[key] for key in EXPERIMENT_KEYS)

    with sqlite3.connect(db_name) as engine:
        # Add experiment to Database
        df_summary = pd.DataFrame([result_payload])
        df_summary.to_sql("experiments", con=engine, if_exists="append", index=False)

        # Add energy history linked to that experiment
        if energy_trajectory:
            history_df = pd.DataFrame({
                "unique_run_id": unique_run_id,
                "step": list(range(1, len(energy_trajectory) + 1)),
                "energy": energy_trajectory
            })
            history_df["energy"] = pd.to_numeric(history_df["energy"], errors='coerce')
            history_df.to_sql("optimization_history", con=engine, if_exists="append", index=False)

    return experiment_fingerprint
