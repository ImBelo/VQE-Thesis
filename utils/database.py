from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import pandas as pd
from loguru import logger

from utils.config.config import DB_PATH, EXPERIMENT_KEYS


def get_completed_runs(
    db_path: Path = DB_PATH,
) -> Set[Tuple[Any, ...]]:
    """Read the database and return a set of completed experiment fingerprints."""
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

        return {tuple(row) for row in df.to_numpy()}

    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return set()


def save_experiment_data(
    result_payload: Dict[str, Any],
    db_name: Path = DB_PATH,
) -> Tuple[Any, ...]:
    """Save experiment data and its energy trajectory to the database."""
    payload: Dict[str, Any] = result_payload.copy()
    energy_trajectory: List[float] = payload.pop("energy_history", [])

    payload.pop("molecule", None)
    payload.pop("ansatz", None)
    payload.pop("optimizer", None)
    payload.pop("noise", None)

    # Build a human-readable run identifier
    unique_run_id: str = "_".join(
        (
            f"run{payload[k]}" if k == "run_id"
            else f"L{payload[k]}" if k == "ansatz_layers"
            else f"lr{payload[k]}" if k == "opt_lr"
            else str(payload[k])
        )
        for k in EXPERIMENT_KEYS
    )
    payload["unique_run_id"] = unique_run_id

    experiment_fingerprint: Tuple[Any, ...] = tuple(
        payload[key] for key in EXPERIMENT_KEYS
    )

    with sqlite3.connect(db_name) as engine:
        df_summary = pd.DataFrame([payload])
        df_summary.to_sql(
            "experiments",
            con=engine,
            if_exists="append",
            index=False,
        )

        if energy_trajectory:
            history_df = pd.DataFrame({
                "unique_run_id": unique_run_id,
                "step": list(range(1, len(energy_trajectory) + 1)),
                "energy": energy_trajectory,
            })
            history_df["energy"] = pd.to_numeric(
                history_df["energy"], errors="coerce"
            )
            history_df.to_sql(
                "optimization_history",
                con=engine,
                if_exists="append",
                index=False,
            )

    return experiment_fingerprint
