import sqlite3
import pandas as pd
from pathlib import Path
from utils.config.config import DB_PATH


def test():
    if not DB_PATH.exists():
        print("[-] Database not found. Run your pipeline with the new ID comprehension first.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        e.unique_run_id,
        e.ansatz_type,
        e.opt_type,
        e.mapping,
        e.noise_model,
        e.noise_strength,

        e.ansatz_layers,
        h_max.step as total_iterations,

       h_max.energy as final_energy
    FROM experiments e

    INNER JOIN (
        SELECT unique_run_id, MAX(step) as max_step
        FROM optimization_history
        GROUP BY unique_run_id
    ) h_meta ON e.unique_run_id = h_meta.unique_run_id
    INNER JOIN optimization_history h_max 
        ON h_meta.unique_run_id = h_max.unique_run_id AND h_meta.max_step = h_max.step

    WHERE e.mol_name = "H2" AND e.mol_basis = "sto-3g";
    """
    
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:

        print(f"[-] Query failed: {e}")
        conn.close()
        return
    
    conn.close()
    
    print("\n=== VERIFYING FINAL ENERGIES BY BASIS SET ===")
    if df.empty:
        print("Database is currently empty. Run a few iterations of your VQE script!")
    else:
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)

        pd.set_option('display.width', 200)
        print(df.to_string(index=False))
        print("===================================\n")

if __name__ == "__main__":
    test()
