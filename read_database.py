import sqlite3
import pandas as pd
from utils.config.config import DB_PATH ,EXPERIMENT_KEYS
def analyze_vqe_results(db_name=DB_PATH):
    with sqlite3.connect(db_name) as engine:
        df = pd.read_sql("SELECT * FROM experiments", engine)

    if df.empty:
        print("The database is empty. No results to analyze.")
        return

    grouping_columns = EXPERIMENT_KEYS[1:]

    actual_grouping_cols = [col for col in grouping_columns if col in df.columns]

    summary_df = df.groupby(actual_grouping_cols).agg(
        total_runs=('run_id', 'count'),                 
        energy_mean=('final_energy', 'mean'),           
        energy_std=('final_energy', 'std'),             
        avg_iterations=('iterations', 'mean')
    ).reset_index()

    summary_df['Energy (Mean ± Std)'] = summary_df.apply(
        lambda row: f"{row['energy_mean']:.6f} ± {row['energy_std']:.6f}" 
                    if pd.notnull(row['energy_std']) 
                    else f"{row['energy_mean']:.6f} (Single Run)", 
        axis=1
    )

    summary_df = summary_df.sort_values(by='energy_mean', ascending=True)

    display_cols = actual_grouping_cols + ['total_runs', 'Energy (Mean ± Std)', 'avg_iterations']
    
    print("\n=== VQE Experiment Results ===")
    print(summary_df[display_cols].to_string(index=False))

if __name__ == "__main__":
    analyze_vqe_results()
