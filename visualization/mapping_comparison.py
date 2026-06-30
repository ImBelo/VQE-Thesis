import matplotlib.pyplot as plt
import seaborn as sns
from utils.config.config import OUTPUT_DIR, CHEMICAL_ACCURACY, STO3G_ENERGY
from . import data_loader


def plot_mapping_delta(db_path, mol_name="H2", basis="sto-3g"):
    df = data_loader.fetch_mapping_delta_data(db_path, mol_name, basis)
    if df.empty: return
    
    df_pivot = df.pivot_table(
        index=['step', 'ansatz_type', 'noise_model', 'opt_type'], 
        columns='mapping', 
        values='energy'
    ).reset_index()
    
    df_pivot = df_pivot.dropna(subset=['jordan_wigner', 'bravyi_kitaev'])
    
    df_pivot['Energy_Delta'] = df_pivot['jordan_wigner'] - df_pivot['bravyi_kitaev']
    df_pivot['Algorithm'] = df_pivot['ansatz_type'].str.upper() + " + " + df_pivot['opt_type'].str.upper()
    
    g = sns.FacetGrid(df_pivot, col="noise_model", col_wrap=2, height=4, aspect=1.4)

    g.map_dataframe(sns.lineplot, x='step', y='Energy_Delta', hue='Algorithm', linewidth=2.2, alpha=0.9)
    
    for ax in g.axes.flatten():
        ymin, ymax = ax.get_ylim()
        
        ax.text(x=200, y=ymax * 0.7, s="▲ BK Wins", color="green", fontsize=9, fontweight="semibold", alpha=0.7)
        ax.text(x=200, y=ymin * 0.7, s="▼ JW Wins", color="blue", fontsize=9, fontweight="semibold", alpha=0.7)
        
    g.add_legend(title="Ansatz Style",bbox_to_anchor=(0.8,0.2))
    g.set_axis_labels("Optimization Step", r"$\Delta$ Energy: $E_{JW} - E_{BK}$ (Hartree)")
    g.set_titles(col_template="Noise Profile: {col_name}", fontweight='bold')
    
    plt.tight_layout()
 
    plt.subplots_adjust(right=0.91)
    
    plt.savefig(OUTPUT_DIR / f"{mol_name}_{basis}_mapping_delta_analysis.png", dpi=300)
    plt.close()
    print(f"[+] Saved custom Delta mapping plot: {mol_name}_{basis}_mapping_delta_analysis.png")

def plot_noise_resilience(db_path, mol_name="H2", basis="sto-3g"):
    df = data_loader.fetch_noise_resilience_data(db_path, mol_name)
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=df, x='noise_model', y='final_energy', hue='ansatz_type', style='mapping', palette="viridis", markers=True, markersize=10, linewidth=2.5, ax=ax)
    
    fci_exact = STO3G_ENERGY

    ax.axhline(y=fci_exact, color='crimson', linestyle=':', linewidth=1.5, label='Exact FCI Limit', alpha=0.8)

    
    ax.set_title(f"Ansatz & Mapping Resilience vs. Noise Channels ({mol_name})", fontweight='bold', pad=12)
    ax.set_xlabel("Noise Environment Profile")

    ax.set_ylabel("Final Converged Energy (Hartree)")
    ax.set_ylim(df['final_energy'].min() - 0.01, df['final_energy'].max() + 0.01)
    ax.legend(title="Ansatz & Mapping Config", loc="best")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{mol_name}_noise_mapping_resilience_benchmark.png")
    plt.close()
