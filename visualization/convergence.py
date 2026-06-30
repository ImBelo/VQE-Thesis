import matplotlib.pyplot as plt
import seaborn as sns
from utils.config.config import OUTPUT_DIR, STO3G_ENERGY
from . import data_loader

def plot_trajectories(db_path, mol_name="H2", basis="sto-3g"):
    fci_exact = STO3G_ENERGY
    df = data_loader.fetch_trajectory_data(db_path, mol_name, basis)
    if df.empty: return
    
    df['Algorithm'] = df['ansatz_type'].str.upper() + " + " + df['opt_type'].str.upper()

    
    g = sns.FacetGrid(df, col="noise_model", col_wrap=2, height=4, aspect=1.4, sharey=True)
    g.map_dataframe(sns.lineplot, x='step', y='energy', hue='Algorithm', units='unique_run_id', estimator=None, alpha=0.85, linewidth=2)
    
    for ax in g.axes.flatten():
        ax.axhline(y=fci_exact, color='crimson', linestyle=':', linewidth=1.5, label='Exact FCI Limit')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlim(0, df['step'].max())
        ax.set_ylim(fci_exact - 0.02, -0.85)
        
    g.add_legend(title="Execution Setup", adjust_subtitles=True)
    g.set_axis_labels("Optimization Step", "Energy (Hartree)")
    g.set_titles(col_template="Noise Profile: {col_name}", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{mol_name}_{basis}_convergence_trajectories.png")
    plt.close()


def plot_multi_noise_rollercoasters(db_path):
    fci_exact = STO3G_ENERGY
    df = data_loader.fetch_rollercoaster_data(db_path)
    if df.empty: return
    
    df['noise_model_label'] = df['noise_model'].str.replace('_', ' ').str.title()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    unique_models = df['noise_model_label'].unique()
    colors = sns.color_palette("tab10", len(unique_models))
    
    for idx, model in enumerate(unique_models):
        df_subset = df[df['noise_model_label'] == model]
        ax.plot(df_subset['ansatz_layers'], df_subset['final_energy'], marker='o', linestyle='-', linewidth=2.5, markersize=7, color=colors[idx], label=model, alpha=0.9)
        

    ax.axhline(y=fci_exact, color='crimson', linestyle='--', linewidth=2.0, label='Ground State')
    ax.set_xlim(0.6, df['ansatz_layers'].max() + 0.4)
    ax.set_xticks(sorted(df['ansatz_layers'].unique()))

    
    ax.set_title("VQE Layer Scaling Profiles: Expressibility vs. Noise Decay", fontweight='bold', pad=15)
    ax.set_xlabel("Hardware Efficient Layer Blocks (Circuit Depth)", fontweight='semibold', labelpad=10)
    ax.set_ylabel("Calculated Energy (Hartree)", fontweight='semibold', labelpad=10)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, edgecolor="0.8")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vqe_multi_noise_rollercoasters.png")
    plt.close()
