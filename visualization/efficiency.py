import matplotlib.pyplot as plt
import seaborn as sns
from utils.config.config import OUTPUT_DIR, CHEMICAL_ACCURACY, STO3G_ENERGY
from . import data_loader

def plot_resource_vs_accuracy_faceted(db_path, mol_name="H2", basis="sto-3g"):
    df = data_loader.fetch_resource_accuracy_data(db_path, mol_name, basis)
    if df.empty: return
    
    fci_exact = STO3G_ENERGY
    df['Energy_Error'] = (df['final_energy'] - fci_exact).abs()
    df['Configuration'] = df['ansatz_type'].str.upper() + " + " + df['opt_type'].str.upper()
    df['Noise_Status'] = df['noise_model'].apply(lambda x: 'Ideal' if str(x).lower() == 'none' else 'Noisy')
    df['Group'] = df['Configuration'] + " (" + df['Noise_Status'] + ")"
    df = df.sort_values(by='ansatz_layers')

    g = sns.FacetGrid(
        df, 
        col="ansatz_layers", 
        col_wrap=3, 
        height=4, 
        aspect=1.2, 
        sharex=True, 
        sharey=True, 
        legend_out=True
    )

    g.map_dataframe(
        sns.scatterplot, 
        x='total_iterations', 
        y='Energy_Error', 
        hue='Group', 
        style='mapping', 
        s=90, 
        alpha=0.85, 
        palette="Paired"
    )    

    for ax in g.axes.flatten():
        ax.axhline(y=CHEMICAL_ACCURACY, color='teal', linestyle='--', linewidth=1.2)
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle=':', alpha=0.5)

        
    g.set_titles(col_template="{col_name} Layers", fontweight='bold')
    g.set_axis_labels("Total Optimization Iterations", "Absolute Energy Error (Log Scale)")
    
    g.add_legend(title="Execution Parameters", frameon=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{mol_name}_{basis}_resource_accuracy_by_layers.png")
    plt.close()

def plot_optimizer_efficiency(db_path, mol_name="H2"):
    df = data_loader.fetch_optimizer_efficiency_data(db_path, mol_name)
    if df.empty: return
    
    df = df[df['final_energy'] < -0.8]
    fci_exact = STO3G_ENERGY
    
    g = sns.FacetGrid(df, col="noise_model", hue="opt_type", col_wrap=2, height=4, aspect=1.3, palette="viridis")
    g.map_dataframe(sns.scatterplot, x='total_iterations', y='final_energy', s=120, alpha=0.9, edgecolor="0.2")
    
    for ax in g.axes.flatten():
        ax.axhline(y=fci_exact, color='r', linestyle=':', alpha=0.7, label='Exact FCI')
        ax.grid(True, linestyle=':', alpha=0.5)
        

    g.add_legend(title="Optimizer Type", adjust_subtitles=True)
    g.set_axis_labels("Iterations to Convergence", "Final Energy (Hartree)")
    g.set_titles(col_template="Noise Profile: {col_name}", fontweight='bold')
    

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "optimizer_efficiency_matrix.png")
    plt.close()
