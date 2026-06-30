from pathlib import Path
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DB_PATH = PROJECT_ROOT / "utils" / "storage" / "db" / "vqe_results.db"
CONFIG_PATH_H2 = PROJECT_ROOT / "utils" / "config" / "config_h2.yaml"
CONFIG_PATH_H2O = PROJECT_ROOT / "utils" / "config" / "config_h2o.yaml"

OUTPUT_DIR = PROJECT_ROOT / "utils" / "storage" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHEMICAL_ACCURACY = 0.001592  
STO3G_ENERGY = -1.137276

EXPERIMENT_KEYS = [
    "run_id",
    "mol_name", 
    "mol_basis", 
    "mapping", 
    "ansatz_type", 
    "ansatz_layers", 
    "opt_type", 
    "gradient_method",
    "opt_lr",
    "noise_model",
    "noise_strength",
]


def apply_publication_theme():
    """Applies global publication-grade styling parameters for figures."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,

        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10

    })
