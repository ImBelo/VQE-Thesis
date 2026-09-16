from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

def _find_project_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file until a directory containing `marker` is found."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Project root not found (no {marker} above {here})")


PROJECT_ROOT: Path = _find_project_root()

DB_PATH: Path = PROJECT_ROOT / "utils" / "storage" / "db" / "vqe_results.db"
CONFIG_PATH_H2: Path = PROJECT_ROOT / "utils" / "config" / "config_h2.yaml"
CONFIG_PATH_H2O: Path = PROJECT_ROOT / "utils" / "config" / "config_h2o.yaml"

OUTPUT_DIR: Path = PROJECT_ROOT / "utils" / "storage" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHEMICAL_ACCURACY: float = 0.001592
STO3G_ENERGY: float = -1.137276

EXPERIMENT_KEYS: list[str] = [
    "run_id",
    "mol_name",
    "mol_basis",
    "mapping",
    "ansatz_type",
    "ansatz_layers",
    "ansatz_entanglement",
    "opt_type",
    "gradient_method",
    "opt_lr",
    "noise_model",
    "noise_strength",
]


def apply_publication_theme() -> None:
    """Apply global publication-grade styling parameters for figures."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,

        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
