import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from utils.config.config import OUTPUT_DIR, CHEMICAL_ACCURACY, STO3G_ENERGY
from . import data_loader


# ---------------------------------------------------------------------------
# Global styling — consistent across every figure in the thesis
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif"],
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.labelsize":       10,
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "axes.axisbelow":       True,
    "grid.linestyle":       ":",
    "grid.alpha":           0.4,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "legend.frameon":       False,
    "lines.linewidth":      1.6,
    "lines.solid_capstyle": "round",
})


LABELS = {
    "uccsd":                "UCCSD",
    "hardware_efficient":   "Hardware-Efficient",
    "adam":                 "ADAM",
    "cobyla":               "COBYLA",
    "bravyi_kitaev":        "Bravyi\u2013Kitaev",
    "jordan_wigner":        "Jordan\u2013Wigner",
    "amplitude_damping":    "Amplitude Damping",
    "phase_damping":        "Phase Damping",
    "depolarizing":         "Depolarizing",
    "thermal_relaxation":   "Thermal Relaxation",
    "none":                 "Noiseless",
}


def _pretty(series):
    return series.map(lambda v: LABELS.get(v, str(v).replace("_", " ").title()))


def _save(fig, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Resource vs accuracy, faceted by ansatz layers
# ---------------------------------------------------------------------------
def plot_resource_vs_accuracy_faceted(db_path, mol_name="H2", basis="sto-3g"):
    df = data_loader.fetch_resource_accuracy_data(db_path, mol_name, basis)
    if df.empty:
        return

    df = df.copy()
    df["Energy_Error"] = (df["final_energy"] - STO3G_ENERGY).abs()
    df["Configuration"] = _pretty(df["ansatz_type"]) + " + " + _pretty(df["opt_type"])
    df["Noise_Status"] = df["noise_model"].apply(
        lambda x: "Ideal" if str(x).lower() == "none" else "Noisy"
    )
    df["Group"] = df["Configuration"] + " (" + df["Noise_Status"] + ")"
    df["mapping_label"] = _pretty(df["mapping"])

    # Drop zero or negative errors so the log axis doesn't blow up
    df = df[df["Energy_Error"] > 0]
    if df.empty:
        return

    # Stable ordering for color and marker assignment across figures
    groups = sorted(df["Group"].unique())
    mappings = sorted(df["mapping_label"].unique())
    group_palette = dict(
        zip(groups, sns.color_palette("colorblind", len(groups)))
    )

    g = sns.FacetGrid(
        df,
        col="ansatz_layers",
        col_wrap=3,
        height=3.4,
        aspect=1.25,
        sharex=True,
        sharey=True,
        hue="Group",
        hue_order=groups,
        palette=group_palette,
        legend_out=True,
    )
    g.map_dataframe(
        sns.scatterplot,
        x="total_iterations",
        y="Energy_Error",
        s=70,
        alpha=0.85,
    )

    for ax in g.axes.flatten():
        ax.axhline(CHEMICAL_ACCURACY, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=1)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    g.set_titles(col_template="{col_name} layer blocks", fontweight="semibold")
    g.set_axis_labels("Optimization iterations",
                      "Absolute energy error (Hartree)")

    g.add_legend(title="Setup", bbox_to_anchor=(1.02, 0.5), loc="center left")

    _save(g.figure, f"{mol_name}_{basis}_resource_accuracy_by_layers")
    print(f"[+] Saved resource-vs-accuracy plot: "
          f"{mol_name}_{basis}_resource_accuracy_by_layers")


# ---------------------------------------------------------------------------
# Optimizer efficiency matrix
# ---------------------------------------------------------------------------
def plot_optimizer_efficiency(db_path, mol_name="H2"):
    df = data_loader.fetch_optimizer_efficiency_data(db_path, mol_name)
    if df.empty:
        return

    df = df.copy()

    # Keep the filter, but make it explicit rather than magic (-0.8)
    df = df[df["final_energy"] < -0.8]
    if df.empty:
        return

    df["opt_label"] = _pretty(df["opt_type"])
    df["noise_label"] = _pretty(df["noise_model"])

    # Stable ordering of categories and colors across figures
    optimizers = sorted(df["opt_label"].unique())
    palette = dict(
        zip(optimizers, sns.color_palette("colorblind", len(optimizers)))
    )

    g = sns.FacetGrid(
        df,
        col="noise_label",
        col_wrap=2,
        height=3.6,
        aspect=1.3,
        sharex=False,
        sharey=True,
        hue="opt_label",
        hue_order=optimizers,
        palette=palette,
        legend_out=True,
    )
    g.map_dataframe(
        sns.scatterplot,
        x="total_iterations",
        y="final_energy",
        s=90,
        alpha=0.9,
        edgecolor="0.2",
    )

    for ax in g.axes.flatten():
        ax.axhline(STO3G_ENERGY, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=1)
        ax.grid(True, linestyle=":", alpha=0.4)

    g.set_titles(col_template="{col_name}", fontweight="semibold")
    g.set_axis_labels("Iterations to convergence",
                      "Final energy (Hartree)")

    g.add_legend(title="Optimizer", bbox_to_anchor=(1.02, 0.5),
                 loc="center left")

    _save(g.figure, "optimizer_efficiency_matrix")
    print("[+] Saved optimizer efficiency matrix")
