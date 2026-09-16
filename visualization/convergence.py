import matplotlib.pyplot as plt
import seaborn as sns
from utils.config.config import OUTPUT_DIR, STO3G_ENERGY
from . import data_loader

plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
    "font.family":         "serif",
    "font.serif":          ["DejaVu Serif"],   # or "Latin Modern Roman"
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "axes.linewidth":      0.8,
    "axes.grid":           True,
    "axes.axisbelow":      True,
    "grid.linestyle":      ":",
    "grid.alpha":          0.4,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "legend.frameon":      False,
    "lines.linewidth":     1.6,
    "lines.solid_capstyle": "round",
    "figure.autolayout":   False,
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
    """Save a figure as both PDF (vector) and PNG (raster)."""
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)

def plot_trajectories(db_path, mol_name="H2", basis="sto-3g"):
    fci_exact = STO3G_ENERGY
    df = data_loader.fetch_trajectory_data(db_path, mol_name, basis)
    if df.empty:
        return

    df = df.copy()
    df["Algorithm"] = _pretty(df["ansatz_type"]) + " + " + _pretty(df["opt_type"])

    ymin = min(df["energy"].min(), fci_exact) - 0.01
    ymax = max(df["energy"].min() + 0.05, fci_exact + 0.02)

    g = sns.FacetGrid(df, col="noise_model", col_wrap=2,
                      height=3.6, aspect=1.3, sharey=True)
    g.map_dataframe(
        sns.lineplot,
        x="step",
        y="energy",
        hue="Algorithm",
        errorbar=("ci", 95),
    )

    for ax in g.axes.flatten():
        ax.axhline(fci_exact, color="0.4", linestyle=(0, (6, 4)),
                   linewidth=1.0, zorder=1)
        ax.set_xlim(0, df["step"].max())
        ax.set_ylim(ymin, ymax)

    g.add_legend(title="Setup", bbox_to_anchor=(1.02, 0.5), loc="center left")
    g.set_axis_labels("Optimization step", "Energy (Hartree)")
    g.set_titles(col_template="{col_name}", fontweight="semibold")

    _save(g.figure, f"{mol_name}_{basis}_convergence_trajectories")

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


def plot_multi_noise_rollercoasters(db_path):
    df = data_loader.fetch_rollercoaster_data(db_path)
    if df.empty:
        return

    df = df.copy()
    df["noise_label"] = (df["noise_model"]
                         .replace("none", "noiseless")
                         .str.replace("_", " ")
                         .str.title())
    df["ent_label"] = df["ansatz_entanglement"].str.title()

    noise_order = [n for n in
                   ("Noiseless", "Depolarizing", "Amplitude Damping",
                    "Phase Damping", "Thermal Relaxation")
                   if n in df["noise_label"].unique()]

    ent_order = ["Full", "Circular", "Linear"]
    ent_palette = dict(zip(ent_order, sns.color_palette("colorblind", 3)))
    ent_markers = dict(zip(ent_order, ["o", "s", "^"]))

    n = len(noise_order)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(10, 3.2 * nrows),
        sharex=True, sharey=True,
        layout="constrained",
    )
    axes = axes.flatten()

    for idx, noise in enumerate(noise_order):
        ax = axes[idx]
        sub = df[df["noise_label"] == noise]

        for ent in ent_order:
            line = sub[sub["ent_label"] == ent].sort_values("ansatz_layers")
            if line.empty:
                continue
            ax.plot(
                line["ansatz_layers"],
                line["final_energy"],
                marker=ent_markers[ent],
                linestyle="-",
                linewidth=1.6,
                markersize=6,
                color=ent_palette[ent],
                label=ent,
            )

        ax.axhline(STO3G_ENERGY, color="crimson", linestyle="--",
           linewidth=1.0, alpha=0.6, zorder=1)

        ax.set_title(noise, fontweight="semibold", pad=6)
        ax.set_xticks(sorted(df["ansatz_layers"].unique()))
        ax.grid(True, linestyle=":", alpha=0.4)

    # Hide empty axes (if n is odd)
    for ax in axes[n:]:
        ax.set_visible(False)

    # Shared labels
    for ax in axes[n - ncols:]:   # bottom row
        if ax.get_visible():
            ax.set_xlabel("Hardware-efficient layers")

    for ax in axes[::ncols]:      # left column
        if ax.get_visible():
            ax.set_ylabel("Energy (Hartree)")

    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               title="Entanglement",
               loc="outside right upper",
               frameon=False)

    ymin = min(df["final_energy"].min(), STO3G_ENERGY) - 0.005
    ymax = df["final_energy"].max() + 0.005
    axes[0].set_ylim(ymin, ymax)

    _save(fig, "vqe_multi_noise_rollercoasters")
