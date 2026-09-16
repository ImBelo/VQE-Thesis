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


# Human-readable labels for raw config values (fixes acronym casing)
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
    """Map raw config strings to display labels, falling back to str.title()."""
    return series.map(lambda v: LABELS.get(v, str(v).replace("_", " ").title()))


def _save(fig, stem):
    """Save both PDF (vector) and PNG (raster) with tight bounding box."""
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mapping delta plot
# ---------------------------------------------------------------------------
def plot_mapping_delta(db_path, mol_name="H2", basis="sto-3g"):
    df = data_loader.fetch_mapping_delta_data(db_path, mol_name, basis)
    if df.empty:
        return

    # NOTE: this pivot collapses the 5 repetitions by taking the mean across
    # them, because a pivot_table without an explicit aggfunc averages
    # duplicates. If you want per-repetition deltas, add `run_id` to the index.
    df_pivot = (
        df.pivot_table(
            index=["step", "ansatz_type", "noise_model", "opt_type"],
            columns="mapping",
            values="energy",
        )
        .reset_index()
        .dropna(subset=["jordan_wigner", "bravyi_kitaev"])
    )

    df_pivot["Energy_Delta"] = df_pivot["jordan_wigner"] - df_pivot["bravyi_kitaev"]
    df_pivot["Algorithm"] = (
        _pretty(df_pivot["ansatz_type"]) + " + " + _pretty(df_pivot["opt_type"])
    )
    df_pivot["noise_label"] = _pretty(df_pivot["noise_model"])

    algorithms = sorted(df_pivot["Algorithm"].unique())
    palette = dict(
        zip(algorithms, sns.color_palette("colorblind", len(algorithms)))
    )

    g = sns.FacetGrid(
        df_pivot,
        col="noise_label",
        col_wrap=2,
        height=4,
        aspect=1.4,
        sharey=True,
        hue="Algorithm",
        hue_order=algorithms,
        palette=palette,
    )
    g.map_dataframe(
        sns.lineplot, x="step", y="Energy_Delta", linewidth=1.8, alpha=0.9
    )

    # Reference line at zero — clean, not competing with the data colors
    for ax in g.axes.flatten():
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0,
                   zorder=1, alpha=0.7)
        # Annotate in axes coordinates so the text sits in the corner
        # regardless of the data range.
        ax.text(0.98, 0.95, "BK wins", transform=ax.transAxes,
                ha="right", va="top", color="black", fontsize=9, alpha=0.6)
        ax.text(0.98, 0.05, "JW wins", transform=ax.transAxes,
                ha="right", va="bottom", color="black", fontsize=9, alpha=0.6)
        ax.set_xlim(0, df_pivot["step"].max())

    g.add_legend(title="Setup", bbox_to_anchor=(1.02, 0.5), loc="center left")
    g.set_axis_labels("Optimization Step",
                      r"$\Delta$ Energy: $E_{JW} - E_{BK}$ (Hartree)")
    g.set_titles(col_template="{col_name}", fontweight="semibold")

    _save(g.figure, f"{mol_name}_{basis}_mapping_delta_analysis")
    print(f"[+] Saved mapping delta plot: "
          f"{mol_name}_{basis}_mapping_delta_analysis")


# ---------------------------------------------------------------------------
# Noise resilience plot
# ---------------------------------------------------------------------------
def plot_noise_resilience(db_path, mol_name="H2", basis="sto-3g"):
    df = data_loader.fetch_noise_resilience_data(db_path, mol_name)
    if df.empty:
        return

    df = df.copy()
    df["ansatz_label"] = _pretty(df["ansatz_type"])
    df["mapping_label"] = _pretty(df["mapping"])
    df["noise_label"] = _pretty(df["noise_model"])

    SHORT = {
        "Amplitude Damping":  "Amp. Damp.",
        "Phase Damping":      "Phase Damp.",
        "Thermal Relaxation": "Therm. Relax.",
    }
    df["noise_label"] = df["noise_label"].map(SHORT).fillna(df["noise_label"])

    noise_order = [
        SHORT.get(LABELS[k], LABELS[k])
        for k in ("none", "depolarizing", "amplitude_damping",
                  "phase_damping", "thermal_relaxation")
        if SHORT.get(LABELS[k], LABELS[k]) in df["noise_label"].unique()
    ]

    ansatz_order = sorted(df["ansatz_label"].unique())
    mapping_order = sorted(df["mapping_label"].unique())

    palette = dict(zip(ansatz_order,
                       sns.color_palette("colorblind", len(ansatz_order))))

    # Distinct marker + linestyle per mapping so they don't visually collapse
    mapping_style = {
        "Bravyi\u2013Kitaev": {"marker": "o", "linestyle": "-",  "mew": 1.2},
        "Jordan\u2013Wigner": {"marker": "X", "linestyle": "--", "mew": 1.2},
    }

    x_pos = {n: i for i, n in enumerate(noise_order)}

    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")

    for (ansatz, mapping), group in df.groupby(
        ["ansatz_label", "mapping_label"], sort=False
    ):
        group = group.set_index("noise_label").reindex(noise_order).reset_index()
        xs = [x_pos[n] for n in group["noise_label"]]

        style = mapping_style.get(mapping, {"marker": "o", "linestyle": "-"})

        ax.errorbar(
            x=xs,
            y=group["final_energy"],
            yerr=group["std_energy"].fillna(0),
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=palette[ansatz],
            linewidth=1.8,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=style.get("mew", 1.0),
            capsize=3,
            label=f"{ansatz} \u2014 {mapping}",
        )

    ax.axhline(STO3G_ENERGY, color="0.4", linestyle=(0, (6, 4)),
               linewidth=1.0, zorder=1)

    ymin = min(df["final_energy"].min() - df["std_energy"].fillna(0).max(),
               STO3G_ENERGY) - 0.005
    ymax = max(df["final_energy"].max() + df["std_energy"].fillna(0).max(),
               STO3G_ENERGY) + 0.005
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(range(len(noise_order)))
    ax.set_xticklabels(noise_order)

    ax.set_title(
        f"Ansatz and mapping resilience vs. noise channels "
        f"({mol_name} / {basis.upper()})",
        fontweight="semibold", pad=10,
    )
    ax.set_xlabel("Noise environment")
    ax.set_ylabel("Final converged energy (Hartree)")

    ax.legend(title="Ansatz / Mapping",
              bbox_to_anchor=(1.02, 0.5),
              loc="center left",
              frameon=False)

    _save(fig, f"{mol_name}_noise_mapping_resilience_benchmark")
