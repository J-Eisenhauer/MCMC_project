"""
Plotting script for the stripe-phase MC project.

Reads the CSV/NPZ files written by mc_fixed_temperature.py and
mc_temperature_drop_parallel.py and creates every figure for the report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import numpy as np


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 9,
})

# Must match the MC scripts
N = 250
SIGMA_0, SIGMA_1 = 1.0, 2.5
GREEN = plt.cm.tab10(2)

FIXED_DENSITIES = (0.100, 0.150, 0.227, 0.291, 0.380)
FIXED_DIR = Path("fixed_temperature_results")

DROP_DENSITY = 0.291
DROP_DIR = Path("temperature_drop_results_parallel")
SNAPSHOT_TEMPERATURES = (0.25, 0.20, 0.19, 0.18, 0.15)


# --------------------- particles drawn to true size + aura -----------------
def with_periodic_images(x, y, box_length, margin):
    """Add periodic copies near the box edges so wrapped particles are whole."""
    xs, ys = [x], [y]
    for shift_x in (-1, 0, 1):
        for shift_y in (-1, 0, 1):
            if shift_x == 0 and shift_y == 0:
                continue
            xi, yi = x + shift_x * box_length, y + shift_y * box_length
            keep = (xi > -margin) & (xi < box_length + margin) & (yi > -margin) & (yi < box_length + margin)
            xs.append(xi[keep])
            ys.append(yi[keep])
    return np.concatenate(xs), np.concatenate(ys)


def draw_particles(ax, x, y, box_length, color, title):
    """
    Draw every particle to scale: a solid core (diameter sigma_0) inside a
    translucent aura (diameter sigma_1).
    """
    xs, ys = with_periodic_images(x, y, box_length, SIGMA_1)
    offsets = np.column_stack((xs, ys))

    aura = EllipseCollection(
        np.full(xs.size, SIGMA_1), np.full(xs.size, SIGMA_1), 0,
        units="xy", offsets=offsets, offset_transform=ax.transData,
        facecolors=(*color[:3], 0.15), edgecolors=(*color[:3], 0.4), linewidths=0.5)
    core = EllipseCollection(
        np.full(xs.size, SIGMA_0), np.full(xs.size, SIGMA_0), 0,
        units="xy", offsets=offsets, offset_transform=ax.transData,
        facecolors=color, edgecolors="none")

    # Clip to the simulation box so the periodic images outside don't show.
    clip = plt.Rectangle((0, 0), box_length, box_length, transform=ax.transData)
    clip.set_visible(False)
    ax.add_patch(clip)
    for collection in (aura, core):
        ax.add_collection(collection)
        collection.set_clip_path(clip)

    ax.plot([0, box_length, box_length, 0, 0], [0, 0, box_length, box_length, 0],
            color="0.4", linewidth=0.8)
    ax.set_xlim(-0.02 * box_length, 1.02 * box_length)
    ax.set_ylim(-0.02 * box_length, 1.02 * box_length)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/\sigma_0$")
    ax.set_ylabel(r"$y/\sigma_0$")
    ax.set_title(title)


# ------------------------- fixed-temperature plots --------------------------
def plot_equilibration_by_density():
    """Energy per particle vs. attempted moves, one panel per density."""
    fig, axes = plt.subplots(1, len(FIXED_DENSITIES), figsize=(3.0 * len(FIXED_DENSITIES), 4.2),
                             sharex=True, sharey=True)
    for ax, density in zip(axes, FIXED_DENSITIES):
        steps, energy = np.loadtxt(FIXED_DIR / f"rho_{density:.3f}" / "equilibration_energy.csv",
                                   delimiter=",", skiprows=1, unpack=True)
        ax.plot(steps, energy, linewidth=1.8, color=GREEN)
        ax.set_title(rf"$\rho^*={density:.3f}$")
        ax.set_xlabel("attempted moves")
    axes[0].set_ylabel(r"$U/N$")
    fig.tight_layout()
    fig.savefig(FIXED_DIR / "equilibration_energy.png")
    plt.close(fig)


def plot_structures_and_rdf():
    """Equilibrium structures (row 1) and g(r) (row 2), one column per density."""
    fig, axes = plt.subplots(2, len(FIXED_DENSITIES), figsize=(3.2 * len(FIXED_DENSITIES), 7.0))

    for idx, density in enumerate(FIXED_DENSITIES):
        directory = FIXED_DIR / f"rho_{density:.3f}"
        box_length = np.sqrt(N / density)

        x, y = np.loadtxt(directory / "final_coordinates.csv", delimiter=",", skiprows=1, unpack=True)
        draw_particles(axes[0, idx], x, y, box_length, GREEN, rf"$\rho^*={density:.3f}$")

        r, g_r = np.loadtxt(directory / "rdf.csv", delimiter=",", skiprows=1, unpack=True)
        ax = axes[1, idx]
        ax.plot(r, g_r, color=GREEN, linewidth=1.8)
        ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
        ax.axvline(SIGMA_0, color="0.3", linestyle=":", linewidth=1.8)
        ax.axvline(SIGMA_1, color="0.3", linestyle="-.", linewidth=1.8)
        ax.set_xlabel(r"$r/\sigma_0$")

    axes[1, 0].set_ylabel(r"$g(r)$")
    fig.tight_layout()
    fig.savefig(FIXED_DIR / "equilibrium_structures_and_rdf.png")
    plt.close(fig)


# ------------------------- temperature-drop plots ---------------------------
def load_chain_results():
    """Load all_chains_results.csv into {chain index: structured row array}."""
    table = np.genfromtxt(DROP_DIR / "all_chains_results.csv", delimiter=",", names=True)
    return {chain: table[table["chain"] == chain] for chain in np.unique(table["chain"]).astype(int)}


def plot_seed0_configurations(box_length):
    """Cooling sequence of realization 1 (seed_0), one panel per saved temperature."""
    chain_dir = DROP_DIR / "chain_seed_0"
    available = np.array(sorted(float(f.stem.split("_T_")[1]) for f in chain_dir.glob("prod_T_*.npz")))

    fig, axes = plt.subplots(1, len(SNAPSHOT_TEMPERATURES),
                             figsize=(3.2 * len(SNAPSHOT_TEMPERATURES), 3.6))

    for ax, target in zip(axes, SNAPSHOT_TEMPERATURES):
        temperature = available[np.argmin(np.abs(available - target))]
        with np.load(chain_dir / f"prod_T_{temperature:.3f}.npz") as d:
            x, y = d["x"], d["y"]
        draw_particles(ax, x, y, box_length, GREEN, rf"$T^*={temperature:.2f}$")

    fig.tight_layout()
    fig.savefig(DROP_DIR / "configurations_realisation1.png")
    plt.close(fig)


def draw_chain_comparison(ax, data, column, ylabel):
    """
    Every realization as a thin dashed line, plus mean +- standard error
    across all realizations as a thicker black line.
    """
    chains = sorted(data.keys())

    for idx, chain in enumerate(chains):
        rows = data[chain]
        ax.plot(rows["temperature"], rows[column], linestyle="--", marker="o",
                markersize=3, linewidth=0.6, alpha=0.6, color=plt.cm.tab10(idx),
                label=f"Realisation {idx + 1}")

    stacked = np.array([data[chain][column] for chain in chains])
    temperatures = data[chains[0]]["temperature"]
    mean = stacked.mean(axis=0)
    standard_error = stacked.std(axis=0, ddof=1) / np.sqrt(len(chains))
    ax.errorbar(temperatures, mean, yerr=standard_error, fmt="o-", color="black",
                markersize=6, linewidth=1.6, capsize=3, zorder=5, label="Mean ± SEM")

    ax.set(xlabel=r"Temperature $T^*$", ylabel=ylabel)


def plot_energy_and_cv(data):
    """Heat capacity and energy per particle side by side, both vs. temperature."""
    fig, (ax_cv, ax_u) = plt.subplots(1, 2, figsize=(12, 4.5),
                                      gridspec_kw={"width_ratios": [2, 1]})

    draw_chain_comparison(ax_cv, data, "heat_capacity_per_particle", r"Heat capacity $C_V/(Nk)$")
    draw_chain_comparison(ax_u, data, "mean_energy_per_particle", r"Energy per particle $U/N$")
    ymin, ymax = ax_u.get_ylim()
    ax_u.set_yticks(np.arange(np.ceil(ymin * 10) / 10, ymax, 0.1))
    ax_cv.legend(loc="upper left", ncol=2, fontsize=7)

    fig.tight_layout()
    fig.savefig(DROP_DIR / "energy_and_cv_vs_temperature.png")
    plt.close(fig)

plot_equilibration_by_density()
plot_structures_and_rdf()
plot_seed0_configurations(np.sqrt(N / DROP_DENSITY))
plot_energy_and_cv(load_chain_results())
print(f"All plots done.")
