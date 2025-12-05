#!/usr/bin/env python3
"""
ht_time_evolution.py
Fig 2:
Compute the scalar sum of jet pT (HT) from an HDF5 file and study its
time evolution over a run. The script produces three plots:

1. HT histogram in time slices (NumEvents_vs_HT.png)
2. HT distribution vs. time as violin plots (HT_violin.png)
3. HT distribution vs. time as box plots (HT_box.png)

Expected HDF5 structure:
  - Datasets j0Eta, j1Eta, ..., j7Eta
  - Datasets j0Pt,  j1Pt,  ..., j7Pt

HT is computed as:
  HT = sum_j pt_j  for jets with pt_j > 20 GeV and |eta_j| < 2.5

Usage:
  python ht_time_evolution.py --input Data/data_Run_2016_283408_longest.h5 \
                              --output-dir outputs \
                              --n-time-bins 10
"""

import argparse
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def load_and_calculate_ht(filename: str,
                          n_jets: int = 8,
                          pt_min: float = 20.0,
                          eta_max: float = 2.5) -> np.ndarray:
    """
    Load jets from an HDF5 file and compute HT per event.

    Parameters
    ----------
    filename : str
        Path to the input HDF5 file.
    n_jets : int, optional
        Number of jets stored as j0, j1, ..., j{n_jets-1}. Default is 8.
    pt_min : float, optional
        Minimum jet pT threshold for HT (GeV). Default is 20.
    eta_max : float, optional
        Maximum |eta| for jets to be included in HT. Default is 2.5.

    Returns
    -------
    ht_values : np.ndarray
        Array of HT values for all events (only events with HT > 0 are kept).
    """
    with h5py.File(filename, "r") as f:
        n_events = f["j0Eta"].shape[0]
        ht_values = np.zeros(n_events)

        for i in range(n_jets):
            eta = f[f"j{i}Eta"][:]
            pt = f[f"j{i}Pt"][:]
            mask = (pt > pt_min) & (np.abs(eta) < eta_max)
            ht_values += pt * mask

    # Keep only events with non-zero HT
    # ht_values = ht_values[ht_values > 0]
    return ht_values


def make_histogram(ht_values: np.ndarray,
                   output_path: Path,
                   n_time_bins: int = 10,
                   ht_min: float = 25.0,
                   ht_max: float = 300.0,
                   n_ht_bins: int = 20) -> None:
    """
    Plot HT histograms for different slices of the run and save as PNG.

    Parameters
    ----------
    ht_values : np.ndarray
        HT values for all events.
    output_path : Path
        Where to save the PNG file.
    n_time_bins : int
        Number of time slices across the run.
    ht_min, ht_max : float
        HT range for histogram x-axis.
    n_ht_bins : int
        Number of HT bins.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=1)
    ht_bins = np.linspace(ht_min, ht_max, n_ht_bins + 1)


    for i in range(n_time_bins):#n_time_bins):
        start, end = int(i*len(ht_values)/n_time_bins), int((i+1)*len(ht_values)/n_time_bins) #lenghtHtmanual/10
        chunk = ht_values[start:end]
        frac = i / n_time_bins
        color = cmap(norm(frac))
        #if (i in [0,3,5,7,9]):
        ax.hist(chunk, bins=ht_bins, histtype="step", 
        color=color, linewidth=1.5)

    ax.set_xlabel("HT",loc='center')#, fontsize=14)
    ax.set_ylabel("Number of Events",loc='center')#, fontsize=14)
    ax.set_yscale('log')
    ax.set_xlim(ht_min, ht_max)
    #ax1.tick_params(labelsize=13)
    ax.grid(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar1 = fig.colorbar(sm, ax=ax)
    cbar1.set_label("Time (Fraction of Run)",loc='center')#, fontsize=13)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_violin_plot(ht_values: np.ndarray,
                     output_path: Path,
                     n_time_bins: int = 10,
                     ht_min: float = 20.0,
                     ht_max: float = 175.0) -> None:
    """
    Plot HT violin plots across time bins.

    Parameters
    ----------
    ht_values : np.ndarray
        HT values for all events.
    output_path : Path
        Where to save the PNG file.
    n_time_bins : int
        Number of time bins for the x-axis.
    ht_min, ht_max : float
        y-axis limits for HT.
    """
    
    time_fraction = np.linspace(0, 1, len(ht_values), endpoint=False)
    df = pd.DataFrame({"HT": ht_values, "time_frac": time_fraction})
    df["time_bin"] = pd.cut(df["time_frac"], bins=n_time_bins)

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.violinplot(
        x="time_bin",
        y="HT",
        data=df,
        inner=None,
        cut=0,
        bw=0.4,
        palette="viridis",
        ax=ax,
    )

    # Draw quartiles Q1, median Q2, Q3
    median_color = "#4e8b50"
    quartile_color = "#d98c4c"
    for i, (_, group) in enumerate(df.groupby("time_bin")):
        q1 = group["HT"].quantile(0.25)
        q2 = group["HT"].quantile(0.50)
        q3 = group["HT"].quantile(0.75)
        ax.hlines([q1], i - 0.25, i + 0.25, color=quartile_color, linewidth=2)
        ax.hlines([q2], i - 0.25, i + 0.25, color=median_color, linewidth=2)
        ax.hlines([q3], i - 0.25, i + 0.25, color=quartile_color, linewidth=2)

    legend_elements = [
        Line2D([0], [0], color=quartile_color, lw=2, label="Q1 / Q3"),
        Line2D([0], [0], color=median_color, lw=2, label="Median"),
    ]
    ax.legend(handles=legend_elements, title="Quartiles", loc="upper right", frameon=True)

    ax.set_xlabel("Time (Fraction of Run)", loc='center')#, fontsize=14)
    ax.set_ylabel("HT",loc='center')#, fontsize=14)
    ax.set_ylim(ht_min, ht_max)
    ax.set_xticks(np.arange(n_time_bins))
    ax.set_xticklabels(
        [f"{b.right:.1f}" for b in df["time_bin"].cat.categories]
    )

    ax.tick_params(axis="both", which="major")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_box_plot(ht_values: np.ndarray,
                  output_path: Path,
                  n_time_bins: int = 10,
                  ht_min: float = 20.0,
                  ht_max: float = 300.0) -> None:
    """
    Plot HT box plots across time bins.

    Parameters
    ----------
    ht_values : np.ndarray
        HT values for all events.
    output_path : Path
        Where to save the PNG file.
    n_time_bins : int
        Number of time bins for the x-axis.
    ht_min, ht_max : float
        y-axis limits for HT.
    """
    time_fraction = np.linspace(0, 1, len(ht_values), endpoint=False)
    df = pd.DataFrame({"HT": ht_values, "time_frac": time_fraction})
    df["time_bin"] = pd.cut(df["time_frac"], bins=n_time_bins)

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.boxplot(
        x="time_bin",
        y="HT",
        data=df,
        hue="time_bin",
        palette="viridis",
        legend=False,
        width=0.6,
        showfliers=True,
        fliersize=1,
        flierprops={"marker": "o", "alpha": 0.3, "color": "gray"},
        ax=ax,
    )

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("HT [GeV]", loc="center")
    ax.set_ylim(ht_min, ht_max)

    ax.set_xticks(np.arange(n_time_bins))
    labels = [f"{b.right:.1f}" for b in df["time_bin"].cat.categories]
    ax.set_xticklabels(labels)

    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study the time evolution of HT from jet-level HDF5 data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Data/data_Run_2016_283408_longest.h5",
        help="Path to input HDF5 file with jet branches j{i}Eta, j{i}Pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save plots (default: outputs).",
    )
    parser.add_argument(
        "--n-time-bins",
        type=int,
        default=10,
        help="Number of time bins across the run (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading jets from: {input_path}")
    ht_values = load_and_calculate_ht(str(input_path))
    ht_values = ht_values[ht_values > 0]
    print(f"[INFO] Loaded {len(ht_values)} events with HT > 0")

    # Histogram over time slices
    make_histogram(
        ht_values,
        output_path=out_dir / "NumEvents_vs_HT.png",
        n_time_bins=args.n_time_bins,
    )
    print(f"[INFO] Saved histogram to {out_dir / 'NumEvents_vs_HT.png'}")

    # Violin plot
    make_violin_plot(
        ht_values,
        output_path=out_dir / "HT_violin.png",
        n_time_bins=args.n_time_bins,
    )
    print(f"[INFO] Saved violin plot to {out_dir / 'HT_violin.png'}")

    # Box plot
    make_box_plot(
        ht_values,
        output_path=out_dir / "HT_box.png",
        n_time_bins=args.n_time_bins,
    )
    print(f"[INFO] Saved box plot to {out_dir / 'HT_box.png'}")


if __name__ == "__main__":
    main()
