import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def cummean(x):
    x = np.asarray(x, dtype=np.float64)
    return np.cumsum(x) / (np.arange(len(x)) + 1.0)

def rel_to_t0(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    return x / max(1e-12, float(x[0]))




# ------------------------- plotting helpers -------------------------
def add_cms_header(fig, left_x=0.13, right_x=0.90, y=0.94, run_label="Run 283408"):
    fig.text(left_x, y, "CMS Open Data",
             ha="left", va="top", fontweight="bold", fontsize=24)
    fig.text(right_x, y, run_label,
             ha="right", va="top", fontsize=24)

def save_pdf_png(fig, basepath, dpi_png=300):
    Path(basepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{basepath}.pdf", bbox_inches="tight")
    fig.savefig(f"{basepath}.png", bbox_inches="tight", dpi=dpi_png)

def plot_rate_with_tolerance(
    time,
    y_const,
    y_pd,
    y_dqn,
    outbase,
    *,
    run_label="Run 283408",
    legend_title="HT Trigger",
    ylim=(0, 200),
    ylabel="Background Rate [kHz]",
    xlabel="Time (Fraction of Run)",
    tol_upper=110.0,
    tol_lower=90.0,
    grid_alpha=0.6,
    # styles (match your paper family)
    const_style=dict(color="tab:blue", linestyle="dashed", linewidth=3.0),
    pd_style=dict(color="mediumblue", linestyle="solid", linewidth=2.5),
    dqn_style=dict(color="tab:purple", linestyle="solid", linewidth=2.5),
    const_label="Constant Menu",
    pd_label="PD Controller",
    dqn_label="DQN",
    add_cms_header=None,
    save_pdf_png=None,
    dpi_png=300,
):
    """
    Plot background rate vs time with:
      - main legend: Constant/PD/DQN under legend_title
      - reference legend: Upper/Lower tolerance lines
    outbase: path WITHOUT extension (str or Path)
    add_cms_header(fig, run_label=...) and save_pdf_png(fig, basepath, dpi_png=...)
      are passed in (so utils doesn't need to import them from your script).
    """
    time = np.asarray(time)
    y_const = np.asarray(y_const)
    y_pd = np.asarray(y_pd)
    y_dqn = np.asarray(y_dqn)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, y_const, **const_style)
    ax.plot(time, y_pd,    **pd_style)
    ax.plot(time, y_dqn,   **dqn_style)

    ax.axhline(y=tol_upper, color="gray", linestyle="--", linewidth=1.5)
    ax.axhline(y=tol_lower, color="gray", linestyle="--", linewidth=1.5)

    ax.set_xlabel(xlabel, loc="center")
    ax.set_ylabel(ylabel, loc="center")
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=grid_alpha)

    # ---- Main legend (Constant/PD/DQN) ----
    h_const = mlines.Line2D([], [], **const_style)
    h_pd    = mlines.Line2D([], [], **pd_style)
    h_dqn   = mlines.Line2D([], [], **dqn_style)

    leg_main = ax.legend(
        [h_const, h_pd, h_dqn],
        [const_label, pd_label, dqn_label],
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fontsize=14,
    )
    ax.add_artist(leg_main)

    # ---- Reference legend (tolerances) ----
    upper = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    lower = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    ax.legend(
        [upper, lower],
        [f"Upper Tolerance ({int(tol_upper)})", f"Lower Tolerance ({int(tol_lower)})"],
        title="Reference",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fontsize=14,
    )

    if add_cms_header is not None:
        # your add_cms_header(fig, run_label=...)
        add_cms_header(fig, run_label=run_label)

    if save_pdf_png is not None:
        save_pdf_png(fig, str(outbase), dpi_png=dpi_png)

    plt.close(fig)



def save_png(fig, basepath, dpi_png=300):
    """Save figure as PNG only. basepath: path without extension."""
    p = Path(basepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p.with_suffix(".png")), bbox_inches="tight", dpi=dpi_png)