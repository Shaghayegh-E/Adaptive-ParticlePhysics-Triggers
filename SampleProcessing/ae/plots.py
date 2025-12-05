from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

try:
    import atlas_mpl_style as aplt
    aplt.use_atlas_style()
except Exception:
    pass


def plot_signal_pass_vs_dim(dim_list, aa_rates, tt_rates, aa_ht, tt_ht, out_path: str):
    plt.figure(figsize=(8, 6))
    plt.plot(dim_list, aa_rates, marker='o', linestyle='-', label="HToAATo4B Pass Rate")
    plt.plot(dim_list, tt_rates, marker='o', linestyle='-', label="ttbar Pass Rate")
    plt.hlines(aa_ht, min(dim_list), max(dim_list), linestyles="dashed",
               label=f"HToAATo4B, HT Efficiency: {aa_ht:.2f}%")
    plt.hlines(tt_ht, min(dim_list), max(dim_list), linestyles="dashed",
               label=f"TTBar, HT Efficiency: {tt_ht:.2f}%")
    plt.plot([], [], ' ', label="Threshold = 99.75 Percentile of Test Background")
    plt.xlabel("Latent Dimension"); plt.ylabel("Signal Pass (%)")
    plt.grid(True); plt.xlim(min(dim_list)-0.5, max(dim_list)+0.5); plt.ylim(0, 100.5)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_signal_pass_vs_dim_data(
    dim_list,
    aa_rates,
    tt_rates,
    aa_ht_eff,
    tt_ht_eff,
    out_prefix: str = "outputs/signal_pass_vs_dimension_data"
):
    """
    Replicates the 'meeting mode' signal-pass-vs-dimension plot,
    but as a reusable function.

    Parameters
    ----------
    dim_list : list[int] or array-like
        Latent dimensions (x-axis).
    aa_rates : list[float]
        HToAATo4B pass rates (%) per dimension.
    tt_rates : list[float]
        TTBar pass rates (%) per dimension.
    aa_ht_eff : float
        HT-only efficiency (%) for HToAATo4B at the chosen threshold.
    tt_ht_eff : float
        HT-only efficiency (%) for TTBar at the chosen threshold.
    out_prefix : str
        Path prefix for output; will write out_prefix + '.pdf' and '.png'.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # Lines for AA and TT as function of latent dimension
    ax.plot(
        dim_list,
        aa_rates,
        marker="o",
        linestyle="-",
        color="tab:green",
        label="HToAAto4B Pass Rate",
    )
    ax.plot(
        dim_list,
        tt_rates,
        marker="o",
        linestyle="-",
        color="goldenrod",
        label="TTBar Pass Rate",
    )

    # Horizontal HT-efficiency lines (meeting-mode style)
    x_min, x_max = min(dim_list), max(dim_list)
    ax.hlines(
        aa_ht_eff,
        x_min,
        x_max,
        color="grey",
        linestyles="dashed",
        label=f"HToAATo4B, HT Efficiency: {aa_ht_eff:.2f}%",
    )
    ax.hlines(
        tt_ht_eff,
        x_min,
        x_max,
        color="grey",
        linestyles="dashed",
        label=f"TTBar, HT Efficiency: {tt_ht_eff:.2f}%",
    )

    # Dummy legend entry for the threshold description
    ax.plot([], [], " ", label="Threshold = 99.75 Percentile of Test Background")

    ax.set_xlabel("Latent Dimension", loc="center")
    ax.set_ylabel("Signal Pass (%)", loc="center")
    ax.grid(True)

    # Match your original "meeting mode" style
    ax.set_xlim(x_min - 0.05, x_max + 1.0)
    ax.set_ylim(0, 200)  # keep the generous 0–200 range like your original
    ax.set_xticks(dim_list)

    ax.legend(fontsize=14, frameon=True, loc="upper right", bbox_transform=ax.transData)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png")
    plt.close(fig)


def plot_hist_pair(dim_a, dim_b, res_a: dict, res_b: dict, out_pair: str,
                   out_a: str, out_b: str, n_bins: int = 50):
    import matplotlib.transforms as mtransforms
    def _panel(ax, res, title_dim, xmax):
        thr = np.percentile(res["bkg_scores"], 99.995)
        bins = np.linspace(0, thr, n_bins)
        ax.hist(np.clip(res["bkg_scores"], None, thr), bins=bins, density=True,
                histtype='step', label='test MinBias Background', linewidth=2)
        ax.hist(np.clip(res["AA_scores"], None, thr), bins=bins, density=True,
                histtype='step', label='HToAATo4B Signal', linewidth=2)
        ax.hist(np.clip(res["TT_scores"], None, thr), bins=bins, density=True,
                histtype='step', label='TTbar Signal', linewidth=2)
        ax.set_xlabel('Anomaly Score'); ax.set_ylabel('Density')
        ax.set_yscale('log'); ax.set_xlim(-0.1, xmax); ax.legend(
            loc='best', frameon=True, title=f'Latent Dimension = {title_dim}\n(Overflows in the Last Bin)')
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    _panel(axs[0], res_a, dim_a, xmax=600)
    _panel(axs[1], res_b, dim_b, xmax=120)
    plt.tight_layout(); fig.savefig(out_pair)

    def save_subplot(fig, ax, filename, pad=0.3):
        fig.canvas.draw()
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        bbox = mtransforms.Bbox.from_extents(bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad)
        fig.savefig(filename, bbox_inches=bbox)

    save_subplot(fig, axs[0], out_a, pad=0.3)
    save_subplot(fig, axs[1], out_b, pad=0.3)
    plt.close(fig)
