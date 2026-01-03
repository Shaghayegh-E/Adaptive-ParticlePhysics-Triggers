from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


import mplhep as hep
hep.style.use("CMS")



def add_cms_header(fig, left_x=0.13, right_x=0.90, y=0.95):
    """
    Add 'CMS Open Data' on the left and 'Run 283876' on the right
    in figure coordinates.
    """
    fig.text(
        left_x, y, "CMS Open Data",
        ha="left", va="top",
        fontweight="bold", fontsize=24
    )
    fig.text(
        right_x, y, "Run 283876",
        ha="right", va="top",
        fontsize=24
    )

    
def add_cms_header_ax(ax, left_x=0.0, right_x=1.0, y=1.02):
    ax.text(
        left_x, y,
        "CMS Open Data",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontweight="bold",
        fontsize=24,
        clip_on=False
    )
    ax.text(
        right_x, y,
        "Run 283876",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=24,
        clip_on=False
    )


def save_pdf_png(fig, outbase: Path, dpi=250):
    outbase = Path(outbase)
    fig.savefig(str(outbase.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(outbase.with_suffix(".png")), bbox_inches="tight", dpi=dpi)

# def plot_signal_pass_vs_dim(dim_list, aa_rates, tt_rates, aa_ht, tt_ht, out_path: str):
#     plt.figure(figsize=(8, 6))
#     plt.plot(dim_list, aa_rates, marker='o', linestyle='-', label="HToAATo4B Pass Rate")
#     plt.plot(dim_list, tt_rates, marker='o', linestyle='-', label="ttbar Pass Rate")
#     plt.hlines(aa_ht, min(dim_list), max(dim_list), linestyles="dashed",
#                label=f"HToAATo4B, HT Efficiency: {aa_ht:.2f}%")
#     plt.hlines(tt_ht, min(dim_list), max(dim_list), linestyles="dashed",
#                label=f"TTBar, HT Efficiency: {tt_ht:.2f}%")
#     plt.plot([], [], ' ', label="Threshold = 99.75 Percentile of Test Background")
#     plt.xlabel("Latent Dimension"); plt.ylabel("Signal Pass (%)")
#     plt.grid(True); plt.xlim(min(dim_list)-0.5, max(dim_list)+0.5); plt.ylim(0, 100.5)
#     plt.legend(loc='best', frameon=True)
#     plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_signal_pass_vs_dim(dims, aa_rates, tt_rates, aa_ht_eff, tt_ht_eff, out_path):
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    dims = np.asarray(dims, dtype=float)
    #x = np.log2(dims)
    x = dims

    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, aa_rates, marker="o", label="HToAATo4B (AD)",color="tab:green")
    plt.plot(x, tt_rates, marker="o", label="TTbar (AD)",color="goldenrod")

    # HT baselines as horizontal dashed lines
    plt.axhline(aa_ht_eff, linestyle="--", label=f"HToAATo4B (HT): {aa_ht_eff:.2f}%",color='grey')
    plt.axhline(tt_ht_eff, linestyle="--", label=f"TTbar (HT): {tt_ht_eff:.2f}%",color='grey')

    plt.xlabel("Latent Dimension")
    plt.ylabel("Efficiency (%)")
    plt.grid(True, alpha=0.3)

    # Nice ticks for powers of 2
    #if np.allclose(dims, 2 ** np.round(x)):
        #xticks = np.arange(int(np.floor(x.min())), int(np.ceil(x.max())) + 1)
    #plt.xticks(x)

    plt.legend()
    plt.tight_layout()
    if "data" in out_path:
        add_cms_header(fig, y=0.98)

    plt.savefig(out_path)
    plt.close()

'''
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
        label="HToAAto4B",
    )
    ax.plot(
        dim_list,
        tt_rates,
        marker="o",
        linestyle="-",
        color="goldenrod",
        label="TTBar",
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
''' 

def plot_hist_pair(dim_a, dim_b, res_a: dict, res_b: dict, out_pair: str,
                   out_a: str, out_b: str, n_bins: int = 50):
    import matplotlib.transforms as mtransforms
    def _panel(ax, res, title_dim, xmax):
        thr = np.percentile(res["bkg_scores"], 99.995)
        bins = np.linspace(0, thr, n_bins)
        ax.hist(np.clip(res["bkg_scores"], None, thr), bins=bins, density=True,
                histtype='step', label='test Background', linewidth=2)
        ax.hist(np.clip(res["AA_scores"], None, thr), bins=bins, density=True,
                histtype='step', label='HToAATo4B Signal', linewidth=2)
        ax.hist(np.clip(res["TT_scores"], None, thr), bins=bins, density=True,
                histtype='step', label='TTbar Signal', linewidth=2)
        ax.set_xlabel('Anomaly Score'); ax.set_ylabel('Density')
        ax.set_yscale('log'); ax.set_xlim(-0.1, xmax); ax.legend(
            loc='best', frameon=True, title=f'Latent Dimension = {title_dim}\n(Overflows in the Last Bin)')
        if "data" in out_pair:
            add_cms_header_ax(ax)
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



def plot_hist_for_dim(
    d: int,
    bkg_scores: np.ndarray,
    tt_scores: np.ndarray,
    aa_scores: np.ndarray,
    thr: float,
    outbase: Path,
    title_suffix: str,
):
    #plot histograms of the scores for given dim d for bkg, tt, aa
    fig, ax = plt.subplots(figsize=(10, 6))
    threshold1 = np.percentile(bkg_scores,99.995)
    #range=(0, threshold)

    # overflow bin
    n_bins = 50
    bins1 = np.linspace(0, threshold1, n_bins)

    # Clip scores: anything above threshold1 goes into threshold1
    bkg_scores = np.clip(bkg_scores, None, threshold1)
    aa_scores  = np.clip(aa_scores,  None, threshold1)
    tt_scores  = np.clip(tt_scores,  None, threshold1)


    ax.hist(bkg_scores, bins=bins1, histtype="step", linewidth=2.2, density=True, label="Background (test)", color='tab:blue')
    ax.hist(tt_scores,  bins=bins1, histtype="step", linewidth=2.2, density=True, label="TTbar",color='goldenrod')
    ax.hist(aa_scores,  bins=bins1, histtype="step", linewidth=2.2, density=True, label="HToAATo4B",color='tab:green')

    ax.axvline(thr, linestyle="--", linewidth=2.0, label=f"thr (99.75%) = {thr:.4g}")

    ax.set_xlabel("Anomaly score (reco MSE)")
    ax.set_ylabel("Density")
    ax.set_yscale('log')
    ax.set_xlim(0,threshold1+100)
    ax.set_ylim(10E-8,1000)
    #ax.set_title(f"AE dim={d} anomaly score distributions {title_suffix}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=f"AE dim={d}\nOverflow in Last Bin",frameon=True)
    if "data" in outbase:
        add_cms_header(fig)

    save_pdf_png(fig, outbase)
    plt.close(fig)