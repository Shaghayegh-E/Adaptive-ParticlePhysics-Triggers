import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import mplhep as hep
hep.style.use("CMS")

def add_cms_header(fig, left_x=0.13, right_x=0.90, y=0.95):
    """
    Add 'CMS Open Data' on the left and 'Run 283408' on the right
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

    
def efficiency_vs_pt(scores, pts, cut, bins):
    eff = []
    bin_centers = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (pts >= lo) & (pts < hi)
        if np.sum(mask) == 0:
            eff.append(np.nan)
        else:
            eff.append(np.sum(scores[mask] > cut) / np.sum(mask))
        bin_centers.append(0.5*(lo+hi))
    return np.array(bin_centers), np.array(eff)


def save_subplot(fig, ax, filename, pad=0.1):
    """Save a single subplot with a little padding (in inches)."""
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    # Expand bbox by pad (fraction of inch)
    bbox = mtransforms.Bbox.from_extents(
        bbox.x0 - pad, bbox.y0 - pad,
        bbox.x1 + pad, bbox.y1 + pad
    )

    fig.savefig(filename, bbox_inches=bbox)

base_dir = "SampleProcessing/models"

# load from ANY fold / dim (HT threshold is global)
#example_1_mc = np.load(os.path.join(base_dir_mc, "fold_0", "scores_dim_1.npz"))
example_2_mc = np.load(os.path.join(base_dir, "scores_autoencoder_model_mc_2.npz"))
#example_3_mc = np.load(os.path.join(base_dir_mc, "fold_0", "scores_dim_4.npz"))


# load from ANY fold / dim (HT threshold is global)
#example_1_data = np.load(os.path.join(base_dir_data, "fold_0", "scores_dim_1.npz"))
example_2_data = np.load(os.path.join(base_dir, "scores_autoencoder_model_realdata_2.npz"))
#example_3_data = np.load(os.path.join(base_dir_data, "fold_0", "scores_dim_4.npz"))


bkg_1_mc = example_2_mc["bkg"]
AA_1_mc = example_2_mc["AA"]
TT_1_mc = example_2_mc["TT"]

bkg_1_data = example_2_data["bkg"]
AA_1_data = example_2_data["AA"]
TT_1_data = example_2_data["TT"]

'''
bkg_2 = example_2_mc["bkg"]
AA_2 = example_2_mc["AA"]
TT_2 = example_2_mc["TT"]

bkg_3 = example_3_mc["bkg"]
AA_3 = example_3_mc["AA"]
TT_3 = example_3_mc["TT"]

'''
HTAA_1_mc = example_2_mc["AAHT"]
HTtt_1_mc = example_2_mc["TTHT"]
bkgHT_1_mc = example_2_mc["bkgHT"]

HTAA_1_data = example_2_data["AAHT"]
HTtt_1_data = example_2_data["TTHT"]
bkgHT_1_data = example_2_data["bkgHT"]


# HT-only cut (global, from background)
cut_HT_mc = np.percentile(bkgHT_1_mc, 99.75)
cut_HT_data = np.percentile(bkgHT_1_data, 99.75)

# HT-only efficiencies
aa_pt_mc, aa_effht_mc = efficiency_vs_pt(
    scores=HTAA_1_mc,
    pts=HTAA_1_mc,
    cut=cut_HT_mc,
    bins=np.linspace(0, 1000, 1000)
)

aa_pt_data, aa_effht_data = efficiency_vs_pt(
    scores=HTAA_1_data,
    pts=HTAA_1_data,
    cut=cut_HT_data,
    bins=np.linspace(0, 1000, 1000)
)

tt_pt_mc, tt_effht_mc = efficiency_vs_pt(
    scores=HTtt_1_mc,
    pts=HTtt_1_mc,
    cut=cut_HT_mc,
    bins=np.linspace(0, 1000, 1000)
)

tt_pt_data, tt_effht_data = efficiency_vs_pt(
    scores=HTtt_1_data,
    pts=HTtt_1_data,
    cut=cut_HT_data,
    bins=np.linspace(0, 1000, 1000)
)


cut1_mc = np.percentile(bkg_1_mc, 99.75)
cut1_data = np.percentile(bkg_1_data, 99.75)
#cut2 = np.percentile(bkg_2, 99.875)
#cut3 = np.percentile(bkg_3, 99.875)


pt_bins = np.linspace(0, 1000, 50)

# dim = 2
aa_pt1_mc, aa_effht1_mc = efficiency_vs_pt(AA_1_mc, HTAA_1_mc, cut1_mc, pt_bins)
aa_pt1_data, aa_effht1_data = efficiency_vs_pt(AA_1_data, HTAA_1_data, cut1_data, pt_bins)

tt_pt1_mc, tt_effht1_mc = efficiency_vs_pt(TT_1_mc, HTtt_1_mc, cut1_mc, pt_bins)
tt_pt1_data, tt_effht1_data = efficiency_vs_pt(TT_1_data, HTtt_1_data, cut1_data, pt_bins)

'''
# dim = 2
aa_pt2, aa_effht2 = efficiency_vs_pt(AA_2, HTAA_1, cut2, pt_bins)
tt_pt2, tt_effht2 = efficiency_vs_pt(TT_2, HTtt_1, cut2, pt_bins)

# dim = 3
aa_pt3, aa_effht3 = efficiency_vs_pt(AA_3, HTAA_1, cut3, pt_bins)
tt_pt3, tt_effht3 = efficiency_vs_pt(TT_3, HTtt_1, cut3, pt_bins)

'''

fig, axes = plt.subplots(
    1, 2, figsize=(20, 8),
    sharey=True,
    gridspec_kw={"wspace": 0.08}
)

# ============================
# Left panel: HToAATo4B
# ============================

ax = axes[0]

ax.step(
    aa_pt_mc, aa_effht_mc,
    where="post",
    color="black",
    linewidth=2.5,
    label=r"$H_T$ Trigger on MC",
    linestyle="dashed"
)

ax.step(
    aa_pt_data, aa_effht_data,
    where="post",
    color="black",
    linewidth=2.5,
    label=r"$H_T$ Trigger on Data",
    linestyle="dotted"
)

ax.plot(
    aa_pt1_mc, aa_effht1_mc,
    color="limegreen",
    marker="*",
    markersize=9,
    linewidth=1.8,
    label=r"AD for Simulation, $d=2$"
)

ax.plot(
    aa_pt1_data, aa_effht1_data,
    color="seagreen",
    marker="o",
    markersize=9,
    linewidth=1.8,
    label=r"AD for Data, $d=2$"
)
'''
ax.plot(
    aa_pt2, aa_effht2,
    color="tab:green",
    marker="o",
    markersize=6,
    linewidth=1.8,
    linestyle="--",
    label=r"AD, $d=2$"
)

ax.plot(
    aa_pt3, aa_effht3,
    color="tab:green",
    marker="s",
    markersize=6,
    linewidth=1.8,
    linestyle=":",
    label=r"AD, $d=4$"
)
'''
ax.set_xlim(0, 700)
ax.set_ylim(0, 1.1)

ax.set_xlabel(r"$H_T$ [GeV]")
ax.set_ylabel("Efficiency")

ax.grid(True, which="both", alpha=0.3)

# Text-only legend entries for cuts
ax.plot([], [], " ", label=rf"$H_T$ cut on MC = {cut_HT_mc:.1f} GeV")
ax.plot([], [], " ", label=rf"$H_T$ cut on data = {cut_HT_data:.1f} GeV")
ax.plot([], [], " ", label=rf"AD($d=2$) cut on MC = {cut1_mc:.1f}")
ax.plot([], [], " ", label=rf"AD($d=2$) cut on data = {cut1_data:.1f}")
#ax.plot([], [], " ", label=rf"AD($d=2$) cut = {cut2:.1f}")
#ax.plot([], [], " ", label=rf"AD($d=4$) cut = {cut3:.1f}")

ax.legend(
    title=r"HToAATo4B",
    fontsize=17,
    #title_fontsize=14,
    frameon=True,
    loc="lower right"
)

add_cms_header_ax(ax)
save_subplot(fig, ax, "outputs/autoencoders/Eff_HT-a.pdf", pad=0.3)

# ============================
# Right panel: TTbar
# ============================

ax = axes[1]

ax.step(
    tt_pt_mc, tt_effht_mc,
    where="post",
    color="black",
    linewidth=2.5,
    label=r"$H_T$ Trigger on MC",
    linestyle="dashed"
)

ax.step(
    tt_pt_data, tt_effht_data,
    where="post",
    color="black",
    linewidth=2.5,
    label=r"$H_T$ Trigger on Data",
    linestyle="dotted"
)

ax.plot(
    tt_pt1_mc, tt_effht1_mc,
    color="tab:orange",
    marker="*",
    markersize=9,
    linewidth=1.8,
    label=r"AD for Simulation, $d=2$"
)

ax.plot(
    tt_pt1_data, tt_effht1_data,
    color="goldenrod",
    marker="o",
    markersize=9,
    linewidth=1.8,
    label=r"AD for Data, $d=2$"
)
'''
ax.plot(
    tt_pt2, tt_effht2,
    color="tab:orange",
    marker="o",
    markersize=6,
    linewidth=1.8,
    linestyle="--",
    label=r"AD, $d=2$"
)

ax.plot(
    tt_pt3, tt_effht3,
    color="tab:orange",
    marker="s",
    markersize=6,
    linewidth=1.8,
    linestyle=":",
    label=r"AD, $d=4$"
)
'''
ax.set_xlim(0, 700)
ax.set_ylim(0, 1.1)

ax.set_xlabel(r"$H_T$ [GeV]")

ax.grid(True, which="both", alpha=0.3)

ax.plot([], [], " ", label=rf"$H_T$ cut on MC = {cut_HT_mc:.1f} GeV")
ax.plot([], [], " ", label=rf"$H_T$ cut on data = {cut_HT_data:.1f} GeV")
ax.plot([], [], " ", label=rf"AD($d=2$) cut on MC = {cut1_mc:.1f}")
ax.plot([], [], " ", label=rf"AD($d=2$) cut on data = {cut1_data:.1f}")
#ax.plot([], [], " ", label=rf"AD($d=2$) cut = {cut2:.1f}")
#ax.plot([], [], " ", label=rf"AD($d=4$) cut = {cut3:.1f}")

ax.legend(
    title=r"TTbar",
    fontsize=17,
    #title_fontsize=14,
    frameon=True,
    loc="lower right"
)
add_cms_header_ax(ax)
save_subplot(fig, ax, "outputs/autoencoders/Eff_HT-b.pdf", pad=0.3)

# ============================
# Global cosmetics
# ============================


#add_cms_header(fig)

#lt.tight_layout()

plt.savefig('outputs/autoencoders/Eff_HT.pdf')
plt.close()





bkg_1_mc = example_2_mc["bkg"]
AA_1_mc = example_2_mc["AA"]
TT_1_mc = example_2_mc["TT"]

bkg_1_data = example_2_data["bkg"]
AA_1_data = example_2_data["AA"]
TT_1_data = example_2_data["TT"]

fig, axes = plt.subplots(1, 2, figsize=(20, 15))

# Extracting data for d=1 and d=4 from results_dim

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for dim = 1
threshold1 = np.percentile(bkg_1_mc,99.995)
#range=(0, threshold)

# overflow bin
n_bins = 50
bins1 = np.linspace(0, threshold1, n_bins)

# Clip scores: anything above threshold1 goes into threshold1
bkg_scores = np.clip(bkg_1_mc, None, threshold1)
AA_scores  = np.clip(AA_1_mc,  None, threshold1)
TT_scores  = np.clip(TT_1_mc,  None, threshold1)

axes[0].hist(bkg_scores, bins=bins1, alpha=1, density=True, label='test Simulation Background', histtype='step', color='tab:blue', linewidth=2)
axes[0].hist(AA_scores, bins=bins1, alpha=1, density=True, label='HToAATo4B Signal', histtype='step', color='tab:green', linewidth=2)
axes[0].hist(TT_scores, bins=bins1, alpha=1, density=True, label='TTbar Signal', histtype='step', color='goldenrod', linewidth=2)

axes[0].set_xlabel('Anomaly Score') #, fontsize=18
axes[0].set_ylabel('Density') #, fontsize=18
axes[0].set_yscale("log")
#axes[0].set_xlim(-0.1, 600)
axes[0].set_xlim(-0.1, threshold1+100)
axes[0].set_ylim(10E-6,100)

# Add note about overflows
axes[0].legend(fontsize=14, loc='best', frameon=True, title='AD Model on MC\n Overflows in the Last Bin', title_fontsize=14) #,

# Plot for dim = 4
threshold4 = np.percentile(bkg_1_data,99.995)

bins4 = np.linspace(0, threshold4, n_bins)

bkg_scores4 = np.clip(bkg_1_data, None, threshold4)
AA_scores4  = np.clip(AA_1_data,  None, threshold4)
TT_scores4  = np.clip(TT_1_data,  None, threshold4)

axes[1].hist(bkg_scores4, bins=bins4, alpha=1, density=True, label='test Real Data Background', histtype='step', color='tab:blue',linewidth=2)
axes[1].hist(AA_scores4, bins=bins4, alpha=1, density=True, label='HToAATo4B Signal', histtype='step', color='tab:green',linewidth=2)
axes[1].hist(TT_scores4, bins=bins4, alpha=1, density=True, label='TTbar Signal', histtype='step', color='goldenrod',linewidth=2)
#axes[1].set_title('Anomaly Score Distribution (d=4)')
axes[1].set_xlabel('Anomaly Score')
axes[1].set_ylabel('Density') #, fontsize=18
axes[1].set_yscale("log")
#axes[1].set_xlim(-0.1,120)
axes[1].set_xlim(-0.1, threshold4+100)
axes[1].set_ylim(10E-6,10)
#axes[1].legend(fontsize=14,title='Latent Dimension = 4', title_fontsize=16, loc='best', frameon=True)
axes[1].legend(fontsize=14, loc='best', frameon=True, title='AD Model on Data\n Overflows in the Last Bin)', title_fontsize=14) #, title_fontsize=14
# Adjust layout and save
add_cms_header_ax(axes[1])

plt.tight_layout()



save_subplot(fig, axes[0], "outputs/autoencoders/AD_hist_2016-a.pdf", pad=0.3)
save_subplot(fig, axes[1], "outputs/autoencoders/AD_hist_2016-b.pdf", pad=0.3)
plt.close(fig)


#plt.savefig("paper/AS_hist_comparison2016.pdf")
#fig.savefig("RealData/paper_data/AD_hist_comparison2016-newdata14.pdf")

