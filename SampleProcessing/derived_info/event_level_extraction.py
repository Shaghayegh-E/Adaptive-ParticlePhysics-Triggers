# Fig 3 Comparison of key per-event features across different samples: real collision data, minimum-bias simulation, SM $t\bar{t}$ events, and a BSM benchmark process Higgs-to-AA-to-4b. 
# These distributions provide insight into the kinematic and topological differences that the trigger strategies aim to capture.
from __future__ import annotations
import argparse
import numpy as np
from typing import Dict, Any
from .data_io import process_h5_file
from .scoring import load_autoencoders, calculate_batch_loss, calculate_H_met, count_njets

import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import mplhep as hep

hep.style.use("CMS")

def plot_normed_to_max(ax, arrays, labels, colors, bins, xlabel):
    hists = [np.histogram(a, bins=bins, density=True)[0] for a in arrays]
    ymax = max((h.max() for h in hists if h.size), default=1.0)
    for a, lab, col in zip(arrays, labels, colors):
        y, edges = np.histogram(a, bins=bins, density=True)
        ax.stairs(y, edges, label=lab, color=col, linewidth=1.8)
    #ax.set_xlim(bins[0], bins[-1])
    #ax.set_ylim(0, 1.05)
    #ax.set_title(title, fontsize=22)
    ax.set_xlabel(xlabel, loc='center')
    ax.set_ylabel("Density",loc='center')#, fontsize=22)
    ax.legend( frameon=True)
def plot_step_hist(ax, arr, bins, label, color, density=True):
    y, edges = np.histogram(arr, bins=bins, density=density)
    ax.stairs(y, edges, label=label, color=color, linewidth=1.8)

def compute_missing_ht(jets):
    px = np.sum(jets[:, :, 2] * np.cos(jets[:, :, 1]), axis=1)  # pt * cos(phi)
    py = np.sum(jets[:, :, 2] * np.sin(jets[:, :, 1]), axis=1)  # pt * sin(phi)
    return np.sqrt(px**2 + py**2)

mc_bkg_jets, mc_bkg_ht = process_h5_file("Data/MinBias_1.h5") 
mc_ttbar_jets, mc_ttbar_ht = process_h5_file("Data/TT_1.h5")
mc_aa_jets, mc_aa_ht = process_h5_file("Data/HToAATo4B.h5")
mc_data_jets, mc_data_ht = process_h5_file("Data/data_Run_2016_283408_longest.h5")

mc_bkg_missing_ht = compute_missing_ht(mc_bkg_jets)
mc_ttbar_missing_ht = compute_missing_ht(mc_ttbar_jets)
mc_aa_missing_ht = compute_missing_ht(mc_aa_jets)
mc_data_missing_ht = compute_missing_ht(mc_data_jets)

mc_bkg_njets = np.sum(mc_bkg_jets[:, :, 2] > 0, axis=1)
mc_ttbar_njets = np.sum(mc_ttbar_jets[:, :, 2] > 0, axis=1)
mc_aa_njets = np.sum(mc_aa_jets[:, :, 2] > 0, axis=1)
mc_data_njets = np.sum(mc_data_jets[:, :, 2] > 0, axis=1)



# Create masks for jets with pT > 0
bkg_mask   = mc_bkg_jets[:, :, 2] > 0
ttbar_mask = mc_ttbar_jets[:, :, 2] > 0
aa_mask    = mc_aa_jets[:, :, 2] > 0
data_mask  = mc_data_jets[:, :, 2] > 0

# Extract and shift pT, eta, phi (only jets with pT > 0)
mc_bkg_pt  = mc_bkg_jets[:, :, 2][bkg_mask]
#mc_bkg_eta = mc_bkg_jets[:, :, 0][bkg_mask] - 5.0
#mc_bkg_phi = mc_bkg_jets[:, :, 1][bkg_mask] - np.pi

print('bkg done')

mc_ttbar_pt  = mc_ttbar_jets[:, :, 2][ttbar_mask]
#mc_ttbar_eta = mc_ttbar_jets[:, :, 0][ttbar_mask] - 5.0
#mc_ttbar_phi = mc_ttbar_jets[:, :, 1][ttbar_mask] - np.pi

print('ttbar done')

mc_aa_pt  = mc_aa_jets[:, :, 2][aa_mask]
#mc_aa_eta = mc_aa_jets[:, :, 0][aa_mask] - 5.0
#mc_aa_phi = mc_aa_jets[:, :, 1][aa_mask] - np.pi

print('aa done')

mc_data_pt  = mc_data_jets[:, :, 2][data_mask]
#mc_data_eta = mc_data_jets[:, :, 0][data_mask] - 5.0
#mc_data_phi = mc_data_jets[:, :, 1][data_mask] - np.pi

print('data done')
# ===================== PLOTTING (separato per subfigure) =====================

# Bins + colori
ht_bins  = np.linspace(0, 700, 16)
mht_bins = np.linspace(0,  175, 8)
pt_bins  = np.linspace(0,  200, 10)
nj_bins  = np.arange(-0.5, 8.5 + 1, 1.0)  # bin centrati su 0..8

COLORS = {
    "data":   "firebrick",   # rosso scuro
    "minbias":"royalblue",   # blu saturo
    "ttbar":  "goldenrod",   # giallo scuro
    "hToAA":  "seagreen"     # verde scuro
}

labels = ["2016 Zerobias Data", "MinBias", "TTbar", "HToAATo4B"]
colors = [COLORS["data"], COLORS["minbias"], COLORS["ttbar"], COLORS["hToAA"]]

# --- 1) HT Distribution ---
fig, ax = plt.subplots(figsize=(7,6))
plot_normed_to_max(
    ax,
    [mc_data_ht, mc_bkg_ht, mc_ttbar_ht, mc_aa_ht],
    labels, colors, ht_bins,
   # "HT Distribution", 
   r"$H_T$ [GeV]"
)
#ax.set_xlim(0, 700)
#ax.set_ylim(0, 0.02)
ax.tick_params(axis='both', which='major')#, labelsize=16)
plt.tight_layout()
plt.savefig("outputs/HT_distribution.pdf")
plt.close()

# --- 2) Number of Jets per Event ---
fig, ax = plt.subplots(figsize=(7,6))
plot_step_hist(ax, mc_data_njets,  nj_bins, "2016 Zerobias Data", COLORS["data"])
plot_step_hist(ax, mc_bkg_njets,   nj_bins, "MinBias",   COLORS["minbias"])
plot_step_hist(ax, mc_ttbar_njets, nj_bins, "TTbar",     COLORS["ttbar"])
plot_step_hist(ax, mc_aa_njets,    nj_bins, "HToAATo4B", COLORS["hToAA"])
#ax.set_title("Number of Jets per Event", fontsize=22)
ax.set_xlabel("Number of Jets",loc='center')#, fontsize=22)
ax.set_ylabel("Density",loc='center')#, fontsize=22)
ax.set_xticks(range(0, 9))
ax.legend(frameon=True)
ax.tick_params(axis='both', which='major')#, labelsize=16)
plt.tight_layout()
plt.savefig("outputs/Njets_distribution.pdf")
plt.close()

# --- 3) Missing HT ---
fig, ax = plt.subplots(figsize=(7,6))
plot_normed_to_max(
    ax,
    [mc_data_missing_ht, mc_bkg_missing_ht, mc_ttbar_missing_ht, mc_aa_missing_ht],
    labels, colors, mht_bins,
    #r"Missing $H_T$ Distribution", 
    r"Missing $H_T$ [GeV]"
)
#ax.set_xlim(0, 175)
#ax.set_ylim(0, 0.04)
ax.tick_params(axis='both', which='major')#, labelsize=16)
plt.tight_layout()
plt.savefig("outputs/MissingHT_distribution.pdf")
plt.close()

# --- 4) Jet pT ---
fig, ax = plt.subplots(figsize=(7,6))
plot_normed_to_max(
    ax,
    [mc_data_pt, mc_bkg_pt, mc_ttbar_pt, mc_aa_pt],
    labels, colors, pt_bins,
  #  r"Jet $p_T$ Distribution",
     r"$p_T$ [GeV]"
)
ax.set_xlim(-10, 220)
#ax.set_ylim(0, 0.035)
#ax.tick_params(axis='both', which='major')#, labelsize=16)
plt.tight_layout()
plt.savefig("outputs/jetPt_distribution.pdf")
plt.close()

# =================== FINE PLOTTING ===================


