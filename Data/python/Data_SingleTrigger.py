##!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import re
import h5py
import hdf5plugin
import mplhep as hep
from pathlib import Path

hep.style.use("CMS")


def add_cms_header(fig, left_x=0.13, right_x=0.90, y=0.94):
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
        right_x, y, "Run 283408",
        ha="right", va="top",
        fontsize=24
    )


def save_pdf_png(fig, basepath, dpi_png=300):
    """
    Save figure as both PDF and PNG.
    basepath: path without extension, e.g. 'paper/bht_rate_pidData'
    """
    base = Path(basepath)
    base.parent.mkdir(parents=True, exist_ok=True)  # <-- ensure "paper/" exists

    fig.savefig(f"{basepath}.pdf", bbox_inches="tight")
    fig.savefig(f"{basepath}.png", bbox_inches="tight", dpi=dpi_png)


def PD_controller1(r_, pre_, cut_):
    #Kp = 2.55
    Kp = 30
    Kd = 5

    target = 0.25
    error = r_ - target
    delta = error - pre_
    newcut_ = cut_ + Kp * error + Kd * delta
    return newcut_, error


def PD_controller2(r_, pre_, cut_):
    Kp = 15
    Kd = 0

    target = 0.25
    error = r_ - target
    delta = error - pre_
    newcut_ = cut_ + Kp * error + Kd * delta
    return newcut_, error


def Sing_Trigger(bht_, ht_cut):
    """Return percentage of events passing the cut."""
    num_ = bht_.shape[0]
    accepted_ht_ = np.sum(bht_ >= ht_cut)
    r_ = 100.0 * accepted_ht_ / num_
    return r_


def read_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Background
        Bas01_tot = h5_file['data_scores01'][:]
        Bas04_tot = h5_file['data_scores04'][:]
        Bht_tot   = h5_file['data_ht'][:]
        B_npvs    = h5_file['data_Npv'][:]

        # Signal (ttbar)
        Sas01_tot1 = h5_file['matched_tt_scores01'][:]
        Sas04_tot1 = h5_file['matched_tt_scores04'][:]
        Sht_tot1   = h5_file['matched_tt_ht'][:]
        S_npvs1    = h5_file['matched_tt_npvs'][:]

        # Signal (AA→4b)
        Sas01_tot2 = h5_file['matched_aa_scores01'][:]
        Sas04_tot2 = h5_file['matched_aa_scores04'][:]
        Sht_tot2   = h5_file['matched_aa_ht'][:]
        S_npvs2    = h5_file['matched_aa_npvs'][:]

    return (
        Sas01_tot1, Sas04_tot1, Sht_tot1, S_npvs1,
        Sas01_tot2, Sas04_tot2, Sht_tot2, S_npvs2,
        Bas01_tot,  Bas04_tot,  Bht_tot,  B_npvs
    )


# =========================================================
# === Load data ===========================================
# =========================================================
# path = "new_Data/Matched_data_2016.h5"
path = "Data/Matched_data_2016_with04_paper.h5"

(
    Sas01_tot1, Sas04_tot1, Sht_tot1, S_npvs1,
    Sas01_tot2, Sas04_tot2, Sht_tot2, S_npvs2,
    Bas01_tot,  Bas04_tot,  Bht_tot,  B_npvs
) = read_data(path)

Nb = len(B_npvs)
Ns = len(S_npvs1)
print('Nb:', Nb)
print('Ns:', Ns)

print('hi')
N = Nb

pre_r1 = 0.0
pre_r2_1 = 0.0
pre_r2_4 = 0.0

# Fixed cuts from a reference window
fixed_Ht_cut  = np.percentile(Bht_tot[200000:210000], 99.75) #
fixed_AS_cut1 = np.percentile(Bas01_tot[200000:210000], 99.75)
fixed_AS_cut4 = np.percentile(Bas04_tot[200000:210000], 99.75)

print('fixed_Ht_cut', fixed_Ht_cut)
print('fixed_AS_cut1', fixed_AS_cut1)
print('fixed_AS_cut4', fixed_AS_cut4)
print(np.percentile(Bht_tot[200000:], 99.75))

percen_9975 = np.percentile(Bas04_tot, 99.75)
AA_passed = 100.0 * np.sum(Sas04_tot2 > percen_9975) / len(Sas04_tot2)
TT_passed = 100.0 * np.sum(Sas04_tot1 > percen_9975) / len(Sas04_tot1)
print('AA_passed', AA_passed)
print('TT_passed', TT_passed)

Ht_cut  = fixed_Ht_cut
AS_cut1 = fixed_AS_cut1
AS_cut4 = fixed_AS_cut4

# Containers
bht = []
bas1 = []
bas4 = []
sht1 = []
sas1_1 = []
sas1_4 = []
sht2 = []
sas2_1 = []
sas2_4 = []

R1 = []     # HT, constant
R2 = []     # HT, PD
L_R3 = []   # HT local sig tt (const)
L_R4 = []   # HT local sig tt (PD)
R3 = [0.0]  # HT cumulative sig tt (const)
R4 = [0.0]  # HT cumulative sig tt (PD)
L_R5 = []   # HT local sig AA (const)
L_R6 = []   # HT local sig AA (PD)
R5 = [0.0]  # HT cumulative sig AA (const)
R6 = [0.0]  # HT cumulative sig AA (PD)

E1_1 = []   # AD bkg, const, dim1
E2_1 = []   # AD bkg, PD,   dim1
E1_4 = []   # AD bkg, const, dim4 (not used in final plots, but kept)
E2_4 = []   # AD bkg, PD,   dim4

E3_1 = [0.0]  # AD cumulative sig tt, const, dim1
E4_1 = [0.0]  # AD cumulative sig tt, PD,   dim1
L_E3_1 = []   # AD local sig tt,     const, dim1
L_E4_1 = []   # AD local sig tt,     PD,   dim1

E3_4 = [0.0]
E4_4 = [0.0]
L_E3_4 = []
L_E4_4 = []

E5_1 = [0.0]  # AD cumulative sig AA, const, dim1
E6_1 = [0.0]  # AD cumulative sig AA, PD,   dim1
L_E5_1 = []   # AD local sig AA,     const, dim1
L_E6_1 = []   # AD local sig AA,     PD,   dim1

E5_4 = [0.0]
E6_4 = [0.0]
L_E5_4 = []
L_E6_4 = []

chunk_size = 20000

# =========================================================
# === Main loop ===========================================
# =========================================================
for I in range(N):
    if I < 200000:
        continue

    if I % chunk_size == 0:
        start_idx = I
        end_idx = min(I + chunk_size, N)
        indices = list(range(start_idx, end_idx))

        bht   = Bht_tot[indices]
        bas1  = Bas01_tot[indices]
        bas4  = Bas04_tot[indices]
        b_npvs = B_npvs[indices]

        npv_min = np.min(b_npvs)
        npv_max = np.max(b_npvs)

        # masks (currently not used, but kept for completeness)
        signal_mask1 = (S_npvs1 >= npv_min) & (S_npvs1 <= npv_max)
        signal_mask2 = (S_npvs2 >= npv_min) & (S_npvs2 <= npv_max)

        # ttbar / AA matched by index (same indices as background)
        sht1   = Sht_tot1[indices]
        sas1_1 = Sas01_tot1[indices]
        sas1_4 = Sas04_tot1[indices]

        sht2   = Sht_tot2[indices]
        sas2_1 = Sas01_tot2[indices]
        sas2_4 = Sas04_tot2[indices]

        # ---------------- HT background ----------------
        rate1 = Sing_Trigger(np.array(bht), fixed_Ht_cut)
        rate2 = Sing_Trigger(np.array(bht), Ht_cut)
        R1.append(rate1)
        R2.append(rate2)

        # ---------------- HT signal: ttbar -------------
        rate3 = Sing_Trigger(np.array(sht1), fixed_Ht_cut)
        rate4 = Sing_Trigger(np.array(sht1), Ht_cut)
        L_R3.append(rate3)
        L_R4.append(rate4)
        a = ((len(R3) - 1) * R3[-1] + rate3) / len(R3)
        R3.append(a)
        b = ((len(R4) - 1) * R4[-1] + rate4) / len(R4)
        R4.append(b)

        # ---------------- HT signal: AA ----------------
        rate3 = Sing_Trigger(np.array(sht2), fixed_Ht_cut)
        rate4 = Sing_Trigger(np.array(sht2), Ht_cut)
        L_R5.append(rate3)
        L_R6.append(rate4)
        a = ((len(R5) - 1) * R5[-1] + rate3) / len(R5)
        R5.append(a)
        b = ((len(R6) - 1) * R6[-1] + rate4) / len(R6)
        R6.append(b)

        # update HT cut
        Ht_cut, pre_r1 = PD_controller1(R2[-1], pre_r1, Ht_cut)

        # ---------------- AD background ----------------
        rate1_1 = Sing_Trigger(np.array(bas1), fixed_AS_cut1)
        rate2_1 = Sing_Trigger(np.array(bas1), AS_cut1)
        E1_1.append(rate1_1)
        E2_1.append(rate2_1)

        rate1_4 = Sing_Trigger(np.array(bas4), fixed_AS_cut4)
        rate2_4 = Sing_Trigger(np.array(bas4), AS_cut4)
        E1_4.append(rate1_4)
        E2_4.append(rate2_4)

        # ---------------- AD signal: ttbar -------------
        rate3_1 = Sing_Trigger(np.array(sas1_1), fixed_AS_cut1)
        rate4_1 = Sing_Trigger(np.array(sas1_1), AS_cut1)
        L_E3_1.append(rate3_1)
        L_E4_1.append(rate4_1)
        a = ((len(E3_1) - 1) * E3_1[-1] + rate3_1) / len(E3_1)
        E3_1.append(a)
        b = ((len(E4_1) - 1) * E4_1[-1] + rate4_1) / len(E4_1)
        E4_1.append(b)

        # (dim=4, ttbar – not used in final plots, but kept)
        rate3_4 = Sing_Trigger(np.array(sas1_4), fixed_AS_cut4)
        rate4_4 = Sing_Trigger(np.array(sas1_4), AS_cut4)
        L_E3_4.append(rate3_4)
        L_E4_4.append(rate4_4)
        a = ((len(E3_4) - 1) * E3_4[-1] + rate3_4) / len(E3_4)
        E3_4.append(a)
        b = ((len(E4_4) - 1) * E4_4[-1] + rate4_4) / len(E4_4)
        E4_4.append(b)

        # ---------------- AD signal: AA ----------------
        rate3_1 = Sing_Trigger(np.array(sas2_1), fixed_AS_cut1)
        rate4_1 = Sing_Trigger(np.array(sas2_1), AS_cut1)
        L_E5_1.append(rate3_1)
        L_E6_1.append(rate4_1)
        a = ((len(E5_1) - 1) * E5_1[-1] + rate3_1) / len(E5_1)
        E5_1.append(a)
        b = ((len(E6_1) - 1) * E6_1[-1] + rate4_1) / len(E6_1)
        E6_1.append(b)

        # (dim=4, AA – not used in final plots, but kept)
        rate3_4 = Sing_Trigger(np.array(sas2_4), fixed_AS_cut4)
        rate4_4 = Sing_Trigger(np.array(sas2_4), AS_cut4)
        L_E5_4.append(rate3_4)
        L_E6_4.append(rate4_4)
        a = ((len(E5_4) - 1) * E5_4[-1] + rate3_4) / len(E5_4)
        E5_4.append(a)
        b = ((len(E6_4) - 1) * E6_4[-1] + rate4_4) / len(E6_4)
        E6_4.append(b)

        # update AD cuts
        AS_cut1, pre_r2_1 = PD_controller2(E2_1[-1], pre_r2_1, AS_cut1)
        AS_cut4, pre_r2_4 = PD_controller2(E2_4[-1], pre_r2_4, AS_cut4)


# =========================================================
# === Convert to numpy, scale rates =======================
# =========================================================
E1_1 = np.array(E1_1) * 400.0
E1_4 = np.array(E1_4) * 400.0
E2_1 = np.array(E2_1) * 400.0
E2_4 = np.array(E2_4) * 400.0

R1 = np.array(R1) * 400.0
R2 = np.array(R2) * 400.0

# Drop the initial dummy element (0) for cumulative arrays
R3 = np.array(R3[1:])
R4 = np.array(R4[1:])
R5 = np.array(R5[1:])
R6 = np.array(R6[1:])

L_R3 = np.array(L_R3)
L_R4 = np.array(L_R4)
L_R5 = np.array(L_R5)
L_R6 = np.array(L_R6)

E3_1 = np.array(E3_1[1:])
E4_1 = np.array(E4_1[1:])
E5_1 = np.array(E5_1[1:])
E6_1 = np.array(E6_1[1:])

L_E3_1 = np.array(L_E3_1)
L_E4_1 = np.array(L_E4_1)
L_E5_1 = np.array(L_E5_1)
L_E6_1 = np.array(L_E6_1)

# =========================================================
# === Common styles for plots =============================
# =========================================================
styles = {
    "Constant": {"linestyle": "dashed", "linewidth": 2.5},
    "PD":       {"linestyle": "solid",  "linewidth": 2.0},
}


# =========================================================
# === 1) HT background: bht_rate_pidData ==================
# =========================================================
time = np.linspace(0, 1, len(R1))
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time, R1, label='Constant Menu', color='tab:blue',
        linewidth=3, linestyle='dashed')
ax.plot(time, R2, label='PD Controller', color='mediumblue',
        linewidth=2.5, linestyle='solid')

# tolerance band
ax.axhline(y=0.275 * 400, color='gray', linestyle='--', linewidth=1.5)
ax.axhline(y=0.225 * 400, color='gray', linestyle='--', linewidth=1.5)

ax.set_xlabel('Time (Fraction of Run)', loc='center')
ax.set_ylabel('Background Rate [kHz]',   loc='center')
ax.set_ylim(0, 200)
ax.grid(True, linestyle='--', alpha=0.6)

# main legend (constant vs PD)
const_rate = mlines.Line2D([], [], color='tab:blue',
                           linestyle='dashed', linewidth=3)
pd_rate    = mlines.Line2D([], [], color='mediumblue',
                           linestyle='solid',  linewidth=2.5)

handles_main = [const_rate, pd_rate]
labels_main  = ["Constant Menu", "PD Controller"]

leg_main = ax.legend(
    handles_main, labels_main,
    title="HT Trigger",
    ncol=1, loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    frameon=True,
    handlelength=2, columnspacing=1, labelspacing=0.8
)
ax.add_artist(leg_main)

# legend for tolerance lines
upper_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)
lower_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)

ax.legend(
    [upper_tol, lower_tol],
    ["Upper Tolerance (110)", "Lower Tolerance (90)"],
    title="Reference",
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    fontsize=14,
    handlelength=2
)

add_cms_header(fig)
save_pdf_png(fig, 'paper/bht_rate_pidData')
plt.show()
plt.close(fig)


# =========================================================
# === 2) AD background: bas_rate_pidData ==================
# =========================================================
time = np.linspace(0, 1, len(E1_1))
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time, E1_1, label='Constant Menu, model dim=1',
        color='tab:blue', linewidth=3, linestyle='dotted')
ax.plot(time, E2_1, label='PD Controller, model dim=1',
        color='mediumblue', linewidth=2.5, linestyle='solid')

ax.axhline(y=0.275 * 400, color='gray', linestyle='--', linewidth=1.5)
ax.axhline(y=0.225 * 400, color='gray', linestyle='--', linewidth=1.5)

ax.set_xlabel('Time (Fraction of Run)', loc='center')
ax.set_ylabel('Background Rate [kHz]',   loc='center')
ax.set_ylim(60, 200)
ax.grid(True, linestyle='--', alpha=0.6)

# main legend
header_const = mlines.Line2D([], [], color='none', linestyle='none')
header_pd    = mlines.Line2D([], [], color='none', linestyle='none')

const_dim1 = mlines.Line2D([], [], color='tab:blue',
                           linestyle='dotted', linewidth=3)
pd_dim1    = mlines.Line2D([], [], color='mediumblue',
                           linestyle='solid', linewidth=2.5)

handles_main = [
    header_const,
    const_dim1,
    header_pd,
    pd_dim1,
]
labels_main = [
    "Constant Menu",
    "model dim=1",
    "PD Controller",
    "model dim=1",
]

leg_main = ax.legend(
    handles_main, labels_main,
    title="AD Trigger",
    fontsize=17,
    ncol=1,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    frameon=True,
    handlelength=2, columnspacing=1, labelspacing=0.8
)
ax.add_artist(leg_main)

# tolerance legend
upper_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)
lower_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)

ax.legend(
    [upper_tol, lower_tol],
    ["Upper Tolerance (110)", "Lower Tolerance (90)"],
    title="Reference",
    fontsize=18,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    handlelength=2
)

add_cms_header(fig)
save_pdf_png(fig, 'paper/bas_rate_pidData')
plt.show()
plt.close(fig)


# =========================================================
# === 3) HT cumulative efficiency: sht_rate_pidData2data ==
# =========================================================
colors_ht = {
    "ttbar": "goldenrod",
    "HToAATo4B": "seagreen",
}

eff0_ttbar     = R3[0]
eff0_AA        = R5[0]
eff0_ttbar_PD  = R4[0]
eff0_AA_PD     = R6[0]

time = np.linspace(0, 1, len(R3))
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    time, R3 / eff0_ttbar,
    label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar:.2f}\%$)",
    color=colors_ht["ttbar"], **styles["Constant"]
)
ax.plot(
    time, R5 / eff0_AA,
    label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA:.2f}\%$)",
    color=colors_ht["HToAATo4B"], **styles["Constant"]
)
ax.plot(
    time, R4 / eff0_ttbar_PD,
    label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_PD:.2f}\%$)",
    color=colors_ht["ttbar"], **styles["PD"]
)
ax.plot(
    time, R6 / eff0_AA_PD,
    label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_PD:.2f}\%$)",
    color=colors_ht["HToAATo4B"], **styles["PD"]
)

ax.set_xlabel("Time (Fraction of Run)", loc='center')
ax.set_ylabel("Relative Cumulative Efficiency", loc='center')
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_ylim(0.85, 1.5)

ax.legend(title="HT Trigger", fontsize=16, frameon=True, loc="best")

add_cms_header(fig)
save_pdf_png(fig, "paper/sht_rate_pidData2data")
plt.show()
plt.close(fig)


# =========================================================
# === 4) AD cumulative efficiency: sas_rate_pidData2data ==
# =========================================================
colors_ad = {
    "ttbar": "goldenrod",
    "HToAATo4B": "limegreen",
}

eff0_ttbar_const = E3_1[0]
eff0_AA_const    = E5_1[0]
eff0_ttbar_pd    = E4_1[0]
eff0_AA_pd       = E6_1[0]

time = np.linspace(0, 1, len(E3_1))
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    time, E3_1 / eff0_ttbar_const,
    label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar_const:.2f}\%$)",
    color=colors_ad["ttbar"], **styles["Constant"]
)
ax.plot(
    time, E5_1 / eff0_AA_const,
    label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA_const:.2f}\%$)",
    color=colors_ad["HToAATo4B"], **styles["Constant"]
)
ax.plot(
    time, E4_1 / eff0_ttbar_pd,
    label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_pd:.2f}\%$)",
    color=colors_ad["ttbar"], **styles["PD"]
)
ax.plot(
    time, E6_1 / eff0_AA_pd,
    label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_pd:.2f}\%$)",
    color=colors_ad["HToAATo4B"], **styles["PD"]
)

ax.set_xlabel("Time (Fraction of Run)", loc='center')
ax.set_ylabel("Relative Cumulative Efficiency", loc='center')
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_ylim(0.98, 1.5)

ax.legend(title="AD Trigger", fontsize=16, frameon=True, loc="best")

add_cms_header(fig)
save_pdf_png(fig, "paper/sas_rate_pidData2data")
plt.show()
plt.close(fig)


# =========================================================
# === 5) HT local efficiency: L_sht_rate_pidData2data =====
# =========================================================
colors_ht_local = {
    "ttbar": "goldenrod",
    "HToAATo4B": "seagreen",
}

eff0_ttbar     = L_R3[0]
eff0_AA        = L_R5[0]
eff0_ttbar_PD  = L_R4[0]
eff0_AA_PD     = L_R6[0]

time = np.linspace(0, 1, len(L_R3))
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    time, L_R3 / eff0_ttbar,
    label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar:.2f}\%$)",
    color=colors_ht_local["ttbar"], **styles["Constant"]
)
ax.plot(
    time, L_R5 / eff0_AA,
    label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA:.2f}\%$)",
    color=colors_ht_local["HToAATo4B"], **styles["Constant"]
)
ax.plot(
    time, L_R4 / eff0_ttbar_PD,
    label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_PD:.2f}\%$)",
    color=colors_ht_local["ttbar"], **styles["PD"]
)
ax.plot(
    time, L_R6 / eff0_AA_PD,
    label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_PD:.2f}\%$)",
    color=colors_ht_local["HToAATo4B"], **styles["PD"]
)

ax.set_xlabel("Time (Fraction of Run)", loc='center')
ax.set_ylabel("Relative Efficiency", loc='center')
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_ylim(0.0, 1.6)

ax.legend(title="HT Trigger", fontsize=15, frameon=True, loc="best")

add_cms_header(fig)
save_pdf_png(fig, "paper/L_sht_rate_pidData2data")
plt.show()
plt.close(fig)


# =========================================================
# === 6) AD local efficiency: L_sas_rate_pidData2data =====
# =========================================================
colors_ad_local = {
    "ttbar": "goldenrod",
    "HToAATo4B": "limegreen",
}

eff0_ttbar_const = L_E3_1[0]
eff0_AA_const    = L_E5_1[0]
eff0_ttbar_pd    = L_E4_1[0]
eff0_AA_pd       = L_E6_1[0]

time = np.linspace(0, 1, len(L_E3_1))
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    time, L_E3_1 / eff0_ttbar_const,
    label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar_const:.2f}\%$)",
    color=colors_ad_local["ttbar"], **styles["Constant"]
)
ax.plot(
    time, L_E5_1 / eff0_AA_const,
    label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA_const:.2f}\%$)",
    color=colors_ad_local["HToAATo4B"], **styles["Constant"]
)
ax.plot(
    time, L_E4_1 / eff0_ttbar_pd,
    label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_pd:.2f}\%$)",
    color=colors_ad_local["ttbar"], **styles["PD"]
)
ax.plot(
    time, L_E6_1 / eff0_AA_pd,
    label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_pd:.2f}\%$)",
    color=colors_ad_local["HToAATo4B"], **styles["PD"]
)

ax.set_xlabel("Time (Fraction of Run)", loc='center')
ax.set_ylabel("Relative Efficiency", loc='center')
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_ylim(0.9, 1.7)

ax.legend(title="AD Trigger", fontsize=15, frameon=True, loc="best")

add_cms_header(fig)
save_pdf_png(fig, "paper/L_sas_rate_pidData2data")
plt.show()
plt.close(fig)
