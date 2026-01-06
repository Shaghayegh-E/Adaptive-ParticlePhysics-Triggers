#!/usr/bin/env python
# Singletrigger_plots.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np, matplotlib.ticker as mticker
import argparse

import os
import matplotlib.lines as mlines

import matplotlib.transforms as mtransforms

import mplhep as hep
hep.style.use("CMS")


# OUTDIR = "outputs"
ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs" / "demo_singletrigger"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_results(path):
    return np.load(path)

# ----------------------------------------------------------------------
# 1) BACKGROUND RATE: HT (fixed-only & PD) -> bht_*.pdf
# ----------------------------------------------------------------------
def plot_ht_background_fixed(data, isMC=True):
    """HT background – constant menu only (fixed cuts)."""
    # R1 = np.asarray(data["R1"]) * 1.0  # already scaled to kHz in producer
    # t = _time_axis(len(R1))

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(t, R1, label="Constant Menu", color="tab:blue",
    #         linewidth=3, linestyle="dashed")
    # _tol_lines(ax)

    # ax.set_xlabel("Time (Fraction of Run)", loc="center")
    # ax.set_ylabel("Background Rate [kHz]",  loc="center")
    # ax.grid(True, linestyle="--", alpha=0.6)
    # ax.legend(title="HT Trigger", frameon=True, loc="best")
    # ax.set_ylim(0, 200)

    # fig.savefig(OUTDIR / "bht_rate_fixed.pdf", bbox_inches="tight", dpi=300)
    # plt.close(fig)
    R1 = np.asarray(data["R1"]) * 1.0  # already scaled to kHz in producer
    time = np.linspace(0, 1, len(R1))
    plt.figure(figsize=(10, 6))
    plt.plot(time, R1, label='Constant Menu', color='tab:blue', linewidth=3, linestyle='dashed')

    plt.axhline(y=110, color='gray', linestyle='--', linewidth=1.5, label='Upper Tolerance (112)')
    plt.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, label='Lower Tolerance (88)')

    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Background Rate [kHz]', loc='center')
    plt.legend(title='HT Trigger', loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    if isMC:
        plt.savefig(f"{OUTDIR}/bht_rate_fixedMC.pdf", bbox_inches='tight')
    else: 
        plt.savefig(f"{OUTDIR}/bht_rate_fixedData.pdf", bbox_inches='tight')

    #plt.savefig(OUTDIR / 'bht_rate_fixed.pdf', bbox_inches='tight', dpi=300)
    
    plt.close()

def plot_ht_background(data, isMC=True):
    R1 = data["R1"]
    R2 = data["R2"]

    time = np.linspace(0, 1, len(R1))
    plt.figure(figsize=(10, 6))
    plt.plot(time, R1, label='Constant Menu', color='tab:blue', linewidth=3, linestyle='dashed')
    plt.plot(time, R2, label='PD Controller', color='mediumblue', linewidth=2.5, linestyle='solid')

    plt.axhline(y=110, color='gray', linestyle='--', linewidth=1.5)
    plt.axhline(y=90, color='gray', linestyle='--', linewidth=1.5)

    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Background Rate [kHz]', loc='center')
    plt.ylim(0, 200)
    plt.grid(True, linestyle='--', alpha=0.6)


    ax = plt.gca()

    # ---- Main legend with column headers ----
    header_const = mlines.Line2D([], [], color='none', linestyle='none')
    header_pd    = mlines.Line2D([], [], color='none', linestyle='none')

    const_rate = mlines.Line2D([], [], color='tab:blue',   linestyle='dashed', linewidth=3)
    pd_rate    = mlines.Line2D([], [], color='mediumblue', linestyle='solid',  linewidth=2.5)

    handles_main = [
        const_rate,   pd_rate  
    ]
    labels_main = [
        "Constant Menu", "PD Controller"
    ]


    leg_main = ax.legend(
        handles_main, labels_main,
        title="HT Trigger",
        fontsize=17,
        ncol=1,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),  
        frameon=True,
        handlelength=2, columnspacing=1, labelspacing=0.8
    )
    ax.add_artist(leg_main)

    
    ## ---- Second legend for tolerance lines ----
    upper_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)
    lower_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)
    leg_tol = ax.legend(
        [upper_tol, lower_tol],
        ["Upper Tolerance (110)", "Lower Tolerance (90)"],
        title="Reference",
        fontsize=18,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),  # ← più interna verso sinistra
        frameon=True,
        handlelength=2
    )

    if isMC:
        plt.savefig(f"{OUTDIR}/bht_rate_pidMC.pdf", bbox_inches='tight')
    else: 
        plt.savefig(f"{OUTDIR}/bht_rate_pidData.pdf", bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------------------
# 2) BACKGROUND RATE: AD (fixed-only & PD) -> bas_*.pdf
# ----------------------------------------------------------------------
def plot_ad_background_fixed(data, isMC=True):
    """AD background – constant menu only (dim=2 fixed cuts)."""
    E1 = np.asarray(data["E1"])

    time = np.linspace(0, 1, len(E1))
    plt.figure(figsize=(10, 6))
    plt.plot(time, E1, label='Constant Menu, model dim=2', color='tab:blue', linewidth=3, linestyle='dotted')
    
    plt.axhline(y=110, color='gray', linestyle='--', linewidth=1.5, label='Upper Tolerance (112)')
    plt.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, label='Lower Tolerance (88)')

    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Background Rate [kHz]', loc='center')
    plt.ylim(0,300)
    plt.legend(title='AD Trigger', loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    if isMC:
        plt.savefig(f"{OUTDIR}/bas_rate_fixedMC.pdf", bbox_inches='tight')
    else: 
        plt.savefig(f"{OUTDIR}/bas_rate_fixedData.pdf", bbox_inches='tight')

    #plt.savefig(OUTDIR / 'bas_rate_fixed.pdf', bbox_inches='tight', dpi=300)

    plt.close()


def plot_ad_background(data, isMC=True):
    E1 = data["E1"]

    E2 = data["E2"]


    time = np.linspace(0, 1, len(E1))
    plt.figure(figsize=(10, 6))

    plt.plot(time, E1, label='Constant Menu, model dim=2',
             color='mediumblue', linewidth=3, linestyle='dotted')
    
    plt.plot(time, E2, label='PD Controller, model dim=2',
             color='mediumblue', linewidth=2.5, linestyle='solid')
    
    plt.axhline(y=110, color='gray', linestyle='--', linewidth=1.5)
    plt.axhline(y=90, color='gray', linestyle='--', linewidth=1.5)

    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Background Rate [kHz]', loc='center')
    plt.ylim(0, 270)
    plt.grid(True, linestyle='--', alpha=0.6)
    # ---- Main legend with column headers ----
    ax = plt.gca()

    # Header placeholders
    header_const = mlines.Line2D([], [], color='none', linestyle='none')
    header_pd    = mlines.Line2D([], [], color='none', linestyle='none')

    # Dummy handles that match the plotted styles
    const_dim2 = mlines.Line2D([], [], color='tab:blue',    linestyle='dotted', linewidth=3)
    pd_dim2    = mlines.Line2D([], [], color='mediumblue',  linestyle='solid',  linewidth=2.5)
   
    handles_main = [
        header_const,   
        const_dim2, 
        header_pd,
        pd_dim2
    ]
    labels_main = [
        "Constant Menu", 
        "model dim=2",   
        "PD Controller",
        "model dim=2"
    ]

    leg_main = ax.legend(
        handles_main, labels_main,
        title="AD Trigger",
        fontsize=16,
        ncol=2,
        loc='upper left',
        #bbox_to_anchor=(0.02, 0.98),  # ← sposta leggermente dentro il grafico
        frameon=True,
        handlelength=2, columnspacing=1, labelspacing=0.8
    )
    ax.add_artist(leg_main)
    upper_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)
    lower_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)

    leg_tol = ax.legend(
        [upper_tol, lower_tol],
        ["110 kHz", "90kHz"],
        title="Reference",
        fontsize=15,
        loc='upper right',
        #bbox_to_anchor=(0.98, 0.98),  # ← più interna verso sinistra
        frameon=True,
        handlelength=2
    )
    plt.ylim(0, 270)
    if isMC:
        plt.savefig(f"{OUTDIR}/bas_rate_pidMC.pdf", bbox_inches='tight')
    else: 
        plt.savefig(f"{OUTDIR}/bas_rate_pidData.pdf", bbox_inches='tight')

    #plt.savefig(f"{OUTDIR}/bas_rate_pidMC.pdf", bbox_inches='tight')
    plt.close()


# ----------------------------------------------------------------------
# 3) HT SIGNAL EFFICIENCY (cumulative + local, fixed & PD)
# ----------------------------------------------------------------------


def plot_ht_cumulative_eff(data, isMC=True):
    R3 = data["R3"]
    R4 = data["R4"]
    R5 = data["R5"]
    R6 = data["R6"]


    time = np.linspace(0, 1, len(R3))

    eff0_ttbar_const = R3[0]
    eff0_higgs_const = R5[0]
    eff0_ttbar_pd    = R4[0]
    eff0_higgs_pd    = R6[0]

    styles = {
        "Constant": {"linestyle": "dashed", "linewidth": 2.5},
        "PD":       {"linestyle": "solid",  "linewidth": 2.0},
    }
    colors = {
        "ttbar":     "goldenrod",
        "HToAATo4B": "seagreen",
    }

    plt.figure(figsize=(10, 6))

    # Constant Menu
    plt.plot(
        time, R3 / eff0_ttbar_const,
        label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar_const:.2f}$)",
        color=colors["ttbar"], **styles["Constant"]
    )
    plt.plot(
        time, R5 / eff0_higgs_const,
        label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_higgs_const:.2f}$)",
        color=colors["HToAATo4B"], **styles["Constant"]
    )

    # PD Controller
    plt.plot(
        time, R4 / eff0_ttbar_pd,
        label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_pd:.2f}$)",
        color=colors["ttbar"], **styles["PD"]
    )
    plt.plot(
        time, R6 / eff0_higgs_pd,
        label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_higgs_pd:.2f}$)",
        color=colors["HToAATo4B"], **styles["PD"]
    )

    plt.xlabel("Time (Fraction of Run)", loc='center')
    plt.ylabel("Relative Cumulative Efficiency", loc='center')
    plt.ylim(0.7, 2.0)
    plt.grid(True, linestyle="--", alpha=0.6)

    header_const = mlines.Line2D([], [], color="none", linestyle="none")
    header_pd    = mlines.Line2D([], [], color="none", linestyle="none")

    const_ttbar = mlines.Line2D([], [], color=colors["ttbar"],     **styles["Constant"])
    const_higgs = mlines.Line2D([], [], color=colors["HToAATo4B"], **styles["Constant"])
    pd_ttbar    = mlines.Line2D([], [], color=colors["ttbar"],     **styles["PD"])
    pd_higgs    = mlines.Line2D([], [], color=colors["HToAATo4B"], **styles["PD"])

    handles = [
        header_const, const_ttbar, const_higgs,
        header_pd,    pd_ttbar,    pd_higgs,
    ]
    labels = [
        "Constant Menu",
        fr"ttbar ($\epsilon[t_0]={eff0_ttbar_const:.2f}$)",
        fr"HToAATo4B ($\epsilon[t_0]={eff0_higgs_const:.2f}$)",
        "PD Controller",
        fr"ttbar ($\epsilon[t_0]={eff0_ttbar_pd:.2f}$)",
        fr"HToAATo4B ($\epsilon[t_0]={eff0_higgs_pd:.2f}$)",
    ]

    plt.legend(
        handles, labels,
        title="HT Trigger",
        fontsize=16,
        ncol=2, loc="best", frameon=True,
        handlelength=2, columnspacing=0.8, labelspacing=0.9
    )

    plt.tight_layout()
    if isMC:
        plt.savefig(f"{OUTDIR}/sht_rate_pidMC.pdf", bbox_inches='tight')
    else: 
        plt.savefig(f"{OUTDIR}/sht_rate_pidData.pdf", bbox_inches='tight')

    #plt.savefig(os.path.join(OUTDIR, "sht_rate_pidMC2.pdf"), bbox_inches='tight')
    plt.close()


def plot_ht_cumulative_eff_fixed(data, isMC=True):
    """HT cumulative signal efficiency, constant menu only."""
    R3 = np.asarray(data["R3"])  # ttbar
    R5 = np.asarray(data["R5"])  # HToAATo4B
    eff0_ttbar = R3[0]
    eff0_AA = R5[0]

    time = np.linspace(0, 1, len(R3))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(time, R3/R3[0], label=fr'ttbar ($\epsilon[t_0]$ = {eff0_ttbar:.2f})', color='goldenrod', linewidth=2.5, linestyle='dashed')
    plt.plot(time, R5/R5[0], label=fr'HToAATo4B ($\epsilon[t_0]$ = {eff0_AA:.2f})', color='seagreen', linewidth=2.5, linestyle='dashed')

    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Relative Cumulative Efficiency', loc='center')
    plt.legend(title='HT Constant Menu', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    if isMC:
        plt.savefig(f"{OUTDIR}/sht_rate_fixedMC.pdf", bbox_inches='tight')
    else: 
        plt.savefig(f"{OUTDIR}/sht_rate_fixedData.pdf", bbox_inches='tight')

    #fig.savefig(OUTDIR / "sht_rate_fixed2.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_ht_local_eff(data, isMC=True):
    L_R3 = data["L_R3"]
    L_R4 = data["L_R4"]
    L_R5 = data["L_R5"]
    L_R6 = data["L_R6"]

    time = np.linspace(0, 1, len(L_R3))

    eff0_ttbar_const = L_R3[0]
    eff0_ttbar_pd    = L_R4[0]
    eff0_higgs_const = L_R5[0]
    eff0_higgs_pd    = L_R6[0]

    styles = {
        "Constant": {"linestyle": "dashed", "linewidth": 2.5},
        "PD":       {"linestyle": "solid",  "linewidth": 2.5},
    }
    colors = {
        "ttbar":     "goldenrod",
        "HToAATo4B": "seagreen",
    }

    plt.figure(figsize=(10, 6))

    plt.plot(time, L_R3 / eff0_ttbar_const,
             label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar_const:.2f}$)",
             color=colors["ttbar"], **styles["Constant"])
    plt.plot(time, L_R5 / eff0_higgs_const,
             label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_higgs_const:.2f}$)",
             color=colors["HToAATo4B"], **styles["Constant"])

    plt.plot(time, L_R4 / eff0_ttbar_pd,
             label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_pd:.2f}$)",
             color=colors["ttbar"], **styles["PD"])
    plt.plot(time, L_R6 / eff0_higgs_pd,
             label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_higgs_pd:.2f}$)",
             color=colors["HToAATo4B"], **styles["PD"])

    plt.xlabel("Time (Fraction of Run)", loc='center')
    plt.ylabel("Relative Efficiency",   loc='center')
    plt.ylim(0.5, 2.5)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend(
        title="HT Trigger",
        ncol=1, loc="best",
        frameon=True,
        handlelength=2, columnspacing=1, labelspacing=0.8,
        fontsize=16,
    )

    plt.tight_layout()
    if isMC:
        plt.savefig(f"{OUTDIR}/L_sht_rate_pidMC.pdf", bbox_inches='tight')
    else: 
        plt.savefig(f"{OUTDIR}/L_sht_rate_pidData.pdf", bbox_inches='tight')

    #plt.savefig(os.path.join(OUTDIR, "L_sht_rate_pidMC2.pdf"), bbox_inches='tight')
    plt.close()

def plot_ht_local_eff_fixed(data, isMC=True):
    """HT local signal efficiency, constant menu only."""
    L_R3 = np.asarray(data["L_R3"])  # ttbar
    L_R5 = np.asarray(data["L_R5"])  # HToAATo4B


    time = np.linspace(0, 1, len(L_R3))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(time, L_R3/L_R3[0], label='ttbar', color='goldenrod', linewidth=2.5, linestyle='dashed')
    plt.plot(time, L_R5/L_R5[0], label='HToAATo4B', color='seagreen', linewidth=2.5, linestyle='dashed')
    plt.ylim(0,1.2)
    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Relative Efficiency', loc='center')
    plt.legend(title='HT Constant Menu', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if isMC:
        plt.savefig(f"{OUTDIR}/L_sht_rate_fixedMC.pdf", bbox_inches='tight')
    else: 
        plt.savefig(f"{OUTDIR}/L_sht_rate_fixedData.pdf", bbox_inches='tight')

    #fig.savefig(OUTDIR / "L_sht_rate_fixed.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# 4) AD SIGNAL EFFICIENCY (cumulative + local, fixed & PD)
# ----------------------------------------------------------------------

styles = {
    "Constant": {"linestyle": "dashed", "linewidth": 2.5},
    "PD":       {"linestyle": "solid",  "linewidth": 1.8},
}
def plot_ad_cumulative_eff_fixed(data, isMC=True):
    """
    AD cumulative signal efficiency – single AE dimension (dim=2 setup),
    constant menu only.
    """
    colors = {
        "ttbar": "goldenrod",
        "HToAATo4B": "limegreen",
    }

    E3 = np.asarray(data["E3"])  # ttbar, constant
    E5 = np.asarray(data["E5"])  # AA, constant


    eff0_ttbar = E3[0]
    eff0_AA    = E5[0]

    time = np.linspace(0, 1, len(E3))

    plt.figure(figsize=(10, 6))
    plt.plot(time, E3 / eff0_ttbar,
             label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar:.2f}\%$)",
             color=colors["ttbar"], **styles["Constant"])
    plt.plot(time, E5 / eff0_AA,
             label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA:.2f}\%$)",
             color=colors["HToAATo4B"], **styles["Constant"])

    plt.xlabel("Time (Fraction of Run)", loc="center")
    plt.ylabel("Relative Cumulative Efficiency", loc="center")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0.8, 1.2)
    plt.legend(title="AD Trigger", fontsize=15, frameon=True, loc="best")

    out = "sas_rate_fixedMC.pdf" if isMC else "sas_rate_fixedData.pdf"
    plt.savefig(f"{OUTDIR}/{out}", bbox_inches="tight", dpi=300)
    plt.close()

def plot_ad_cumulative_eff(data, isMC=True):
    """
    AD cumulative signal efficiency.
    """

    # --- Load arrays (single dim only) ---
    E3 = np.asarray(data["E3"]).copy()  # ttbar, constant
    E4 = np.asarray(data["E4"]).copy()  # ttbar, PD
    E5 = np.asarray(data["E5"]).copy()  # HToAATo4B, constant
    E6 = np.asarray(data["E6"]).copy()  # HToAATo4B, PD




    time = np.linspace(0, 1, len(E3))

    eff0_ttbar_const = E3[0]
    eff0_AA_const    = E5[0]
    eff0_ttbar_pd    = E4[0]
    eff0_AA_pd       = E6[0]

    colors = {
        "ttbar": "goldenrod",
        "HToAATo4B": "limegreen",
    }

    plt.figure(figsize=(10, 6))

    # --- Constant Menu ---
    plt.plot(
        time, E3 / eff0_ttbar_const,
        label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar_const:.2f}\%$)",
        color=colors["ttbar"], **styles["Constant"]
    )
    plt.plot(
        time, E5 / eff0_AA_const,
        label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA_const:.2f}\%$)",
        color=colors["HToAATo4B"], **styles["Constant"]
    )

    # --- PD Controller (optional) ---
    plt.plot(
    time, E4 / eff0_ttbar_pd,
    label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_pd:.2f}\%$)",
    color=colors["ttbar"], **styles["PD"]
    )
    plt.plot(
    time, E6 / eff0_AA_pd,
    label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_pd:.2f}\%$)",
        color=colors["HToAATo4B"], **styles["PD"]
    )

    plt.xlabel("Time (Fraction of Run)", loc="center")
    plt.ylabel("Relative Cumulative Efficiency", loc="center")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0.7, 2.5)
    plt.legend(title="AD Trigger", fontsize=15, frameon=True, loc="best")

    out = "sas_rate_pidMC.pdf" if isMC else "sas_rate_pidData.pdf"
    plt.savefig(f"{OUTDIR}/{out}", bbox_inches="tight", dpi=300)
    plt.close()


def plot_ad_local_eff(data, isMC=True):
    """
    AD local signal efficiency)

    """

    # --- Load arrays (single dim only) ---
    L_E3 = np.asarray(data["L_E3"])  # ttbar, constant
    L_E4 = np.asarray(data["L_E4"])  # ttbar, PD
    L_E5 = np.asarray(data["L_E5"])  # HToAATo4B, constant
    L_E6 = np.asarray(data["L_E6"])  # HToAATo4B, PD


    time = np.linspace(0, 1, len(L_E3))

    eff0_ttbar_const = L_E3[0]
    eff0_AA_const    = L_E5[0]
    eff0_ttbar_pd    = L_E4[0]
    eff0_AA_pd       = L_E6[0]

    colors = {
        "ttbar": "goldenrod",
        "HToAATo4B": "limegreen",
    }

    plt.figure(figsize=(10, 6))

    # --- Constant Menu ---
    plt.plot(
        time, L_E3 / eff0_ttbar_const,
        label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar_const:.2f}\%$)",
        color=colors["ttbar"], **styles["Constant"]
    )
    plt.plot(
        time, L_E5 / eff0_AA_const,
        label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA_const:.2f}\%$)",
        color=colors["HToAATo4B"], **styles["Constant"]
    )

    
    plt.plot(
    time, L_E4 / eff0_ttbar_pd,
    label=fr"PD Controller, ttbar ($\epsilon[t_0]={eff0_ttbar_pd:.2f}\%$)",
    color=colors["ttbar"], **styles["PD"])
    plt.plot(
    time, L_E6 / eff0_AA_pd,
    label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={eff0_AA_pd:.2f}\%$)",
    color=colors["HToAATo4B"], **styles["PD"]
    )

    plt.xlabel("Time (Fraction of Run)", loc="center")
    plt.ylabel("Relative Efficiency", loc="center")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0.6, 2.5)

    plt.legend(
        title="AD Trigger",
        fontsize=15,
        frameon=True,
        loc="best"
    )

    out = "L_sas_rate_pidMC.pdf" if isMC else "L_sas_rate_pidData.pdf"
    plt.savefig(f"{OUTDIR}/{out}", bbox_inches="tight", dpi=300)
    plt.close()


def plot_ad_local_eff_fixed(data, isMC=True):
    """
    AD local signal efficiency – single AE dimension,
    constant menu only.
    """
    colors = {
        "ttbar": "goldenrod",
        "HToAATo4B": "limegreen",
    }

    L_E3 = np.asarray(data["L_E3"])  # ttbar
    L_E5 = np.asarray(data["L_E5"])  # AA

    n = min(len(L_E3), len(L_E5))
    L_E3, L_E5 = L_E3[:n], L_E5[:n]

    eff0_ttbar = L_E3[0]
    eff0_AA    = L_E5[0]

    time = np.linspace(0, 1, n)

    plt.figure(figsize=(10, 6))
    plt.plot(time, L_E3 / eff0_ttbar,
             label=fr"Constant Menu, ttbar ($\epsilon[t_0]={eff0_ttbar:.2f}\%$)",
             color=colors["ttbar"], **styles["Constant"])
    plt.plot(time, L_E5 / eff0_AA,
             label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={eff0_AA:.2f}\%$)",
             color=colors["HToAATo4B"], **styles["Constant"])

    plt.xlabel("Time (Fraction of Run)", loc="center")
    plt.ylabel("Relative Efficiency", loc="center")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0.6, 1.2)
    plt.legend(title="AD Trigger", fontsize=15, frameon=True, loc="best")

    out = "L_sas_rate_fixedMC.pdf" if isMC else "L_sas_rate_fixedData.pdf"
    plt.savefig(f"{OUTDIR}/{out}", bbox_inches="tight", dpi=300)
    plt.close()





# SingTig.py -> PID plots
import matplotlib
matplotlib.use("Agg")




def update_accumulated(acc_list, new_val, new_sample_size, total_samples_list):
    if not acc_list:
        acc_list.append(new_val)
        total_samples_list.append(new_sample_size)
    else:
        prev_total = total_samples_list[-1]
        tot = prev_total + new_sample_size
        new_avg = (acc_list[-1] * prev_total + new_val * new_sample_size) / max(1, tot)
        acc_list.append(new_avg)
        total_samples_list.append(tot)




def multi_path_panels(
    time, Id1_R, Id1_r_bht, Id1_r_bas,
    Id1_r1_s, Id1_r2_s, Id1_r1_sht, Id1_r1_sas,
    Id1_r2_sht, Id1_r2_sas, Id1_E, Id1_GE,
    out_pdf=None, case_label="Case 3"
):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # (a) Background rates (kHz)
    ax = axes[0,0]
    ax.plot(time, Id1_R,     label="Total Ideal Rate")
    ax.plot(time, Id1_r_bht, label="HT Ideal Rate", linestyle="--")
    ax.plot(time, Id1_r_bas, label="AD Ideal Rate", linestyle="-.")
    ax.set_xlabel("Time (Fraction of Run)"); ax.set_ylabel("Background Rate (kHz)")
    ax.set_xlim(0,1); ax.set_ylim(0,200); ax.legend(title=case_label); ax.grid()

    # (b) Total & cumulative signal eff
    ax = axes[0,1]
    ax.plot(time, Id1_E,  label="Total", linestyle="--")
    ax.plot(time, Id1_GE, label="Cumulative Total")
    ax.set_xlabel("Time (Fraction of Run)"); ax.set_ylabel("Combined Signal Efficiency (%)")
    ax.set_xlim(0,1); ax.legend(title=case_label); ax.grid()

    # (c) Signal2 pieces
    ax = axes[1,0]
    ax.plot(time, Id1_r2_s,   label="Total")
    ax.plot(time, Id1_r2_sht, label="from HT", linestyle="--")
    ax.plot(time, Id1_r2_sas, label="from AD", linestyle="-.")
    ax.set_xlabel("Time (Fraction of Run)"); ax.set_ylabel("HToAATo4B Efficiency (%)")
    ax.set_xlim(0,1); ax.legend(title=case_label); ax.grid()

    # (d) Signal1 pieces
    ax = axes[1,1]
    ax.plot(time, Id1_r1_s,   label="Total")
    ax.plot(time, Id1_r1_sht, label="from HT", linestyle="--")
    ax.plot(time, Id1_r1_sas, label="from AD", linestyle="-.")
    ax.set_xlabel("Time (Fraction of Run)"); ax.set_ylabel("TTbar Efficiency (%)")
    ax.set_xlim(0,1); ax.legend(title=case_label); ax.grid()

    plt.tight_layout()
    if out_pdf:
        fig.savefig(out_pdf)
    return fig, axes

def save_subplot(fig, ax, filename, pad=0.3):
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    bbox = mtransforms.Bbox.from_extents(
        bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad
    )
    fig.savefig(filename, bbox_inches=bbox)

def plot_evolution(Ht_vals, AS_vals, real_Ht_vals, real_AS_vals, title="real_evoultion_plot(r1)", out_pdf = ""):
    plt.figure(figsize=(8, 6))

    # Plot all points
    #plt.scatter(Ht_vals, AS_vals, c=np.arange(len(Ht_vals)), marker='o', cmap="viridis", label="ideal Points",s=50)
    plt.scatter(real_Ht_vals, real_AS_vals, c=np.arange(len(real_Ht_vals)),marker='+', cmap="viridis", label="Real-time Points",s=50)

    # Plot arrows showing evolution
    #for i in range(len(Ht_vals) - 1):
        #plt.quiver(Ht_vals[i], AS_vals[i], Ht_vals[i+1] - Ht_vals[i], AS_vals[i+1] - AS_vals[i], 
        # angles="xy", scale_units="xy", scale=1, color="green", alpha=0.6)

    # Highlight start and end points
    #plt.scatter(Ht_vals[0], AS_vals[0],marker='o', color="blue", s=100, label="ideal Start")
    #plt.scatter(Ht_vals[-1], AS_vals[-1],marker='o', color="red", s=100, label="ideal End")

    #plt.scatter(real_Ht_vals[0], real_AS_vals[0],marker='+', color="blue", s=100, label="real Start")
    #plt.scatter(real_Ht_vals[-1], real_AS_vals[-1],marker='+', color="red", s=100, label="real End")

    plt.xlabel("Ht Cut", fontsize=22)
    plt.ylabel("AD Cut", fontsize=22)
    plt.xlim(75,350)
    plt.ylim(75,250)
    plt.title(title, fontsize=22)
    plt.colorbar(label="Iteration Step")
    plt.legend(fontsize=16, loc='best', frameon=True)
    plt.grid(True)
    plt.savefig(out_pdf)


def plot_evolution_ideal(Ht_vals, AS_vals, real_Ht_vals, real_AS_vals, title="real_evoultion_plot(r1)", out_pdf = ""):
    plt.figure(figsize=(8, 6))

    # Plot all points
    plt.scatter(Ht_vals, AS_vals, c=np.arange(len(Ht_vals)), marker='o', cmap="viridis", label="ideal Points",s=50)
    # plt.scatter(real_Ht_vals, real_AS_vals, c=np.arange(len(real_Ht_vals)),marker='+', cmap="viridis", label="real-time Points",s=50)

    # Plot arrows showing evolution
    #for i in range(len(Ht_vals) - 1):
        #plt.quiver(Ht_vals[i], AS_vals[i], Ht_vals[i+1] - Ht_vals[i], AS_vals[i+1] - AS_vals[i], 
        # angles="xy", scale_units="xy", scale=1, color="green", alpha=0.6)

    # Highlight start and end points
    #plt.scatter(Ht_vals[0], AS_vals[0],marker='o', color="blue", s=100, label="ideal Start")
    #plt.scatter(Ht_vals[-1], AS_vals[-1],marker='o', color="red", s=100, label="ideal End")

    #plt.scatter(real_Ht_vals[0], real_AS_vals[0],marker='+', color="blue", s=100, label="real Start")
    #plt.scatter(real_Ht_vals[-1], real_AS_vals[-1],marker='+', color="red", s=100, label="real End")

    plt.xlabel("Ht Cut", fontsize=22)
    plt.ylabel("AD Cut", fontsize=22)
    plt.xlim(75,350)
    plt.title(title, fontsize=22)
    plt.colorbar(label="Iteration Step")
    plt.legend(fontsize=16, loc='best', frameon=True)
    plt.grid(True)
    plt.savefig(out_pdf)


def main():
    ap = argparse.ArgumentParser(description="Single Trigger Plots")

    ap.add_argument("--bkgType", default="MC", choices=["MC", "RealData"],
                    help="MC: Monte Carlo simulated samples MinBias bkg; RealData: real data background run. AA/TT are always MC.")

    args = ap.parse_args()
    bkgType = args.bkgType

    if bkgType=="MC" :
        npz_path = "Control/singletrigger_results_mc.npz"
        is_mc = True
    else:
        npz_path = "Control/singletrigger_results_realdata.npz"
        is_mc = False
    data = load_results(npz_path)
    
    plot_ht_background(data, is_mc)
    plot_ht_background_fixed(data, is_mc)
    
    plot_ad_background(data, is_mc)
    plot_ad_background_fixed(data, is_mc)

    plot_ht_cumulative_eff(data, is_mc)
    plot_ht_cumulative_eff_fixed(data, is_mc)
    
    plot_ht_local_eff_fixed(data, is_mc)
    plot_ht_local_eff(data, is_mc)

    plot_ad_cumulative_eff(data, is_mc)
    plot_ad_cumulative_eff_fixed(data, is_mc)
    
    plot_ad_local_eff(data, is_mc)
    plot_ad_local_eff_fixed(data, is_mc)


if __name__ == "__main__":
    main()
