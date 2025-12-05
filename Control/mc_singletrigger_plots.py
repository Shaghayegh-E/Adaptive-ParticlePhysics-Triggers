#!/usr/bin/env python
# singletrigger_plots.py
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import matplotlib.transforms as mtransforms

NPZ_PATH = "MC/mc_singletrigger_results.npz"
# OUTDIR = "outputs"
ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs" / "demo_singletrigger"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_results():
    return np.load(NPZ_PATH)

def plot_ht_background(data):
    R1 = data["R1"]
    R2 = data["R2"]

    time = np.linspace(0, 1, len(R1))
    plt.figure(figsize=(10, 6))
    plt.plot(time, R1, label='Constant Menu', color='tab:blue', linewidth=3, linestyle='dashed')
    plt.plot(time, R2, label='PD Controller', color='mediumblue', linewidth=2.5, linestyle='solid')

    plt.axhline(y=0.28 * 400, color='gray', linestyle='--', linewidth=1.5)
    plt.axhline(y=0.22 * 400, color='gray', linestyle='--', linewidth=1.5)

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


    # --- Prima legenda (sinistra) ---
    leg_main = ax.legend(
        handles_main, labels_main,
        title="HT Trigger",
        fontsize=17,
        ncol=1,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),  # ← sposta leggermente dentro il grafico
        frameon=True,
        handlelength=2, columnspacing=1, labelspacing=0.8
    )
    ax.add_artist(leg_main)

    # --- Seconda legenda (destra) ---
    #
    ## ---- Second legend for tolerance lines ----
    upper_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)
    lower_tol = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5)
    leg_tol = ax.legend(
        [upper_tol, lower_tol],
        ["Upper Tolerance (112)", "Lower Tolerance (88)"],
        title="Reference",
        fontsize=18,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),  # ← più interna verso sinistra
        frameon=True,
        handlelength=2
    )


    plt.savefig(f"{OUTDIR}/bht_rate_pidMC.pdf", bbox_inches='tight')
    plt.close()
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _time_axis(length: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, length)


def _tol_lines(ax, rate_scale=400.0):
    """Draw 22–28% tolerance band (fixed at 400 kHz input)."""
    upper = 0.28 * rate_scale
    lower = 0.22 * rate_scale
    ax.axhline(y=upper, color='gray', linestyle='--', linewidth=1.5)
    ax.axhline(y=lower, color='gray', linestyle='--', linewidth=1.5)
    return upper, lower


# ----------------------------------------------------------------------
# 1) BACKGROUND RATE: HT (fixed-only & PD) -> bht_*.pdf
# ----------------------------------------------------------------------
def plot_ht_background_fixed(data: dict):
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

    plt.axhline(y=0.28*400, color='gray', linestyle='--', linewidth=1.5, label='Upper Tolerance (112)')
    plt.axhline(y=0.22*400, color='gray', linestyle='--', linewidth=1.5, label='Lower Tolerance (88)')

    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Background Rate [kHz]', loc='center')
    plt.legend(title='HT Trigger', loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(OUTDIR / 'bht_rate_fixed.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_ad_background(data):
    E1_1 = data["E1_1"]
    E1_4 = data["E1_4"]
    E2_1 = data["E2_1"]
    E2_4 = data["E2_4"]

    time = np.linspace(0, 1, len(E1_1))
    plt.figure(figsize=(10, 6))

    plt.plot(time, E1_1, label='Constant Menu, model dim=1',
             color='mediumblue', linewidth=3, linestyle='dotted')
    plt.plot(time, E1_4, label='Constant Menu, model dim=4',
             color='cyan', linewidth=3, linestyle='dashed')
    plt.plot(time, E2_1, label='PD Controller, model dim=1',
             color='mediumblue', linewidth=2.5, linestyle='solid')
    plt.plot(time, E2_4, label='PD Controller, model dim=4',
             color='cyan', linewidth=2.5, linestyle='solid')

    plt.axhline(y=0.28*400, color='gray', linestyle='--', linewidth=1.5)
    plt.axhline(y=0.22*400, color='gray', linestyle='--', linewidth=1.5)

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
    const_dim1 = mlines.Line2D([], [], color='tab:blue',    linestyle='dotted', linewidth=3)
    pd_dim1    = mlines.Line2D([], [], color='mediumblue',  linestyle='solid',  linewidth=2.5)
    const_dim4 = mlines.Line2D([], [], color='tab:blue',    linestyle='dashed', linewidth=3)
    pd_dim4    = mlines.Line2D([], [], color='cyan',        linestyle='solid',  linewidth=2.5)

    handles_main = [
        header_const,   
        const_dim1,   
        const_dim4,  
        header_pd,
        pd_dim1,   
        pd_dim4    
    ]
    labels_main = [
        "Constant Menu", 
        "model dim=1",   
        "model dim=4",
        "PD Controller",
        "model dim=1",  
        "model dim=4"
    ]

    # --- Prima legenda (sinistra) ---
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
    # --- Seconda legenda (destra) ---
    leg_tol = ax.legend(
        [upper_tol, lower_tol],
        ["Upper Tolerance (112)", "Lower Tolerance (88)"],
        title="Reference",
        fontsize=15,
        loc='upper right',
        #bbox_to_anchor=(0.98, 0.98),  # ← più interna verso sinistra
        frameon=True,
        handlelength=2
    )
    plt.ylim(0, 270)

    plt.savefig(f"{OUTDIR}/bas_rate_pidMC.pdf", bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------------------
# 2) BACKGROUND RATE: AD (fixed-only & PD) -> bas_*.pdf
# ----------------------------------------------------------------------
def plot_ad_background_fixed(data: dict):
    """AD background – constant menu only (dim=1,4 fixed cuts)."""
    E1_1 = np.asarray(data["E1_1"])
    E1_4 = np.asarray(data["E1_4"])
    time = np.linspace(0, 1, len(E1_1))
    plt.figure(figsize=(10, 6))
    plt.plot(time, E1_1, label='Constant Menu, model dim=1', color='tab:blue', linewidth=3, linestyle='dotted')
    plt.plot(time, E1_4, label='Constant Menu, model dim=4', color='tab:blue', linewidth=3, linestyle='dashed')

    plt.axhline(y=0.28*400, color='gray', linestyle='--', linewidth=1.5, label='Upper Tolerance (112)')
    plt.axhline(y=0.22*400, color='gray', linestyle='--', linewidth=1.5, label='Lower Tolerance (88)')

    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Background Rate [kHz]', loc='center')
    plt.ylim(0,300)
    plt.legend(title='AD Trigger', loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(OUTDIR / 'bas_rate_fixed.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_ht_cumulative_eff(data):
    R3 = data["R3"]
    R4 = data["R4"]
    R5 = data["R5"]
    R6 = data["R6"]

    n = min(len(R3), len(R4), len(R5), len(R6))
    R3 = np.array(R3[:n]); R4 = np.array(R4[:n])
    R5 = np.array(R5[:n]); R6 = np.array(R6[:n])

    time = np.linspace(0, 1, n)

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
    plt.savefig(os.path.join(OUTDIR, "sht_rate_pidMC2.pdf"), bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------------------
# 3) HT SIGNAL EFFICIENCY (cumulative + local, fixed & PD)
# ----------------------------------------------------------------------
def plot_ht_cumulative_eff_fixed(data: dict):
    """HT cumulative signal efficiency, constant menu only."""
    R3 = np.asarray(data["R3"])  # ttbar
    R5 = np.asarray(data["R5"])  # HToAATo4B
    eff0_ttbar = R3[0]
    eff0_AA = R5[0]
    n = min(len(R3), len(R5))
    R3, R5 = R3[:n], R5[:n]
    t = _time_axis(n)
    time = np.linspace(0, 1, len(R3))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(time, R3/R3[0], label=fr'ttbar ($\epsilon[t_0]$ = {eff0_ttbar:.2f})', color='goldenrod', linewidth=2.5, linestyle='dashed')
    plt.plot(time, R5/R5[0], label=fr'HToAATo4B ($\epsilon[t_0]$ = {eff0_AA:.2f})', color='seagreen', linewidth=2.5, linestyle='dashed')

    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Relative Cumulative Efficiency', loc='center')
    plt.legend(title='HT Constant Menu', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    fig.savefig(OUTDIR / "sht_rate_fixed2.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_ht_local_eff(data):
    L_R3 = data["L_R3"]
    L_R4 = data["L_R4"]
    L_R5 = data["L_R5"]
    L_R6 = data["L_R6"]

    n = min(len(L_R3), len(L_R4), len(L_R5), len(L_R6))
    L_R3 = np.array(L_R3[:n]); L_R4 = np.array(L_R4[:n])
    L_R5 = np.array(L_R5[:n]); L_R6 = np.array(L_R6[:n])

    time = np.linspace(0, 1, n)

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
    plt.savefig(os.path.join(OUTDIR, "L_sht_rate_pidMC2.pdf"), bbox_inches='tight')
    plt.close()

def plot_ht_local_eff_fixed(data: dict):
    """HT local signal efficiency, constant menu only."""
    L_R3 = np.asarray(data["L_R3"])  # ttbar
    L_R5 = np.asarray(data["L_R5"])  # HToAATo4B
    n = min(len(L_R3), len(L_R5))
    L_R3, L_R5 = L_R3[:n], L_R5[:n]
    t = _time_axis(n)
    time = np.linspace(0, 1, len(L_R3))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(time, L_R3/L_R3[0], label='ttbar', color='goldenrod', linewidth=2.5, linestyle='dashed')
    plt.plot(time, L_R5/L_R5[0], label='HToAATo4B', color='seagreen', linewidth=2.5, linestyle='dashed')
    plt.ylim(0,1.2)
    plt.xlabel('Time (Fraction of Run)', loc='center')
    plt.ylabel('Relative Efficiency', loc='center')
    plt.legend(title='HT Constant Menu', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)

    fig.savefig(OUTDIR / "L_sht_rate_fixed.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# 4) AD SIGNAL EFFICIENCY (cumulative + local, fixed & PD)
# ----------------------------------------------------------------------
def plot_ad_cumulative_eff_fixed(data: dict):
    """AD cumulative signal efficiency, constant menu only."""
    E3_1 = np.asarray(data["E3_1"])
    E3_4 = np.asarray(data["E3_4"])
    E5_1 = np.asarray(data["E5_1"])
    E5_4 = np.asarray(data["E5_4"])
    n = min(len(E3_1), len(E3_4), len(E5_1), len(E5_4))
    E3_1, E3_4, E5_1, E5_4 = E3_1[:n], E3_4[:n], E5_1[:n], E5_4[:n]
    t = _time_axis(n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, E3_1 / E3_1[0], label="ttbar, dim=1",
            color="goldenrod", linewidth=2.5, linestyle="dashed")
    ax.plot(t, E3_4 / E3_4[0], label="ttbar, dim=4",
            color="orangered", linewidth=2.5, linestyle="dashed")
    ax.plot(t, E5_1 / E5_1[0], label="HToAATo4B, dim=1",
            color="limegreen", linewidth=2.5, linestyle="dashed")
    ax.plot(t, E5_4 / E5_4[0], label="HToAATo4B, dim=4",
            color="seagreen", linewidth=2.5, linestyle="dashed")

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Cumulative Efficiency", loc="center")
    ax.set_ylim(0.5, 2.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="AD Constant Menu", frameon=True, loc="best")

    fig.savefig(OUTDIR / "sas_rate_fixed.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_ad_cumulative_eff_fixed_relative(data: dict):
    """AD cumulative signal efficiency, constant menu only."""
    E3_1 = np.asarray(data["E3_1"])
    E3_4 = np.asarray(data["E3_4"])
    E5_1 = np.asarray(data["E5_1"])
    E5_4 = np.asarray(data["E5_4"])
    n = min(len(E3_1), len(E3_4), len(E5_1), len(E5_4))
    E3_1, E3_4, E5_1, E5_4 = E3_1[:n], E3_4[:n], E5_1[:n], E5_4[:n]
    t = _time_axis(n)
    eff0_ttbar1 = E3_1[0]
    eff0_ttbar4 = E3_4[0]
    eff0_AA1 = E5_1[0]
    eff0_AA4 = E5_4[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, E3_1 / E3_1[0], label=fr'ttbar (dim=1, $\epsilon[t_0]$={eff0_ttbar1:.2f})',
            color="goldenrod", linewidth=2.5, linestyle="dashed")
    ax.plot(t, E3_4 / E3_4[0], label=fr'ttbar (dim=4, $\epsilon[t_0]$={eff0_ttbar4:.2f})',
            color="orangered", linewidth=2.5, linestyle="dashed")
    ax.plot(t, E5_1 / E5_1[0], label=fr'HToAATo4B (dim=1, $\epsilon[t_0]$={eff0_AA1:.2f})',
            color="limegreen", linewidth=2.5, linestyle="dashed")
    ax.plot(t, E5_4 / E5_4[0], label=fr'HToAATo4B (dim=4, $\epsilon[t_0]$={eff0_AA4:.2f})',
            color="seagreen", linewidth=2.5, linestyle="dashed")

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Cumulative Efficiency", loc="center")
    ax.set_ylim(0.5, 2.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="AD Constant Menu", frameon=True, loc="best")

    fig.savefig(OUTDIR / "sas_rate_fixed2.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_ad_cumulative_eff(data):
    # Copy so we don't mutate the npz-backed arrays
    E3_1 = data["E3_1"].copy()
    E3_4 = data["E3_4"].copy()
    E4_1 = data["E4_1"].copy()
    E4_4 = data["E4_4"].copy()
    E5_1 = data["E5_1"].copy()
    E5_4 = data["E5_4"].copy()
    E6_1 = data["E6_1"].copy()
    E6_4 = data["E6_4"].copy()

    # Drop first element, like in notebook (warm-up)
    def drop_first(x):
        return x[1:] if len(x) > 1 else x

    E3_1 = drop_first(E3_1); E3_4 = drop_first(E3_4)
    E4_1 = drop_first(E4_1); E4_4 = drop_first(E4_4)
    E5_1 = drop_first(E5_1); E5_4 = drop_first(E5_4)
    E6_1 = drop_first(E6_1); E6_4 = drop_first(E6_4)

    n = min(len(E3_1), len(E3_4), len(E4_1), len(E4_4),
            len(E5_1), len(E5_4), len(E6_1), len(E6_4))

    E3_1 = np.array(E3_1[:n]); E3_4 = np.array(E3_4[:n])
    E4_1 = np.array(E4_1[:n]); E4_4 = np.array(E4_4[:n])
    E5_1 = np.array(E5_1[:n]); E5_4 = np.array(E5_4[:n])
    E6_1 = np.array(E6_1[:n]); E6_4 = np.array(E6_4[:n])

    time = np.linspace(0, 1, n)

    eff0_ttbar1     = E3_1[0]
    eff0_ttbar4     = E3_4[0]
    eff0_AA1        = E5_1[0]
    eff0_AA4        = E5_4[0]
    eff0_ttbar1_PD  = E4_1[0]
    eff0_ttbar4_PD  = E4_4[0]
    eff0_AA1_PD     = E6_1[0]
    eff0_AA4_PD     = E6_4[0]

    styles = {
        "Constant": {"linestyle": "dashed", "linewidth": 2.5},
        "PD":       {"linestyle": "solid",  "linewidth": 1.8},
    }
    colors = {
        "ttbar_dim1": "goldenrod",
        "ttbar_dim4": "orangered",
        "higgs_dim1": "limegreen",
        "higgs_dim4": "seagreen",
    }

    plt.figure(figsize=(10, 6))

    # Constant Menu
    plt.plot(time, E3_1 / eff0_ttbar1,
             label=fr"Constant Menu, ttbar (dim=1, $\epsilon[t_0]={eff0_ttbar1:.2f}$)",
             color=colors["ttbar_dim1"], **styles["Constant"])
    plt.plot(time, E3_4 / eff0_ttbar4,
             label=fr"Constant Menu, ttbar (dim=4, $\epsilon[t_0]={eff0_ttbar4:.2f}$)",
             color=colors["ttbar_dim4"], **styles["Constant"])
    plt.plot(time, E5_1 / eff0_AA1,
             label=fr"Constant Menu, HToAATo4B (dim=1, $\epsilon[t_0]={eff0_AA1:.2f}$)",
             color=colors["higgs_dim1"], **styles["Constant"])
    plt.plot(time, E5_4 / eff0_AA4,
             label=fr"Constant Menu, HToAATo4B (dim=4, $\epsilon[t_0]={eff0_AA4:.2f}$)",
             color=colors["higgs_dim4"], **styles["Constant"])

    # PD Controller
    plt.plot(time, E4_1 / eff0_ttbar1_PD,
             label=fr"PD Controller, ttbar (dim=1, $\epsilon[t_0]={eff0_ttbar1_PD:.2f}$)",
             color=colors["ttbar_dim1"], **styles["PD"])
    plt.plot(time, E4_4 / eff0_ttbar4_PD,
             label=fr"PD Controller, ttbar (dim=4, $\epsilon[t_0]={eff0_ttbar4_PD:.2f}$)",
             color=colors["ttbar_dim4"], **styles["PD"])
    plt.plot(time, E6_1 / eff0_AA1_PD,
             label=fr"PD Controller, HToAATo4B (dim=1, $\epsilon[t_0]={eff0_AA1_PD:.2f}$)",
             color=colors["higgs_dim1"], **styles["PD"])
    plt.plot(time, E6_4 / eff0_AA4_PD,
             label=fr"PD Controller, HToAATo4B (dim=4, $\epsilon[t_0]={eff0_AA4_PD:.2f}$)",
             color=colors["higgs_dim4"], **styles["PD"])

    plt.xlabel("Time (Fraction of Run)", loc='center')
    plt.ylabel("Relative Cumulative Efficiency", loc='center')
    plt.ylim(0.6, 3.2)
    plt.grid(True, linestyle="--", alpha=0.6)

    header_const = mlines.Line2D([], [], color="none", linestyle="none")
    header_pd    = mlines.Line2D([], [], color="none", linestyle="none")

    const_ttbar_dim1 = mlines.Line2D([], [], color=colors["ttbar_dim1"], **styles["Constant"])
    const_ttbar_dim4 = mlines.Line2D([], [], color=colors["ttbar_dim4"], **styles["Constant"])
    const_higgs_dim1 = mlines.Line2D([], [], color=colors["higgs_dim1"], **styles["Constant"])
    const_higgs_dim4 = mlines.Line2D([], [], color=colors["higgs_dim4"], **styles["Constant"])
    pd_ttbar_dim1    = mlines.Line2D([], [], color=colors["ttbar_dim1"], **styles["PD"])
    pd_ttbar_dim4    = mlines.Line2D([], [], color=colors["ttbar_dim4"], **styles["PD"])
    pd_higgs_dim1    = mlines.Line2D([], [], color=colors["higgs_dim1"], **styles["PD"])
    pd_higgs_dim4    = mlines.Line2D([], [], color=colors["higgs_dim4"], **styles["PD"])

    handles = [
        header_const,
        const_ttbar_dim1, const_ttbar_dim4, const_higgs_dim1, const_higgs_dim4,
        header_pd,
        pd_ttbar_dim1, pd_ttbar_dim4, pd_higgs_dim1, pd_higgs_dim4,
    ]
    labels = [
        "Constant Menu",
        fr"ttbar (dim=1, $\epsilon[t_0]={eff0_ttbar1:.2f}$)",
        fr"ttbar (dim=4, $\epsilon[t_0]={eff0_ttbar4:.2f}$)",
        fr"HToAATo4B (dim=1, $\epsilon[t_0]={eff0_AA1:.2f}$)",
        fr"HToAATo4B (dim=4, $\epsilon[t_0]={eff0_AA4:.2f}$)",
        "PD Controller",
        fr"ttbar (dim=1, $\epsilon[t_0]={eff0_ttbar1_PD:.2f}$)",
        fr"ttbar (dim=4, $\epsilon[t_0]={eff0_ttbar4_PD:.2f}$)",
        fr"HToAATo4B (dim=1, $\epsilon[t_0]={eff0_AA1_PD:.2f}$)",
        fr"HToAATo4B (dim=4, $\epsilon[t_0]={eff0_AA4_PD:.2f}$)",
    ]

    plt.legend(
        handles, labels,
        title="AD Trigger",
        ncol=2, loc="best", frameon=True,
        handlelength=2, columnspacing=1, labelspacing=0.8,
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "sas_rate_pidMC2.pdf"), bbox_inches="tight")
    plt.close()


def plot_ad_local_eff(data):
    L_E3_1 = data["L_E3_1"]
    L_E3_4 = data["L_E3_4"]
    L_E4_1 = data["L_E4_1"]
    L_E4_4 = data["L_E4_4"]
    L_E5_1 = data["L_E5_1"]
    L_E5_4 = data["L_E5_4"]
    L_E6_1 = data["L_E6_1"]
    L_E6_4 = data["L_E6_4"]

    n = min(len(L_E3_1), len(L_E3_4), len(L_E4_1), len(L_E4_4),
            len(L_E5_1), len(L_E5_4), len(L_E6_1), len(L_E6_4))

    L_E3_1 = np.array(L_E3_1[:n]); L_E3_4 = np.array(L_E3_4[:n])
    L_E4_1 = np.array(L_E4_1[:n]); L_E4_4 = np.array(L_E4_4[:n])
    L_E5_1 = np.array(L_E5_1[:n]); L_E5_4 = np.array(L_E5_4[:n])
    L_E6_1 = np.array(L_E6_1[:n]); L_E6_4 = np.array(L_E6_4[:n])

    time = np.linspace(0, 1, n)

    eff0_ttbar1_const = L_E3_1[0]
    eff0_ttbar4_const = L_E3_4[0]
    eff0_higgs1_const = L_E5_1[0]
    eff0_higgs4_const = L_E5_4[0]
    eff0_ttbar1_pd    = L_E4_1[0]
    eff0_ttbar4_pd    = L_E4_4[0]
    eff0_higgs1_pd    = L_E6_1[0]
    eff0_higgs4_pd    = L_E6_4[0]

    styles = {
        "Constant": {"linestyle": "dashed", "linewidth": 2.5},
        "PD":       {"linestyle": "solid",  "linewidth": 1.5},
    }
    colors = {
        "ttbar_dim1": "goldenrod",
        "ttbar_dim4": "orangered",
        "higgs_dim1": "limegreen",
        "higgs_dim4": "seagreen",
    }

    plt.figure(figsize=(10, 6))

    # Constant
    plt.plot(time, L_E3_1 / eff0_ttbar1_const,
             label=fr"Constant Menu, ttbar (dim=1, $\epsilon[t_0]={eff0_ttbar1_const:.2f}$)",
             color=colors["ttbar_dim1"], **styles["Constant"])
    plt.plot(time, L_E3_4 / eff0_ttbar4_const,
             label=fr"Constant Menu, ttbar (dim=4, $\epsilon[t_0]={eff0_ttbar4_const:.2f}$)",
             color=colors["ttbar_dim4"], **styles["Constant"])
    plt.plot(time, L_E5_1 / eff0_higgs1_const,
             label=fr"Constant Menu, HToAATo4B (dim=1, $\epsilon[t_0]={eff0_higgs1_const:.2f}$)",
             color=colors["higgs_dim1"], **styles["Constant"])
    plt.plot(time, L_E5_4 / eff0_higgs4_const,
             label=fr"Constant Menu, HToAATo4B (dim=4, $\epsilon[t_0]={eff0_higgs4_const:.2f}$)",
             color=colors["higgs_dim4"], **styles["Constant"])

    # PD
    plt.plot(time, L_E4_1 / eff0_ttbar1_pd,
             label=fr"PD Controller, ttbar (dim=1, $\epsilon[t_0]={eff0_ttbar1_pd:.2f}$)",
             color=colors["ttbar_dim1"], **styles["PD"])
    plt.plot(time, L_E4_4 / eff0_ttbar4_pd,
             label=fr"PD Controller, ttbar (dim=4, $\epsilon[t_0]={eff0_ttbar4_pd:.2f}$)",
             color=colors["ttbar_dim4"], **styles["PD"])
    plt.plot(time, L_E6_1 / eff0_higgs1_pd,
             label=fr"PD Controller, HToAATo4B (dim=1, $\epsilon[t_0]={eff0_higgs1_pd:.2f}$)",
             color=colors["higgs_dim1"], **styles["PD"])
    plt.plot(time, L_E6_4 / eff0_higgs4_pd,
             label=fr"PD Controller, HToAATo4B (dim=4, $\epsilon[t_0]={eff0_higgs4_pd:.2f}$)",
             color=colors["higgs_dim4"], **styles["PD"])

    plt.xlabel("Time (Fraction of Run)", loc='center')
    plt.ylabel("Relative Efficiency",   loc='center')
    plt.ylim(0.5, 4.3)
    plt.grid(True, linestyle="--", alpha=0.6)

    header_const = mlines.Line2D([], [], color="none", linestyle="none")
    header_pd    = mlines.Line2D([], [], color="none", linestyle="none")

    const_ttbar_dim1 = mlines.Line2D([], [], color=colors["ttbar_dim1"], **styles["Constant"])
    const_ttbar_dim4 = mlines.Line2D([], [], color=colors["ttbar_dim4"], **styles["Constant"])
    const_higgs_dim1 = mlines.Line2D([], [], color=colors["higgs_dim1"], **styles["Constant"])
    const_higgs_dim4 = mlines.Line2D([], [], color=colors["higgs_dim4"], **styles["Constant"])
    pd_ttbar_dim1    = mlines.Line2D([], [], color=colors["ttbar_dim1"], **styles["PD"])
    pd_ttbar_dim4    = mlines.Line2D([], [], color=colors["ttbar_dim4"], **styles["PD"])
    pd_higgs_dim1    = mlines.Line2D([], [], color=colors["higgs_dim1"], **styles["PD"])
    pd_higgs_dim4    = mlines.Line2D([], [], color=colors["higgs_dim4"], **styles["PD"])

    handles = [
        header_const,
        const_ttbar_dim1, const_ttbar_dim4, const_higgs_dim1, const_higgs_dim4,
        header_pd,
        pd_ttbar_dim1, pd_ttbar_dim4, pd_higgs_dim1, pd_higgs_dim4,
    ]
    labels = [
        "Constant Menu",
        fr"ttbar (dim=1, $\epsilon[t_0]={eff0_ttbar1_const:.2f}$)",
        fr"ttbar (dim=4, $\epsilon[t_0]={eff0_ttbar4_const:.2f}$)",
        fr"HToAATo4B (dim=1, $\epsilon[t_0]={eff0_higgs1_const:.2f}$)",
        fr"HToAATo4B (dim=4, $\epsilon[t_0]={eff0_higgs4_const:.2f}$)",
        "PD Controller",
        fr"ttbar (dim=1, $\epsilon[t_0]={eff0_ttbar1_pd:.2f}$)",
        fr"ttbar (dim=4, $\epsilon[t_0]={eff0_ttbar4_pd:.2f}$)",
        fr"HToAATo4B (dim=1, $\epsilon[t_0]={eff0_higgs1_pd:.2f}$)",
        fr"HToAATo4B (dim=4, $\epsilon[t_0]={eff0_higgs4_pd:.2f}$)",
    ]

    plt.legend(
        handles, labels,
        title="AD Trigger",
        ncol=2, loc="best", frameon=True,
        handlelength=2, columnspacing=1, labelspacing=0.8,
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "L_sas_rate_pidMC2.pdf"), bbox_inches="tight")
    plt.close()

def plot_ad_local_eff_fixed(data: dict):
    """AD local signal efficiency, constant menu only."""
    L_E3_1 = np.asarray(data["L_E3_1"])
    L_E3_4 = np.asarray(data["L_E3_4"])
    L_E5_1 = np.asarray(data["L_E5_1"])
    L_E5_4 = np.asarray(data["L_E5_4"])
    n = min(len(L_E3_1), len(L_E3_4), len(L_E5_1), len(L_E5_4))
    L_E3_1, L_E3_4, L_E5_1, L_E5_4 = (
        L_E3_1[:n], L_E3_4[:n], L_E5_1[:n], L_E5_4[:n]
    )
    t = _time_axis(n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, L_E3_1 / L_E3_1[0], label="ttbar, dim=1",
            color="goldenrod", linewidth=2.5, linestyle="dashed")
    ax.plot(t, L_E3_4 / L_E3_4[0], label="ttbar, dim=4",
            color="orangered", linewidth=2.5, linestyle="dashed")
    ax.plot(t, L_E5_1 / L_E5_1[0], label="HToAATo4B, dim=1",
            color="limegreen", linewidth=2.5, linestyle="dashed")
    ax.plot(t, L_E5_4 / L_E5_4[0], label="HToAATo4B, dim=4",
            color="seagreen", linewidth=2.5, linestyle="dashed")

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="AD Constant Menu", frameon=True, loc="best")

    fig.savefig(OUTDIR / "L_sas_rate_fixed.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_ad_local_eff_fixed_relative(data: dict):
    """AD local signal efficiency (relative), constant menu only for relative efficiency."""
    L_E3_1 = np.asarray(data["L_E3_1"])
    L_E3_4 = np.asarray(data["L_E3_4"])
    L_E5_1 = np.asarray(data["L_E5_1"])
    L_E5_4 = np.asarray(data["L_E5_4"])
    n = min(len(L_E3_1), len(L_E3_4), len(L_E5_1), len(L_E5_4))
    time = np.linspace(0, 1, n)
    eff0_ttbar1 = L_E3_1[0]
    eff0_ttbar4 = L_E3_4[0]
    eff0_AA1 = L_E5_1[0]
    eff0_AA4 = L_E5_4[0]
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.array(L_E3_1[:n])/eff0_ttbar1, label=fr'ttbar (dim=1, $\epsilon[t_0]$={eff0_ttbar1:.2f})', color='goldenrod', linewidth=2.5, linestyle='dashed')
    plt.plot(time, np.array(L_E3_4[:n])/eff0_ttbar4, label=fr'ttbar (dim=4, $\epsilon[t_0]$={eff0_ttbar4:.2f})', color='orangered', linewidth=2.5, linestyle='dashed')
    plt.plot(time, np.array(L_E5_1[:n])/eff0_AA1, label=fr'HToAATo4B (dim=1, $\epsilon[t_0]$={eff0_AA1:.2f})', color='limegreen', linewidth=2.5, linestyle='dashed')
    plt.plot(time, np.array(L_E5_4[:n])/eff0_AA4, label=fr'HToAATo4B (dim=4, $\epsilon[t_0]$={eff0_AA4:.2f})', color='seagreen', linewidth=2.5, linestyle='dashed')
    plt.xlabel('Time (Fraction of Run)',loc='center')
    plt.ylabel('Relative Efficiency',loc='center')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='AD Constant Menu', frameon=True, fontsize=19)
    plt.savefig(OUTDIR /'L_sas_rate_fixed2.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# all plotting (summary, evolution, figure save utils)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np, matplotlib.ticker as mticker
def evolution(real_Ht_vals, real_AS_vals, title, outdir="outputs"):
    plt.figure(figsize=(8,6))
    plt.scatter(real_Ht_vals, real_AS_vals, c=np.arange(len(real_Ht_vals)),
                marker='+', cmap="viridis", label="real-time Points", s=50)
    plt.xlabel("Ht Cut"); plt.ylabel("AD Cut")
    plt.xlim(75, 350); plt.ylim(75, 250)
    plt.title(title); plt.colorbar(label="Iteration Step")
    plt.legend(loc='best', frameon=True); plt.grid(True)
    plt.savefig(f"{outdir}/{title}.pdf"); plt.close()

def summary(abs_rb, rs, b_TC, b_EC, total_cost, perf, title, outdir="outputs"):
    t = np.arange(len(abs_rb))
    fig = plt.figure(figsize=(14,10))
    plt.subplot(2,2,1); sc=plt.scatter(abs_rb, total_cost, c=t, cmap='viridis')
    plt.xlabel('1/(1 - |rb - 0.25|)'); plt.ylabel('Total Computational Cost')
    plt.title('Background Robustness vs Total Cost'); plt.colorbar(sc, label='Time')

    plt.subplot(2,2,2); sc=plt.scatter(rs, total_cost, c=t, cmap='viridis')
    plt.xlabel('rs'); plt.ylabel('Total Computational Cost')
    plt.title('Signal Eff vs Total Cost'); plt.colorbar(sc, label='Time')

    plt.subplot(2,2,3); sc=plt.scatter(perf, total_cost, c=t, cmap='viridis')
    plt.xlabel('rs/(1 - |rb - 0.25|)'); plt.ylabel('Total Computational Cost')
    plt.title('Performance vs Total Cost'); plt.colorbar(sc, label='Time')

    plt.subplot(2,2,4); sc=plt.scatter(b_TC, b_EC, c=t, cmap='viridis')
    plt.xlabel('Trigger Computational Cost'); plt.ylabel('Event Computational Cost')
    plt.title('Event vs Trigger Cost'); plt.colorbar(sc, label='Time')

    plt.tight_layout(); fig.savefig(f"{outdir}/{title}.pdf"); plt.close(fig)

def save_subplot(fig, ax, filename, pad=0.3):
    fig.canvas.draw()
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    bbox = mtransforms.Bbox.from_extents(bbox.x0-pad, bbox.y0-pad, bbox.x1+pad, bbox.y1+pad)
    fig.savefig(filename, bbox_inches=bbox)

def rate_efficiency_panels(time, R, Rht, Ras, Id1_R, GE, Eht, Eas, Id1_GE, out_pdf, out_a, out_b, case_label="Case 3", control = "MC"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Background
    axes[0].plot(time, R,   label="Total Rate", color='navy', linewidth=2, marker='o')
    axes[0].plot(time, Rht, label="HT Rate",    color='cyan', linewidth=1, marker='o')
    axes[0].plot(time, Ras, label="AD Rate",    color='deepskyblue', linewidth=1, marker='o')
    axes[0].plot(time, Id1_R, linestyle="dashed", color='dodgerblue', linewidth=2, label="Total Ideal Rate")
    axes[0].axhline(y=112, color='grey', linestyle='--', linewidth=1.5)
    axes[0].axhline(y=88,  color='grey', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Time (Fraction of Run)', labelpad=10, loc='center'); axes[0].set_ylabel('Average Background Rate (kHz)', labelpad=10, loc='center')
    axes[0].set_xlim(0,1); axes[0].set_ylim(0,270); axes[0].legend(title=case_label); axes[0].grid(True)
    # axes[0].set_ylim(40, 270)

    # Signal
    axes[1].plot(time, GE,  label="Total Cumulative Signal Efficiency", color='mediumvioletred', linewidth=2, marker='o')
    axes[1].plot(time, Eht, label="HT Signal Efficiency", color='mediumpurple', linewidth=1, marker='o')
    axes[1].plot(time, Eas, label="AD Signal Efficiency", color='orchid', linewidth=1, marker='o')
    axes[1].plot(time, Id1_GE, linestyle="dashed", color='black', linewidth=2, label="Total Ideal Cumulative Signal Efficiency")
    axes[1].set_xlabel('Time (Fraction of Run)', labelpad=10, loc='center'); axes[1].set_ylabel('Average Efficiency (%)', labelpad=10, loc='center')
    axes[1].set_xlim(0,1)
    if control == "MC":
        axes[1].set_ylim(60,90); axes[1].legend(title=case_label, loc='upper left'); axes[1].grid(True)
    else:
        axes[1].set_xlim(-0.02, 1.02)
        print('Real Data setting')
        axes[1].set_ylim(40, 70)
        from matplotlib.ticker import MaxNLocator
        axes[1].xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
        axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
        axes[1].set_yticks(np.arange(35, 70, 5))
        axes[1].tick_params(axis='both')
        yticks1 = axes[1].get_yticks()
        axes[1].set_yticks(yticks1[1:])
        axes[1].legend(title=case_label, fontsize=15,
               loc='best', frameon=True, handlelength=2.5,
               handletextpad=0.5, borderpad=0.8)
        axes[1].grid(True)




    plt.tight_layout(); fig.savefig(out_pdf)
    save_subplot(fig, axes[0], out_a, pad=0.3); save_subplot(fig, axes[1], out_b, pad=0.3)
    plt.close(fig)
    return fig, axes




# --- Multi_path.py ------------------------------------------------
# def multi_path_panels(time, Id1_R, Id1_r_bht, Id1_r_bas,
#                       Id1_r1_s, Id1_r2_s, Id1_r1_sht, Id1_r1_sas,
#                       Id1_r2_sht, Id1_r2_sas, Id1_E, Id1_GE,
#                       out_pdf="paper/multi_path_plots(case3).pdf"):
#     fig, axes = plt.subplots(2, 2, figsize=(18, 12))

#     # (0,0) background
#     axes[0,0].plot(time, Id1_R,      label="Total Ideal Rate", color='tab:blue')
#     axes[0,0].plot(time, Id1_r_bht,  label="HT Ideal Rate",    linestyle='--', color='cyan')
#     axes[0,0].plot(time, Id1_r_bas,  label="AD Ideal Rate",    linestyle='-.', color='mediumblue')
#     axes[0,0].set_xlabel('Time (Fraction of Run)'); axes[0,0].set_ylabel('Background Rate (kHz)')
#     axes[0,0].set_xlim(0,1); axes[0,0].set_ylim(0,200); axes[0,0].legend(title='Case 3'); axes[0,0].grid(True)

#     # (1,1) ttbar efficiency
#     axes[1,1].plot(time, Id1_r1_s,    label="Total",     color='tab:orange')
#     axes[1,1].plot(time, Id1_r1_sht,  label="from HT",   linestyle='--', color='gold')
#     axes[1,1].plot(time, Id1_r1_sas,  label="from AD",   linestyle='-.', color='goldenrod')
#     axes[1,1].set_xlabel('Time (Fraction of Run)'); axes[1,1].set_ylabel('TTbar Efficiency (%)')
#     axes[1,1].set_xlim(0,1); axes[1,1].legend(title='Case 3'); axes[1,1].grid(True)

#     # (1,0) HToAATo4B efficiency
#     axes[1,0].plot(time, Id1_r2_s,    label="Total",     color='tab:green')
#     axes[1,0].plot(time, Id1_r2_sht,  label="from HT",   linestyle='--', color='lime')
#     axes[1,0].plot(time, Id1_r2_sas,  label="from AD",   linestyle='-.', color='darkgreen')
#     axes[1,0].set_xlabel('Time (Fraction of Run)'); axes[1,0].set_ylabel('HToAATo4B Efficiency (%)')
#     axes[1,0].set_xlim(0,1); axes[1,0].legend(title='Case 3'); axes[1,0].grid(True)

#     # (0,1) total signal eff
#     axes[0,1].plot(time, Id1_E,  label="Total",            linestyle='--', color='purple')
#     axes[0,1].plot(time, Id1_GE, label="Cumulative Total",               color='purple')
#     axes[0,1].set_xlabel('Time (Fraction of Run)'); axes[0,1].set_ylabel('Combined Signal Efficiency (%)')
#     axes[0,1].set_xlim(0,1); axes[0,1].legend(title='Case 3'); axes[0,1].grid(True)

#     plt.tight_layout()
#     fig.savefig(out_pdf)
#     return fig, axes
# ---------------------------------------------------------------------------





# SingTig.py -> PID plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def pid_background_plot(time, const_series, pd_series, out_pdf, ylabel="Acceptance(%)",
                        title="", tol_lo=0.22, tol_hi=0.28, ylim=(0,0.7),
                        const_label="Constant Menu", pd_label="PD Controller",
                        color_const="tab:blue", color_pd="mediumblue"):
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.linspace(0, 1, 6)))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    plt.plot(time, const_series, label=const_label, color=color_const, linewidth=3, linestyle="dashed")
    plt.plot(time, pd_series,    label=pd_label,    color=color_pd,   linewidth=2.5, linestyle="solid")
    plt.axhline(y=tol_hi, color='gray', linestyle='--', linewidth=1.5, label=f'Upper Tolerance ({tol_hi})')
    plt.axhline(y=tol_lo, color='gray', linestyle='--', linewidth=1.5, label=f'Lower Tolerance ({tol_lo})')
    plt.ylim(*ylim)
    plt.title(title, fontsize=18)
    # plt.xlabel('Time (data batch)', fontsize=18)
    plt.xlabel('Time (Fraction of Run)', fontsize=18)
    plt.xlim(0, 1)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(fontsize=14, loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()

def pid_normed_pairs_plot(time, const_a, const_b, pd_a, pd_b, out_pdf, title, ylabel,
                          labels=("Const A","Const B","PD A","PD B"),
                          styles=("dashed","dotted","solid","solid"),
                          colors=("tab:orange","tab:green","tab:orange","tab:green"),
                          ylim=(0.75,1.6)):
    # normalize by own means
    const_a_n = const_a/np.mean(const_a) if len(const_a) else const_a
    const_b_n = const_b/np.mean(const_b) if len(const_b) else const_b
    pd_a_n    = pd_a/np.mean(pd_a)       if len(pd_a)    else pd_a
    pd_b_n    = pd_b/np.mean(pd_b)       if len(pd_b)    else pd_b
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.linspace(0, 1, 6)))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    for series, lab, style, col in zip([const_a_n,const_b_n,pd_a_n,pd_b_n], labels, styles, colors):
        plt.plot(time, series, label=f"{lab}", color=col, linewidth=2.5, linestyle=style)
    plt.ylim(*ylim)
    plt.title(title, fontsize=18)
    # plt.xlabel('Time (data batch)', fontsize=18)
    plt.xlabel('Time (Fraction of Run)', fontsize=18)
    plt.xlim(0, 1) 
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(fontsize=14, loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()

def pid_local_normed_pairs_plot(time, const_a, const_b, pd_a, pd_b, out_pdf, title, ylabel,
                                ylim=(0.5,2.2)):
    pid_normed_pairs_plot(
        time, const_a, const_b, pd_a, pd_b, out_pdf,
        title=title, ylabel=ylabel, ylim=ylim,
        labels=("Constant Menu A","Constant Menu B","PD Controller A","PD Controller B"),
        styles=("dashed","dotted","solid","solid")
    )



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
    data = load_results()
    plot_ht_background(data)
    plot_ht_background_fixed(data)
    plot_ad_background(data)
    plot_ad_background_fixed(data)

    plot_ht_cumulative_eff(data)
    plot_ht_local_eff(data)
    plot_ht_cumulative_eff_fixed(data)
    plot_ht_local_eff_fixed(data)


    plot_ad_cumulative_eff(data)
    plot_ad_cumulative_eff_fixed(data)
    plot_ad_cumulative_eff_fixed_relative(data)
    plot_ad_local_eff(data)
    plot_ad_local_eff_fixed(data)
    plot_ad_local_eff_fixed_relative(data)


if __name__ == "__main__":
    main()
