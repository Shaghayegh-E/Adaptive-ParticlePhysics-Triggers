# Trigger_io.py
import h5py
import numpy as np
from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines





def read_mc_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        Bas_tot = h5_file['bkg_score02'][:]
        Bht_tot   = h5_file['bkg_ht'][:]
        B_npvs    = h5_file['bkg_Npv'][:]

        Sas_tot1 = h5_file['tt_score02'][:]
        Sht_tot1   = h5_file['tt_ht'][:]
        S_npvs1    = h5_file['tt_Npv'][:]

        Sas_tot2 = h5_file['aa_score02'][:]
        Sht_tot2   = h5_file['aa_ht'][:]
        S_npvs2    = h5_file['aa_Npv'][:]

    return (
        Sas_tot1, Sht_tot1, S_npvs1,
        Sas_tot2, Sht_tot2, S_npvs2,
        Bas_tot,  Bht_tot,  B_npvs
    )

def load_trigger_food(path: str): #used for Multi Trigger Logics
    with h5py.File(path, 'r') as h5_file:
        Bas_tot = h5_file['bkg_score02'][:]
        Bht_tot   = h5_file['bkg_ht'][:]
        B_npvs    = h5_file['bkg_Npv'][:]
        B_njets    = h5_file['bkg_njet'][:]

        Sas_tot1 = h5_file['tt_score02'][:]
        Sht_tot1   = h5_file['tt_ht'][:]
        S_npvs1    = h5_file['tt_Npv'][:]
        S_njets1    = h5_file['tt_njet'][:]

        Sas_tot2 = h5_file['aa_score02'][:]
        Sht_tot2   = h5_file['aa_ht'][:]
        S_npvs2    = h5_file['aa_Npv'][:]
        S_njets2    = h5_file['aa_njet'][:]

    return (
        Sas_tot1, Sht_tot1, S_npvs1, S_njets1,
        Sas_tot2, Sht_tot2, S_npvs2, S_njets2,
        Bas_tot,  Bht_tot,  B_npvs, B_njets
    )

# all plotting (summary, evolution, figure save utils)

def evolution(Ht_vals, AD_vals, title, outdir="outputs"):
    plt.figure(figsize=(8,6))
    plt.scatter(Ht_vals, AD_vals, c=np.arange(len(Ht_vals)),
                marker='+', cmap="viridis", s=50)
    plt.xlabel("Ht Cut"); plt.ylabel("AD Cut")
    plt.xlim(np.min(Ht_vals) * 0.9, np.max(Ht_vals) * 1.1)
    plt.ylim(np.min(AD_vals) * 0.9, np.max(AD_vals)* 1.1)

    plt.title(title); plt.colorbar(label="Iteration Step")
    #plt.legend(loc='best', frameon=True); plt.grid(True)
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


def rate_efficiency_panels(time, R, Rht, Ras, Id1_R, GE, Eht, Eas, Id1_GE, out_pdf, out_a, out_b, case_label="Case 1", bkg = "MC"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Background
    axes[0].plot(time, R,   label="Total Rate", color='navy', linewidth=2, marker='o')
    axes[0].plot(time, Rht, label="HT Rate",    color='cyan', linewidth=1, marker='o')
    axes[0].plot(time, Ras, label="AD Rate",    color='deepskyblue', linewidth=1, marker='o')
    axes[0].plot(time, Id1_R, linestyle="dashed", color='dodgerblue', linewidth=2, label="Total Ideal Rate")
    axes[0].axhline(y=110, color='grey', linestyle='--', linewidth=1.5)
    axes[0].axhline(y=90,  color='grey', linestyle='--', linewidth=1.5)
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
    if bkg == "MC":
        axes[1].set_ylim(60,90); axes[1].legend(title=case_label, loc='upper left'); axes[1].grid(True)
    else:
        print('Real Data setting')
        axes[1].set_ylim(40, 70)

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

    print("Saved the figures!")


def multi_path_panels(time, Id1_R, Id1_r_bht, Id1_r_bas,
    Id1_r1_s, Id1_r2_s, Id1_r1_sht, Id1_r1_sas,
    Id1_r2_sht, Id1_r2_sas, Id1_E, Id1_GE,
    out_pdf="outputs/demo_IdealMultiTrigger/multi_path_plots(case1).pdf",case='Case1'):
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # (0,0) background
    axes[0,0].plot(time, Id1_R,      label="Total Ideal Rate", color='tab:blue')
    axes[0,0].plot(time, Id1_r_bht,  label="HT Ideal Rate",    linestyle='--', color='cyan')
    axes[0,0].plot(time, Id1_r_bas,  label="AD Ideal Rate",    linestyle='-.', color='mediumblue')
    axes[0,0].set_xlabel('Time (Fraction of Run)'); axes[0,0].set_ylabel('Background Rate (kHz)')
    axes[0,0].set_xlim(0,1); axes[0,0].set_ylim(0,200); axes[0,0].legend(title=f"{case}"); axes[0,0].grid(True)

    # (1,1) ttbar efficiency
    axes[1,1].plot(time, Id1_r1_s,    label="Total",     color='tab:orange')
    axes[1,1].plot(time, Id1_r1_sht,  label="from HT",   linestyle='--', color='gold')
    axes[1,1].plot(time, Id1_r1_sas,  label="from AD",   linestyle='-.', color='goldenrod')
    axes[1,1].set_xlabel('Time (Fraction of Run)'); axes[1,1].set_ylabel('TTbar Efficiency (%)')
    axes[1,1].set_xlim(0,1); axes[1,1].legend(title=f"{case}"); axes[1,1].grid(True)

    # (1,0) HToAATo4B efficiency
    axes[1,0].plot(time, Id1_r2_s,    label="Total",     color='tab:green')
    axes[1,0].plot(time, Id1_r2_sht,  label="from HT",   linestyle='--', color='lime')
    axes[1,0].plot(time, Id1_r2_sas,  label="from AD",   linestyle='-.', color='darkgreen')
    axes[1,0].set_xlabel('Time (Fraction of Run)'); axes[1,0].set_ylabel('HToAATo4B Efficiency (%)')
    axes[1,0].set_xlim(0,1); axes[1,0].legend(title=f"{case}"); axes[1,0].grid(True)

    # (0,1) total signal eff
    axes[0,1].plot(time, Id1_E,  label="Total",            linestyle='--', color='purple')
    axes[0,1].plot(time, Id1_GE, label="Cumulative Total",               color='purple')
    axes[0,1].set_xlabel('Time (Fraction of Run)'); axes[0,1].set_ylabel('Combined Signal Efficiency (%)')
    axes[0,1].set_xlim(0,1); axes[0,1].legend(title=f"{case}"); axes[0,1].grid(True)

    plt.tight_layout()
    fig.savefig(out_pdf)
    return fig, axes
