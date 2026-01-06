# summary.py
# putting the summary binning + plotting
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Callable

from .trigger_io import load_trigger_food

from .agents import V3, local_V3, V1, local_V1, V2, local_V2, Trigger
import mplhep as hep
hep.style.use("CMS")

# trigger/run_summary_plots.py
import argparse, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable





def average_pair_over_bins(x_list: list[float], y_list: list[float], n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(x_list))
    bins = np.array_split(idx, n_bins)
    x_avg, y_avg = [], []
    #x_avg.append(x_list[0])
    #y_avg.append(y_list[0])
    for b in bins:
        x_avg.append(np.mean([x_list[i] for i in b]))
        y_avg.append(np.mean([y_list[i] for i in b]))
    return np.asarray(x_avg), np.asarray(y_avg)


def build_initial_cuts_from_first_chunk(
    agent_funcs: dict[str, Callable],
    bht: np.ndarray, bas: np.ndarray, bnjets: np.ndarray,
    sht1: np.ndarray, sas1: np.ndarray,
    sht2: np.ndarray, sas2: np.ndarray,
    ref: np.ndarray
) -> dict[str, tuple[float, float]]:
    """
    For each agent, scan its grid once and take the (HT, AS) at min cost.
    """
    initial_cuts = {}
    
    for name, agent in agent_funcs.items():
        agent, _ = pick_agent(name)
        cost, *_ , HT, AS = agent(bht, sht1, sht2, bas, sas1, sas2, bnjets, ref)
        i0, j0 = np.unravel_index(np.argmin(cost), cost.shape)
        initial_cuts[name] = (float(HT[i0, j0]), float(AS[i0, j0]))
    return initial_cuts


def run_summary_stream(
    bkgType: str,
    path: str,
    ref: np.ndarray,
    start_offset: int,
    chunk: int,
    fixed_calib_len: int,
) -> dict:
    """
    Returns a dict keyed by case name, each holding lists of metrics.
    """
    Sas1, Sht1, Snpv1, Snjets, Sas2, Sht2, Snpv2, Snjets2, Bas, Bht, Bnpv, Bnjets = load_trigger_food(path)


    Bas = Bas[start_offset:]; Bht = Bht[start_offset:]; Bnpv = Bnpv[start_offset:]; Bnjets = Bnjets[start_offset:]
    N = Bnpv.size
    if bkgType=="RealData":
        Sas1 = Sas1[start_offset:]; Sht1 = Sht1[start_offset:]; Snpv1 = Snpv1[start_offset:]; Snjets = Snjets[start_offset:]
        Sas2 = Sas2[start_offset:]; Sht2 = Sht2[start_offset:]; Snpv2 = Snpv2[start_offset:]; Snjets2 = Snjets2[start_offset:]


    # cases storage
    cases_lists = {
        'Fixed Menu':        {'cost': [], 'absrb': [], 'rs': [], 'w0absrb': [], 'w1rs': [], 'Tcost': [], 'Ecost': []},
        'Standard':          {'cost': [], 'absrb': [], 'rs': [], 'w0absrb': [], 'w1rs': [], 'Tcost': [], 'Ecost': []},
        'Anomaly Focused':   {'cost': [], 'absrb': [], 'rs': [], 'w0absrb': [], 'w1rs': [], 'Tcost': [], 'Ecost': []},
        'Low-Comp Focused':  {'cost': [], 'absrb': [], 'rs': [], 'w0absrb': [], 'w1rs': [], 'Tcost': [], 'Ecost': []},
    }

    # initial grid (first chunk) to get cuts per agent
    init = slice(0, min(chunk, N))
    bht_i, bas_i, bnjets_i = Bht[init], Bas[init], Bnjets[init]
    npv_i = Bnpv[init]
    m1_i = (Snpv1 >= npv_i.min()) & (Snpv1 <= npv_i.max())
    m2_i = (Snpv2 >= npv_i.min()) & (Snpv2 <= npv_i.max())
    
    if bkgType=="MC":
        sht1_i, sas1_i = Sht1[m1_i], Sas1[m1_i]
        sht2_i, sas2_i = Sht2[m2_i], Sas2[m2_i]
    else:
        sht1_i   = Sht1[init]
        sas1_i = Sas1[init]
        sht2_i   = Sht2[init]
        sas2_i = Sas2[init]

    agent_for_init = {
        'Standard': V1,
        'Anomaly Focused': V2,
        'Low-Comp Focused': V3,
    }
    init_cuts = build_initial_cuts_from_first_chunk(agent_for_init, bht_i, bas_i, bnjets_i, sht1_i, sas1_i, sht2_i, sas2_i,ref)

    # Fixed menu: use percentiles over the first fixed_calib_len
    calib = slice(0, min(fixed_calib_len, N))
    Ht_fixed = float(np.percentile(Bht[calib], 99.8))
    AS_fixed = float(np.percentile(Bas[calib], 99.9))

    # live cuts per case
    cuts = {
        'Standard':         {'Ht': init_cuts['Standard'][0],         'AS': init_cuts['Standard'][1]},
        'Anomaly Focused':  {'Ht': init_cuts['Anomaly Focused'][0],  'AS': init_cuts['Anomaly Focused'][1]},
        'Low-Comp Focused': {'Ht': init_cuts['Low-Comp Focused'][0], 'AS': init_cuts['Low-Comp Focused'][1]},
    }

    # local agents used for iterative updates
    local_agents = {
        'Standard': local_V1,
        'Anomaly Focused': local_V2,
        'Low-Comp Focused': local_V3,
    }
    
    five_fold = {
        'Standard':[],
        'Anomaly Focused':[],
        'Low-Comp Focused': [],
        
    }

    # stream over chunks
    for I in range(0, N, chunk):
        idx = slice(I, min(I + chunk, N))
        bht_c, bas_c, bnjets_c, bnpv_c = Bht[idx], Bas[idx], Bnjets[idx], Bnpv[idx]
        npv_min, npv_max = float(bnpv_c.min()), float(bnpv_c.max())
        m1 = (Snpv1 >= npv_min) & (Snpv1 <= npv_max)
        m2 = (Snpv2 >= npv_min) & (Snpv2 <= npv_max)
    
        
        if bkgType=="MC":
            sht1, sas1 = Sht1[m1], Sas1[m1]
            sht2, sas2 = Sht2[m2], Sas2[m2]
        else:
            sht1  = Sht1[idx]
            sas1 = Sas1[idx]
            sht2   = Sht2[idx]
            sas2 = Sas2[idx]


        # Fixed Menu
        r_b, r_s, _, _, _, _, Ecost, Tcost = Trigger(
            bht_c,sht1,sht2, bas_c,sas1, sas2, bnjets_c, Ht_fixed, AS_fixed
        )
        cases_lists['Fixed Menu']['absrb'].append(abs(400.0 * r_b - 100.0))
        cases_lists['Fixed Menu']['w0absrb'].append(100.0 * abs(r_b - 0.25))
        cases_lists['Fixed Menu']['rs'].append(r_s)
        cases_lists['Fixed Menu']['w1rs'].append(0.2 * (100.0 - r_s))
        cases_lists['Fixed Menu']['Tcost'].append(Tcost)
        cases_lists['Fixed Menu']['Ecost'].append(Ecost)
        cases_lists['Fixed Menu']['cost'].append(Ecost + Tcost)

        # Adaptive cases (V1/V2/V3 local agents)
        for case, agent in local_agents.items():
            _, agent = pick_agent(case)
            ht_cut = cuts[case]['Ht']; as_cut = cuts[case]['AS']

            r_b, r_s, _, _, _, _, Ecost, Tcost = Trigger(
                bht_c,sht1,sht2, bas_c,sas1, sas2, bnjets_c, ht_cut, as_cut
            )
            cases_lists[case]['absrb'].append(abs(400.0 * r_b - 100.0))
            cases_lists[case]['w0absrb'].append(100.0 * abs(r_b - 0.25))
            cases_lists[case]['rs'].append(r_s)
            cases_lists[case]['w1rs'].append(0.2 * (100.0 - r_s))
            cases_lists[case]['Tcost'].append(Tcost)
            cases_lists[case]['Ecost'].append(Ecost)
            cases_lists[case]['cost'].append(Ecost + Tcost)

            # local scan update
            cost_grid, *_ , HT, AS = agent(bht_c, sht1, sht2, bas_c, sas1, sas2, bnjets_c, ref, ht_cut, as_cut)
            
            ii, jj = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
            five_fold[case].append([HT[ii,jj],AS[ii,jj]])
            
            if len(five_fold[case])>=5: 
                #Ht_cut, AS_cut = np.mean(np.array(fivefold_window)[-5:], axis=0)
                cuts[case]['Ht'], cuts[case]['AS'] = np.mean(np.array(five_fold[case])[-5:], axis=0)
            else : 
                cuts[case]['Ht'], cuts[case]['AS'] = HT[ii, jj], AS[ii, jj]

            
            #cuts[case]['Ht'], cuts[case]['AS'] = float(HT[ii, jj]), float(AS[ii, jj])

        print(f"Processed chunk starting at {I}")

    return cases_lists



def plot_case_comparison(cases_data, n_bins=10, save_path=None):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.ravel()

    titles  = [r'$(1-\epsilon)/\sigma_s$ vs $|r_b-r_t|/\sigma_b$',
               r'Total Computational Cost vs $|r_b-r_t|$',
               r'Total Computational Cost vs $\epsilon(\%)$',
               'Average Trigger Cost vs Average Event Cost']
    x_keys  = ['w0absrb', 'absrb', 'rs', 'Tcost']
    y_keys  = ['w1rs',    'cost',  'cost', 'Ecost']
    xlabels = [r'$|r_b-r_t|/\sigma_b$',
               r'$|r_b-r_t|$, $r_b =$ Background Rate(kHz)',
               r'$\epsilon$: Total Signal Efficiency $(\%)$',
               'Average Trigger Cost']
    ylabels = [r'$(1-\epsilon)/\sigma_s$',
               'Total Computational Cost',
               'Total Computational Cost',
               'Average Event Cost']

    markers = ['o', 's', 'P', 'D']
    base_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    legend_handles, legend_labels = [], []

    for c_idx, (case_name, data) in enumerate(cases_data.items()):
        base_color = mpl.colors.to_rgb(base_colors[c_idx % len(base_colors)])
        #print(case_name, base_colors[c_idx % len(base_colors)])
        for p in range(4):
            x_vals, y_vals = average_pair_over_bins(data[x_keys[p]], data[y_keys[p]], n_bins=n_bins)
            if len(x_vals) < 2:
                continue
            points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            t_vals = np.linspace(0, 1, len(segments))
            colors = [(1 - t) * np.array(base_color) + t * np.ones(3) for t in t_vals]

            lc = LineCollection(segments, colors=colors, linewidth=min(2*(4-c_idx),4), alpha=1, label=case_name)
            axs[p].add_collection(lc)
            axs[p].plot(x_vals[0], y_vals[0], marker='*', color=base_color, markersize=12)
            axs[p].plot(x_vals[1:], y_vals[1:], marker=markers[c_idx % len(markers)],
                        linestyle='None', color=base_color, markersize=5)

            # legend “template” handle
            line, = axs[0].plot([], [], color=base_color, marker=markers[c_idx % len(markers)],
                                linestyle='-', linewidth=2)
            if p == 0:
                legend_handles.append(line); legend_labels.append(case_name)

    for p in range(4):
        axs[p].set_xlabel(xlabels[p], fontsize=15)
        axs[p].set_ylabel(ylabels[p], fontsize=15)
        axs[p].tick_params(axis='both', labelsize=15)
        axs[p].grid(True)
        yt = axs[p].get_yticks()
        axs[p].set_yticks(yt[1:])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.8, wspace=0.3, hspace=0.3)
    fig.suptitle("Summary Plots for Performance Comparison", fontsize=22, x=0.4, y=0.95)

    # global colorbar (time direction)
    sm = ScalarMappable(cmap='Greys', norm=plt.Normalize(0, 1)); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Relative Time (Dark → Light)', fontsize=18); cbar.set_ticks([])

    from matplotlib.lines import Line2D
    star_legend = Line2D([], [], marker='*', color='black', linestyle='None', markersize=10)
    legend_handles.append(star_legend); legend_labels.append("Start of the Run")
    fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(0.99, 0.92),
               fontsize=16, ncol=1, frameon=False)

    if save_path:
        plt.savefig(save_path)
    else:
        print("No save_path provided")


def pick_agent(name: str):
    #name = name.lower()

    # Wrap agents so they all share the same call signature:
    #   global_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,snj1,snj2)
    #   local_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,Ht_cut,AS_cut)
    if name == "Standard":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj,ref):
            # V1 does not use jet multiplicities; ignore bnj/snj*
            return V1(bht, sht1, sht2, bas, sas1, sas2)
        def local_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref, Ht_cut, AS_cut):
            return local_V1(bht, sht1, sht2, bas, sas1, sas2, Ht_cut, AS_cut)
        return global_agent, local_agent

    if name == "Anomaly Focused":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj,ref):
            return V2(bht, sht1, sht2, bas, sas1, sas2)
        def local_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref, Ht_cut, AS_cut):
            return local_V2(bht, sht1, sht2, bas, sas1, sas2, Ht_cut, AS_cut)
        return global_agent, local_agent

    if name == "Low-Comp Focused":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref):
            return V3(bht, sht1, sht2, bas, sas1, sas2, bnj, ref)
        def local_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref,Ht_cut, AS_cut):
            return local_V3(bht, sht1, sht2, bas, sas1, sas2, bnj, ref, Ht_cut, AS_cut)
        return global_agent, local_agent

    raise ValueError(f"Unknown agent name {name}")


def main():
    ap = argparse.ArgumentParser(description="Summary plots across strategies (Fixed / V1 / V2 / V3)")
    ap.add_argument("--bkgType", default="MC")
    ap.add_argument("--path", default="Data/Trigger_food_MC.h5")
    ap.add_argument("--out", default="outputs/SummaryPanels_MC.pdf")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--costRef", type=float, nargs="+", default=[5.7, 2.5])
    ap.add_argument("--forceCostRef", action="store_true")

    args = ap.parse_args()
    
    if args.bkgType=="RealData":
        chunk_size = 20000
        CostRef = [4.3, 3.5]
    else :
        chunk_size = 50000
        CostRef = [5.7, 2.5]
        
    if args.forceCostRef : CostRef = args.costRef
    

    cases_lists = run_summary_stream(args.bkgType, args.path, CostRef, start_offset=10*chunk_size, chunk=chunk_size, fixed_calib_len=2*chunk_size)

    # shape into plotting dict (same keys as your original)
    cases_data = {
        name: {
            'w0absrb': d['w0absrb'],
            'w1rs':    d['w1rs'],
            'cost':    d['cost'],
            'absrb':   d['absrb'],
            'rs':      d['rs'],
            'Tcost':   d['Tcost'],
            'Ecost':   d['Ecost'],
        } for name, d in cases_lists.items()
    }

    plot_case_comparison(cases_data, n_bins=args.bins, save_path=args.out)

if __name__ == "__main__":
    main()
