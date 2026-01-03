# trigger/run_multi_path.py
from __future__ import annotations
import argparse, numpy as np, matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    import atlas_mpl_style as aplt
    aplt.use_atlas_style()
except Exception:
    pass
import os
from .Trigger_io import load_trigger_food_mc
from .agents_multi_path import V3, V1, V2
# from MC.trigger.metrics import comp_cost_test
from .metrics import update_accumulated, comp_cost_test
from .mc_singletrigger_plots import multi_path_panels, save_subplot, plot_evolution_ideal

def pick_agent(name: str):
    name = name.lower()

    # Wrap agents so they all share the same call signature:
    #   global_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,snj1,snj2)
    #   local_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,Ht_cut,AS_cut)
    if name == "v1":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2):
            # V1 does not use jet multiplicities; ignore bnj/snj*
            return V1(bht, sht1, sht2, bas, sas1, sas2)
        local_agent = None
        return global_agent, local_agent, "case1"

    if name == "v2":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2):
            return V2(bht, sht1, sht2, bas, sas1, sas2)
        local_agent = None
        return global_agent, local_agent, "case2"

    if name == "v3":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2):
            return V3(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2)
        local_agent = None
        return global_agent, local_agent, "case3"

    raise ValueError(f"Unknown agent name {name}")

def main():
    p = argparse.ArgumentParser(description="Multi-path ideal grid scan across chunks")
    p.add_argument("--path", default="Data/Trigger_food_MC.h5")     # your Zenodo MC file
    p.add_argument("--agent", default="v3", choices=["v0","v1","v2","v3","v5","v6"])
    p.add_argument("--chunk", type=int, default=50000)
    p.add_argument("--warmup_chunks", type=int, default=10)       # skip first N chunks
    p.add_argument("--outdir", default="outputs/demo_multipath")
    args = p.parse_args()


    os.makedirs(args.outdir, exist_ok=True)
    global_agent, local_agent, case_label = pick_agent(args.agent)

    D = load_trigger_food_mc(args.path)
    Bas = D["mc_bkg_score01"]; Bht = D["mc_bkg_ht"];  Bnpv = D["mc_bkg_Npv"]; Bnj = D["mc_bkg_njets"]
    Sas1, Sht1, Snpv1, Snj1 = D["mc_tt_score01"], D["mc_tt_ht"], D["tt_Npv"], D["mc_tt_njets"]
    Sas2, Sht2, Snpv2, Snj2 = D["mc_aa_score01"], D["mc_aa_ht"], D["aa_Npv"], D["mc_aa_njets"]

    Nb = len(Bnpv)

    # storage
    contour_H, contour_A = [], []
    Id1_R, Id1_E, Id1_GE = [], [], []
    Id1_r1_s, Id1_r2_s = [], []
    Id1_r_bht, Id1_r_bas = [], []
    Id1_r1_sht, Id1_r2_sht, Id1_r1_sas, Id1_r2_sas = [], [], [], []
    total_samples_r_s = []
    # --- storage for computational cost test (from Multi_path_comp_test.py) ---
    test_Ecost_both, test_Tcost_both = [], []
    test_Ecost_ht,   test_Tcost_ht   = [], []
    test_Ecost_as,   test_Tcost_as   = [], []

    for I in range(0, Nb, args.chunk):
        if I < 10*args.chunk:
            continue
        
        if I%args.chunk==0 : 
            idx = slice(I, min(I+args.chunk, Nb))
            bht, bas, bnpv, bnj = Bht[idx], Bas[idx], Bnpv[idx], Bnj[idx]

            npv_min, npv_max = float(bnpv.min()), float(bnpv.max())
            m1 = (Snpv1 >= npv_min) & (Snpv1 <= npv_max)
            m2 = (Snpv2 >= npv_min) & (Snpv2 <= npv_max)
            sht1, sas1, snj1 = Sht1[m1], Sas1[m1], Snj1[m1]
            sht2, sas2, snj2 = Sht2[m2], Sas2[m2], Snj2[m2]

        
            cost, r_b, r1_s, r2_s, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS = global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2)
       

            ii, jj = np.unravel_index(np.argmin(cost), cost.shape)
            contour_H.append(float(HT[ii,jj])); contour_A.append(float(AS[ii,jj]))

            Id1_R.append(r_b[ii, jj])
            Id1_r_bht.append(r_bht[ii, jj])
            Id1_r_bas.append(r_bas[ii, jj])

            # Store signal rates
            Id1_r1_s.append(r1_s[ii, jj])
            Id1_r2_s.append(r2_s[ii, jj])
            r_s = (r1_s[ii, jj]*sht1.shape[0] + r2_s[ii, jj]*sht2.shape[0])/(sht1.shape[0]+sht2.shape[0])
            # rates per-cut
            Id1_E.append(r_s)

            # Store signal rates per cut
            Id1_r1_sht.append(r1_sht[ii, jj])
            Id1_r2_sht.append(r2_sht[ii, jj])
            Id1_r1_sas.append(r1_sas[ii, jj])
            Id1_r2_sas.append(r2_sas[ii, jj])

            # # partial rates for panels (HT-only / AS-only)
            # Id1_r_bht.append(100.0 * (bht[:,None,None] >= HT[ii,jj]).sum() / bht.size)
            # Id1_r_bas.append(100.0 * (bas[:,None,None] >= AS[ii,jj]).sum() / bht.size)

            # Id1_r1_s.append(r1_s[ii, jj])
            # Id1_r2_s.append(r2_s[ii, jj])


            update_accumulated(Id1_GE, r_s, (sht1.shape[0]+sht2.shape[0]), total_samples_r_s)

            # --- computational cost diagnostics (from Multi_path_comp_test.py) ---

            # both paths on
            if args.agent == "v1":
                cost_both, E_both, T_both = comp_cost_test(
                    bht, bas, bnj, sht1, sas1, snj1, sht2, sas2, snj2,
                    use_path1=True, use_path2=True
                )
                ib, jb = np.unravel_index(np.argmin(cost_both), cost_both.shape)
                test_Ecost_both.append(float(E_both[ib, jb]))
                test_Tcost_both.append(float(T_both[ib, jb]))

                # HT path only
                cost_ht, E_ht, T_ht = comp_cost_test(
                    bht, bas, bnj, sht1, sas1, snj1, sht2, sas2, snj2,
                    use_path1=True, use_path2=False
                )
                ih, jh = np.unravel_index(np.argmin(cost_ht), cost_ht.shape)
                test_Ecost_ht.append(float(E_ht[ih, jh]))
                test_Tcost_ht.append(float(T_ht[ih, jh]))

                # AS path only
                cost_as, E_as, T_as = comp_cost_test(
                    bht, bas, bnj, sht1, sas1, snj1, sht2, sas2, snj2,
                    use_path1=False, use_path2=True
                )
                ia, ja = np.unravel_index(np.argmin(cost_as), cost_as.shape)
                test_Ecost_as.append(float(E_as[ia, ja]))
                test_Tcost_as.append(float(T_as[ia, ja]))


            print("time index:", I)
    #finished all chunks, make plots
    Id1_R = np.array(Id1_R)*400
    Id1_r_bht = np.array(Id1_r_bht)*400
    Id1_r_bas = np.array(Id1_r_bas)*400

    #Ideal evolution plot -> Motivation for Local Multi
    # evo_name_ideal = f"ideal_evoultion_{case_label}.pdf"  
    # plot_evolution_ideal(
    #     ideal_H, ideal_A,
    #     real_H, real_A,
    #     title=f"ideal_evoultion_{case_label}",
    #     out_pdf=f"{args.outdir}/{evo_name_ideal}",
    # )


    base = f"multi_path_plots({case_label})"
    base_null = f"multi_path_plots({case_label}"
    out_pdf = f"{args.outdir}/{base}.pdf"
    out_a   = f"{args.outdir}/{base_null}-a).pdf"
    out_b   = f"{args.outdir}/{base_null}-b).pdf"
    out_c   = f"{args.outdir}/{base_null}-c).pdf"
    out_d   = f"{args.outdir}/{base_null}-d).pdf"

    pretty_case = case_label if case_label.startswith("Case") else case_label.capitalize().replace("case", "Case ")

    time = np.linspace(0, 1, len(Id1_R))
    fig, axes = multi_path_panels(
        time, Id1_R, Id1_r_bht, Id1_r_bas,
        Id1_r1_s, Id1_r2_s, Id1_r1_sht, Id1_r1_sas,
        Id1_r2_sht, Id1_r2_sas, Id1_E, Id1_GE,
        out_pdf=out_pdf, case_label=pretty_case
    )

    # export individual panels (same filenames you used)
    from .mc_singletrigger_plots import save_subplot
    save_subplot(fig, axes[0,0], out_a, pad=0.3)
    save_subplot(fig, axes[0,1], out_b, pad=0.3)
    save_subplot(fig, axes[1,0], out_c, pad=0.3)
    save_subplot(fig, axes[1,1], out_d, pad=0.3)

    # --- make computational cost histograms (Event / Trigger cost) ---

    if len(test_Ecost_both) > 0:
        test_Ecost_both_arr = np.array(test_Ecost_both)
        test_Ecost_ht_arr   = np.array(test_Ecost_ht)
        test_Ecost_as_arr   = np.array(test_Ecost_as)

        test_Tcost_both_arr = np.array(test_Tcost_both)
        test_Tcost_ht_arr   = np.array(test_Tcost_ht)
        test_Tcost_as_arr   = np.array(test_Tcost_as)

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

        # Event cost histogram
        axes2[0].hist(
            test_Ecost_both_arr, bins=20, density=True, histtype="step", color='tab:blue',
            linewidth=2, label=f"both paths, mean = {np.mean(test_Ecost_both_arr):.1f}"
        )
        axes2[0].hist(
            test_Ecost_ht_arr, bins=20, density=True, histtype="step",color='tab:orange',
            linewidth=2, label=f"HT path only, mean = {np.mean(test_Ecost_ht_arr):.1f}"
        )
        axes2[0].hist(
            test_Ecost_as_arr, bins=20, density=True, histtype="step",color='tab:green',
            linewidth=2, label=f"AD path only, mean = {np.mean(test_Ecost_as_arr):.1f}"
        )
        axes2[0].set_xlabel(f"Event Cost in {pretty_case}")
        axes2[0].set_ylabel("Density")
        axes2[0].set_ylim(0, 1)
        axes2[0].legend(loc="best", fontsize=12)

        # Trigger cost histogram
        axes2[1].hist(
            test_Tcost_both_arr, bins=20, density=True, histtype="step",color='tab:blue',
            linewidth=2, label=f"both paths, mean = {np.mean(test_Tcost_both_arr):.1f}"
        )
        axes2[1].hist(
            test_Tcost_ht_arr, bins=20, density=True, histtype="step",color='tab:orange',
            linewidth=2, label=f"HT path only, mean = {np.mean(test_Tcost_ht_arr):.1f}"
        )
        axes2[1].hist(
            test_Tcost_as_arr, bins=20, density=True, histtype="step",color='tab:green',
            linewidth=2, label=f"AD path only, mean = {np.mean(test_Tcost_as_arr):.1f}"
        )
        axes2[1].set_xlabel(f"Trigger Cost in {pretty_case}")
        axes2[1].set_ylabel("Density")
        axes2[1].set_yscale("log")
        axes2[1].legend(loc="best", fontsize=12)

        plt.tight_layout()
        out_comp = f"{args.outdir}/comp_cost_test({case_label}).pdf"
        fig2.savefig(out_comp)
        plt.close(fig2)

if __name__ == "__main__":
    main()
