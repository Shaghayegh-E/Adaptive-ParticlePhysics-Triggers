# trigger/run_multi_path.py
from __future__ import annotations
import argparse, numpy as np, matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


import mplhep as hep
hep.style.use("CMS")

import os
from .Trigger_io import load_trigger_food

from .metrics import comp_cost_test



def main():
    p = argparse.ArgumentParser(description="Multi-path ideal grid scan across chunks")
    p.add_argument("--bkgType", default="MC")     
    p.add_argument("--path", default="Data/Trigger_food_MC.h5")    
    #p.add_argument("--chunk", type=int, default=50000)
    #p.add_argument("--warmup_chunks", type=int, default=10)       # skip first N chunks
    p.add_argument("--outdir", default="outputs/demo_IdealMultiTrigger_mc")

    args = p.parse_args()
    
    if args.bkgType=="RealData":
        chunk_size = 20000
    else :
        chunk_size = 50000

    os.makedirs(args.outdir, exist_ok=True)
    
    #global_agent = V1
    case_label = "case1"

    Sas1, Sht1, Snpv1, Snj1,\
    Sas2, Sht2, Snpv2, Snj2,\
    Bas,  Bht,  Bnpv, Bnj = load_trigger_food(args.path)
    
    

    
    
    Nb = len(Bnpv)

    
    # --- storage for computational cost test ---
    test_Ecost_both, test_Tcost_both = [], []
    test_Ecost_ht,   test_Tcost_ht   = [], []
    test_Ecost_as,   test_Tcost_as   = [], []

    for I in range(0, Nb, chunk_size):
        if I < 10*chunk_size:
            continue
        
        if I%chunk_size==0 : 
            idx = slice(I, min(I+chunk_size, Nb))
            bht, bas, bnpv, bnj = Bht[idx], Bas[idx], Bnpv[idx], Bnj[idx]

            npv_min, npv_max = float(bnpv.min()), float(bnpv.max())
            m1 = (Snpv1 >= npv_min) & (Snpv1 <= npv_max)
            m2 = (Snpv2 >= npv_min) & (Snpv2 <= npv_max)
            
            if args.bkgType=="MC":
                sht1, sas1, snj1 = Sht1[m1], Sas1[m1], Snj1[m1]
                sht2, sas2, snj2 = Sht2[m2], Sas2[m2], Snj2[m2]
            else:
                sht1   = Sht1[idx]
                sas1 = Sas1[idx]
                snj1 = Snj1[idx]
                

                # H→AATo4B
                sht2   = Sht2[idx]
                sas2 = Sas2[idx]
                snj2 = Snj2[idx]

            # --- computational cost diagnostics (prepare for V3) ---

            # both paths on

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

            # AD path only
            cost_as, E_as, T_as = comp_cost_test(
                bht, bas, bnj, sht1, sas1, snj1, sht2, sas2, snj2,
                use_path1=False, use_path2=True
            )
            ia, ja = np.unravel_index(np.argmin(cost_as), cost_as.shape)
            test_Ecost_as.append(float(E_as[ia, ja]))
            test_Tcost_as.append(float(T_as[ia, ja]))


            print("time index:", I)
            




    pretty_case = case_label if case_label.startswith("Case") else case_label.capitalize().replace("case", "Case ")


    # --- computational cost histograms (Event / Trigger cost) ---

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
        #axes2[0].set_ylim(0, 2)
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
        
        print("########### Suggested Reference Costs for Running Case3 ###########")
        print(f"Event Level Reference: {np.mean(test_Ecost_both_arr):.1f}")
        print(f"Trigger Path Level Reference: {np.mean(test_Tcost_both_arr):.1f}")

if __name__ == "__main__":
    main()
