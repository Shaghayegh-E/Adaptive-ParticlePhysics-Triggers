# ideal trigger
from __future__ import annotations
import argparse, numpy as np, matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


import mplhep as hep
hep.style.use("CMS")

import os
from .Trigger_io import load_trigger_food
from .agents import V3, V1, V2

from .metrics import update_accumulated
from .Trigger_io import multi_path_panels, save_subplot, evolution

def pick_agent(name: str):
    name = name.lower()

    # Wrap agents so they all share the same call signature:
    #   Ideal_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,snj1,snj2)
    #   local_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,Ht_cut,AS_cut)
    if name == "v1":
        def Ideal_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref):
            return V1(bht, sht1, sht2, bas, sas1, sas2)
        local_agent = None
        return Ideal_agent, local_agent, "case1"

    if name == "v2":
        def Ideal_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref):
            return V2(bht, sht1, sht2, bas, sas1, sas2)
        local_agent = None
        return Ideal_agent, local_agent, "case2"

    if name == "v3":
        def Ideal_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref):
            return V3(bht, sht1, sht2, bas, sas1, sas2, bnj, ref)
        local_agent = None
        return Ideal_agent, local_agent, "case3"

    raise ValueError(f"Unknown agent name {name}")

def main():
    p = argparse.ArgumentParser(description="Multi-path ideal grid scan across chunks")
    p.add_argument("--bkgType", default="MC")     
    p.add_argument("--path", default="Data/Trigger_food_MC.h5")    
    p.add_argument("--agent", default="v1", choices=["v1","v2","v3"])
    #p.add_argument("--chunk", type=int, default=50000)
    #p.add_argument("--warmup_chunks", type=int, default=10)       # skip first N chunks
    p.add_argument("--outdir", default="outputs/demo_IdealMultiTrigger_mc")
    p.add_argument("--costRef", type=float, nargs="+", default=[5.7, 2.5])
    p.add_argument("--forceCostRef", action="store_true")
    args = p.parse_args()
    
    if args.bkgType=="RealData":
        chunk_size = 20000
        CostRef = [4.3, 3.5]
    else :
        chunk_size = 50000
        CostRef = [5.7, 2.5]
        
    if args.forceCostRef : CostRef = args.costRef


    os.makedirs(args.outdir, exist_ok=True)
    global_agent, local_agent, case_label = pick_agent(args.agent)

    Sas1, Sht1, Snpv1, Snj1,\
    Sas2, Sht2, Snpv2, Snj2,\
    Bas,  Bht,  Bnpv, Bnj = load_trigger_food(args.path)
    
    Nb = len(Bnpv)

    # storage
    contour_H, contour_A = [], []
    Id1_R, Id1_E, Id1_GE = [], [], []
    Id1_r1_s, Id1_r2_s = [], []
    Id1_r_bht, Id1_r_bas = [], []
    Id1_r1_sht, Id1_r2_sht, Id1_r1_sas, Id1_r2_sas = [], [], [], []
    total_samples_r_s = []


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
                sht1, sas1 = Sht1[m1], Sas1[m1]
                sht2, sas2 = Sht2[m2], Sas2[m2]
            else:
                #ttbar
                sht1   = Sht1[idx]
                sas1 = Sas1[idx]

                # H→AATo4B
                sht2   = Sht2[idx]
                sas2 = Sas2[idx]

        
            cost, r_b, r1_s, r2_s,s1_overlap,s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas, HT, AS = global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, CostRef)


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

            update_accumulated(Id1_GE, r_s, (sht1.shape[0]+sht2.shape[0]), total_samples_r_s)

            print("time index:", I)
            
    #finished all chunks, make plots
    Id1_R = np.array(Id1_R)*400
    Id1_r_bht = np.array(Id1_r_bht)*400
    Id1_r_bas = np.array(Id1_r_bas)*400

    #Ideal evolution -> Motivation for Local Multi
    
    evolution(
        contour_H, contour_A,
        title=f"ideal_evoultion_{case_label}",
        outdir=f"{args.outdir}",
    )


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
        out_pdf=out_pdf, case=pretty_case
    )

    # export individual panels (same filenames)
    save_subplot(fig, axes[0,0], out_a, pad=0.3)
    save_subplot(fig, axes[0,1], out_b, pad=0.3)
    save_subplot(fig, axes[1,0], out_c, pad=0.3)
    save_subplot(fig, axes[1,1], out_d, pad=0.3)

    

if __name__ == "__main__":
    main()
