# CLI: calls agents, saves figures
from __future__ import annotations
import argparse, numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import os

from .agents import V3, local_V3, V1, local_V1, V2, local_V2, Trigger
from .metrics import update_accumulated, average_perf_bins
from .Trigger_io import rate_efficiency_panels, evolution, load_trigger_food

def pick_agent(name: str):
    name = name.lower()

    # Wrap agents so they all share the same call signature:
    #   global_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,snj1,snj2)
    #   local_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,Ht_cut,AS_cut)
    if name == "v1":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj,ref):
            # V1 does not use jet multiplicities; ignore bnj/snj*
            return V1(bht, sht1, sht2, bas, sas1, sas2)
        def local_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref, Ht_cut, AS_cut):
            return local_V1(bht, sht1, sht2, bas, sas1, sas2, Ht_cut, AS_cut)
        return global_agent, local_agent, "case1"

    if name == "v2":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj,ref):
            return V2(bht, sht1, sht2, bas, sas1, sas2)
        def local_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref, Ht_cut, AS_cut):
            return local_V2(bht, sht1, sht2, bas, sas1, sas2, Ht_cut, AS_cut)
        return global_agent, local_agent, "case2"

    if name == "v3":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref):
            return V3(bht, sht1, sht2, bas, sas1, sas2, bnj, ref)
        def local_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, ref,Ht_cut, AS_cut):
            return local_V3(bht, sht1, sht2, bas, sas1, sas2, bnj, ref, Ht_cut, AS_cut)
        return global_agent, local_agent, "case3"

    raise ValueError(f"Unknown agent name {name}")


def main():
    p = argparse.ArgumentParser(description="Local multi-step controller over Trigger_food datasets")
    p.add_argument("--bkgType", default="MC") 
    p.add_argument("--path", default="Data/Trigger_food_MC.h5")
    p.add_argument("--agent", default="v3", choices=["v1","v2","v3"])
    #p.add_argument("--chunk", type=int, default=50000)
    #p.add_argument("--n_bins", type=int, default=25)
    p.add_argument("--outdir", default="outputs/demo_RealMultiTrigger_mc")
    p.add_argument("--costRef", type=float, nargs="+", default=[5.6,2.7])
    p.add_argument("--forceCostRef", action="store_true")

    args = p.parse_args()
    
    if args.bkgType=="RealData":
        chunk_size = 20000
        CostRef = [4.3, 2.9]
    else :
        chunk_size = 50000
        CostRef = [5.6, 2.7]
        
    if args.forceCostRef : CostRef = args.costRef
    
    os.makedirs(args.outdir, exist_ok=True)
    global_agent, local_agent, case_label = pick_agent(args.agent)

    Sas1, Sht1, Snpv1, Snj1,\
    Sas2, Sht2, Snpv2, Snj2,\
    Bas,  Bht,  Bnpv, Bnj = load_trigger_food(args.path)
    
    Nb = len(Bnpv)

    # storage
    R=[]; Rht=[]; Ras=[]; Id1_R=[]
    E=[]; GE=[]; Eht=[]; Eas=[]; Id1_E=[]; Id1_GE=[]
    total_samples_ef=[]; total_samples_r_s=[]

    real_H=[]; real_A=[] 
    five_win=[]

    abs_rb_list = []
    rs_list = []
    b_TC_list = []
    b_EC_list = []
    total_cost_list = []
    performance_list = []


    for I in range(0, Nb, chunk_size):
        if I < 10*chunk_size:  # warmup region 
            continue

        if I==10*chunk_size: 
            idx = slice(I, min(I+chunk_size, Nb))
            bht, bas, bnpv, bnj = Bht[idx], Bas[idx], Bnpv[idx], Bnj[idx]
            npv_min, npv_max = float(bnpv.min()), float(bnpv.max())
            # Select signal events that fall within this Npv range
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


            cost, *_, HT, AD = global_agent(
                bht, sht1, sht2, bas, sas1, sas2, bnj, CostRef
            )
            ii, jj = np.unravel_index(np.argmin(cost), cost.shape)
            Ht_cut, AD_cut = float(HT[ii, jj]), float(AD[ii, jj])


        if I%chunk_size==0:
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


            r, ef, rht, ras, efht, efas, b_EC_val, b_TC_val = Trigger(bht,sht1,sht2,bas,sas1,sas2, bnj, Ht_cut,AD_cut)

            abs_rb = 1/(1 - (np.abs(r - 0.25)))
            rs_val = ef

            total_cost_val = b_EC_val + b_TC_val
            performance_val = abs_rb * rs_val
        
            abs_rb_list.append(abs_rb)
            rs_list.append(rs_val)
            b_TC_list.append(b_TC_val)
            b_EC_list.append(b_EC_val)
            total_cost_list.append(total_cost_val)
            performance_list.append(performance_val)
        
            R.append(r)
            Rht.append(rht)
            Ras.append(ras)

            E.append(ef)
            Eht.append(efht)
            Eas.append(efas)

            update_accumulated(GE, ef, (sht1.shape[0]+sht2.shape[0]), total_samples_ef)

            #real point evo
            real_H.append(Ht_cut)
            real_A.append(AD_cut)
        
        
            cost, r_b, r_s, HT, AD = local_agent(
            bht, sht1, sht2, bas, sas1, sas2, bnj, CostRef, Ht_cut, AD_cut)

            bestcost_index = np.argmin(cost)
            i, j = np.unravel_index(bestcost_index, cost.shape)
        
            Ht_cut = HT[i,j]
            AD_cut = AD[i,j]
        
            five_win.append([HT[i,j],AD[i,j]])
            if len(five_win)>=5: 
                Ht_cut, AD_cut = np.mean(np.array(five_win)[-5:], axis=0)
            
            #b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas,
            cost, r_b, r1_s, r2_s, *_ = global_agent(
            bht, sht1, sht2, bas, sas1, sas2, bnj, CostRef)

            bestcost_index = np.argmin(cost)
            i, j = np.unravel_index(bestcost_index, cost.shape)
        
            Id1_R.append(r_b[i, j])
        
            r_s = (r1_s[i, j]*sht1.shape[0] + r2_s[i, j]*sht2.shape[0])/(sht1.shape[0]+sht2.shape[0])
            Id1_E.append(r_s)  # Total signal rate

    
            update_accumulated(Id1_GE, r_s, (sht1.shape[0]+sht2.shape[0]), total_samples_r_s)

            print('time index:',I)



    # ---- Post: average over bins + plot
    time = np.linspace(0, 1, len(R))
    #n_bins = 25

    time = average_perf_bins(time)
    R, Rht, Ras, Id1_R = average_perf_bins(np.array(R)*400), average_perf_bins(np.array(Rht)*400), average_perf_bins(np.array(Ras)*400), average_perf_bins(np.array(Id1_R)*400)
    E, GE, Eht, Eas = average_perf_bins(np.array(E)), average_perf_bins(np.array(GE)), average_perf_bins(np.array(Eht)), average_perf_bins(np.array(Eas))  
    Id1_E, Id1_GE = average_perf_bins(Id1_E), average_perf_bins(Id1_GE)

    base = f"simple_controller_{case_label}(averaged)"
    out_pdf = f"{args.outdir}/{base}.pdf"
    out_a   = f"{args.outdir}/{base}-a.pdf"
    out_b   = f"{args.outdir}/{base}-b.pdf"

    pretty_case = case_label if case_label.startswith("Case") else case_label.capitalize().replace("case", "Case ")

    rate_efficiency_panels(
        time, R, Rht, Ras, Id1_R, GE, Eht, Eas, Id1_GE,
        out_pdf=out_pdf,
        out_a=out_a,
        out_b=out_b,
        case_label=pretty_case,  
        bkg=args.bkgType
    )
    
    # Evolution plot
    evolution(
        real_H, real_A,
        title=f"real_evoultion_{case_label}",
        outdir=f"{args.outdir}",
    )


if __name__ == "__main__":
    main()
