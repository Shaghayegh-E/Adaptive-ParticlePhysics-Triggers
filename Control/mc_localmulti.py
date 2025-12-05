# CLI: orchestrates chunks, calls agents, saves figures
from __future__ import annotations
import argparse, numpy as np
import matplotlib.pyplot as plt
try:
    import atlas_mpl_style as aplt
    aplt.use_atlas_style()
except Exception:
    pass
import os
from .mc_singletrigger_io import load_trigger_food_mc
from .agents_local_multi import V3, local_V3, V1, local_V1, V2, local_V2, Trigger
from .metrics import update_accumulated, average_perf_bins
from .mc_singletrigger_plots import rate_efficiency_panels, plot_evolution, plot_evolution_ideal

def pick_agent(name: str):
    name = name.lower()

    # Wrap agents so they all share the same call signature:
    #   global_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,snj1,snj2)
    #   local_agent(bht,sht1,sht2,bas,sas1,sas2,bnj,Ht_cut,AS_cut)
    if name == "v1":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2):
            # V1 does not use jet multiplicities; ignore bnj/snj*
            return V1(bht, sht1, sht2, bas, sas1, sas2)
        def local_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2, Ht_cut, AS_cut):
            return local_V1(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2, Ht_cut, AS_cut)
        return global_agent, local_agent, "case1"

    if name == "v2":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2):
            return V2(bht, sht1, sht2, bas, sas1, sas2)
        def local_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2, Ht_cut, AS_cut):
            return local_V2(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2, Ht_cut, AS_cut)
        return global_agent, local_agent, "case2"

    if name == "v3":
        def global_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2):
            return V3(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2)
        def local_agent(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2,Ht_cut, AS_cut):
            return local_V3(bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2, Ht_cut, AS_cut)
        return global_agent, local_agent, "case3"

    raise ValueError(f"Unknown agent name {name}")


def main():
    p = argparse.ArgumentParser(description="Local multi-step controller over Trigger_food datasets")
    p.add_argument("--path", default="Data/Trigger_food_MC.h5")
    p.add_argument("--agent", default="v3", choices=["v1","v2","v3"])
    p.add_argument("--chunk", type=int, default=50000)
    p.add_argument("--n_bins", type=int, default=25)
    p.add_argument("--outdir", default="outputs/demo_localmulti")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    global_agent, local_agent, case_label = pick_agent(args.agent)

    D = load_trigger_food_mc(args.path)
    Bas = D["mc_bkg_score01"]; Bht = D["mc_bkg_ht"]; Bnpv = D["mc_bkg_Npv"]; Bnj = D["mc_bkg_njets"]
    Sas1, Sht1, Snpv1, Snj1 = D["mc_tt_score01"], D["mc_tt_ht"], D["tt_Npv"], D["mc_tt_njets"]
    Sas2, Sht2, Snpv2, Snj2 = D["mc_aa_score01"], D["mc_aa_ht"], D["aa_Npv"], D["mc_aa_njets"]

    Nb = len(Bnpv)
    fixed_Ht_cut = np.percentile(Bht[500000:600000],99.8)
    fixed_AS_cut = np.percentile(Bas[500000:600000],99.9)

    Ht_cut = fixed_Ht_cut
    AS_cut = fixed_AS_cut

    # storage
    R=[]; Rht=[]; Ras=[]; Id1_R=[]
    E=[]; GE=[]; Eht=[]; Eas=[]; Id1_E=[]; Id1_GE=[]
    total_samples_ef=[]; total_samples_r_s=[]
    ideal_H=[]; ideal_A=[]  # contour_i1, contour_j1,
    real_H=[]; real_A=[] #contour_f, contour_g,
    five_win=[]

    abs_rb_list = []
    rs_list = []
    b_TC_list = []
    b_EC_list = []
    total_cost_list = []
    performance_list = []


    for I in range(0, Nb, args.chunk):
        if I < 10*args.chunk:  # skip warmup region like your code
            continue

        if I==10*args.chunk: 
            idx = slice(I, min(I+args.chunk, Nb))
            bht, bas, bnpv, bnj = Bht[idx], Bas[idx], Bnpv[idx], Bnj[idx]
            npv_min, npv_max = float(bnpv.min()), float(bnpv.max())
            # Select signal events that fall within this Npv range
            mask1 = (Snpv1 >= npv_min) & (Snpv1 <= npv_max)
            mask2 = (Snpv2 >= npv_min) & (Snpv2 <= npv_max)
            sht1, sas1, snj1 = Sht1[mask1], Sas1[mask1], Snj1[mask1]
            sht2, sas2, snj2 = Sht2[mask2], Sas2[mask2], Snj2[mask2]


            cost, r_b, r1_s, r2_s, HT, AS = global_agent(
                bht, sht1, sht2, bas, sas1, sas2, bnj, snj1, snj2
            )
            ii, jj = np.unravel_index(np.argmin(cost), cost.shape)
            Ht_cut, AS_cut = float(HT[ii, jj]), float(AS[ii, jj])
            ideal_H.append(float(HT[ii, jj]))
            ideal_A.append(float(AS[ii, jj]))

        if I%args.chunk==0:
            idx = slice(I, min(I+args.chunk, Nb))

            bht, bas, bnpv, bnj = Bht[idx], Bas[idx], Bnpv[idx], Bnj[idx]
        
            npv_min, npv_max = float(bnpv.min()), float(bnpv.max())

            # Select signal events that fall within this Npv range
            signal_mask1 = (Snpv1 >= npv_min) & (Snpv1 <= npv_max)
    
            # Extract matching signal events
            sht1 = Sht1[signal_mask1]
            sas1 = Sas1[signal_mask1]
            snjets1 = Snj1[signal_mask1]

            signal_mask2 = (Snpv2 >= npv_min) & (Snpv2 <= npv_max)
            # Extract matching signal events
            sht2 = Sht2[signal_mask2]
            sas2 = Sas2[signal_mask2]
            snjets2 = Snj2[signal_mask2]
            r, ef, rht, ras, efht, efas, b_EC_val, b_TC_val = Trigger(bht,sht1,sht2,bas,sas1,sas2, bnj, Ht_cut,AS_cut)

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
            real_A.append(AS_cut)
        
        
            cost, r_b, r_s, HT, AS = local_agent(
            bht, sht1, sht2, bas, sas1, sas2, bnj, snjets1, snjets2, Ht_cut, AS_cut) #r_bht, r_bas, r_sht, r_sas, 

            bestcost_index = np.argmin(cost)
            i, j = np.unravel_index(bestcost_index, cost.shape)
        
            Ht_cut = HT[i,j]
            AS_cut = AS[i,j]
        
            five_win.append([HT[i,j],AS[i,j]])
            if len(five_win)>=5: 
                Ht_cut, AS_cut = np.mean(np.array(five_win)[-5:], axis=0)
            
            #b_overlap, s1_overlap, s2_overlap, r_bht, r_bas, r1_sht, r2_sht, r1_sas, r2_sas,
            cost, r_b, r1_s, r2_s, HT, AS= global_agent(
            bht, sht1, sht2, bas, sas1, sas2, bnj, snjets1, snjets2)

            bestcost_index = np.argmin(cost)
            i, j = np.unravel_index(bestcost_index, cost.shape)


            ideal_H.append(HT[i, j])
            ideal_A.append(AS[i, j])
        
        
            Id1_R.append(r_b[i, j])
        
            r_s = (r1_s[i, j]*sht1.shape[0] + r2_s[i, j]*sht2.shape[0])/(sht1.shape[0]+sht2.shape[0])
            Id1_E.append(r_s)  # Total signal rate

    
            update_accumulated(Id1_GE, r_s, (sht1.shape[0]+sht2.shape[0]), total_samples_r_s)

            print('time index:',I)


     

    # ---- Post: average over bins + plot
    time = np.linspace(0, 1, len(R))
    def avg(x): 
        return average_perf_bins(x, n_bins=args.n_bins)
    time = average_perf_bins(time, n_bins=args.n_bins)
    R, Rht, Ras, Id1_R = avg(R)*400, avg(Rht)*400, avg(Ras)*400, avg(Id1_R)*400
    E, GE, Eht, Eas = avg(E), avg(GE), avg(Eht), avg(Eas)  # placeholders if you split ht/as eff separately
    Id1_E, Id1_GE = avg(Id1_E), avg(Id1_GE)

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
    )
    # Evolution plot: ideal vs real thresholds in the (Ht, AD) plane
    evo_name = f"real_evoultion_{case_label}.pdf"  
    plot_evolution(
        ideal_H, ideal_A,
        real_H, real_A,
        title=f"real_evoultion_{case_label}",
        out_pdf=f"{args.outdir}/{evo_name}",
    )
    plot_evolution_ideal(
        ideal_H, ideal_A,
        real_H, real_A,
        title=f"ideal_evoultion_{case_label}",
        out_pdf=f"{args.outdir}/{evo_name}",
    )
    #contour_i1,contour_j1,contour_f,contour_g,

if __name__ == "__main__":
    main()
