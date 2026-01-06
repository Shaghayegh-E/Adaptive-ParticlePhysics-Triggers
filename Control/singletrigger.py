#!/usr/bin/env python
import sys
from pathlib import Path

# Make repo root importable 
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import argparse
from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
from .trigger_io import read_mc_data

SCRIPT_DIR = Path(__file__).resolve().parent

def main():
    
    ap = argparse.ArgumentParser(description="Single Trigger Run")

    ap.add_argument("--bkgType", default="MC", choices=["MC", "RealData"],
                    help="MC: Monte Carlo simulated samples MinBias bkg; RealData: real data background run. AA/TT are always MC.")

    args = ap.parse_args()
    bkgType = args.bkgType
    if bkgType=="MC":
        PATH = "Data/Trigger_food_MC.h5"
    else:
        PATH = "Data/Trigger_food_Data.h5"


    (
        Sas_tot1, Sht_tot1, S_npvs1,
        Sas_tot2, Sht_tot2, S_npvs2,
        Bas_tot,  Bht_tot,  B_npvs
    ) = read_mc_data(PATH)

    Nb = len(B_npvs)
    N = Nb

    pre_r1   = 0.0
    pre_r2 = 0.0

    if bkgType=="MC":
        fixed_Ht_cut  = np.percentile(Bht_tot[500000:600000], 99.75)
        fixed_AD_cut = np.percentile(Bas_tot[500000:600000], 99.75)
        print("fixed_Ht_cut",  fixed_Ht_cut)
        print("fixed_AS_cut1", fixed_AD_cut)

        print("Bht 99.75 after 500k:", np.percentile(Bht_tot[500000:], 99.75))
        chunk_size = 50000

    
    else:
        fixed_Ht_cut  = np.percentile(Bht_tot[200000:240000], 99.75)
        fixed_AD_cut = np.percentile(Bas_tot[200000:240000], 99.75)
        print("fixed_Ht_cut",  fixed_Ht_cut)
        print("fixed_AS_cut1", fixed_AD_cut)

        print("Bht 99.75 after 500k:", np.percentile(Bht_tot[200000:], 99.75))
        chunk_size = 20000

        


    #percen_9975 = np.percentile(Bas_tot, 99.75)
    #AA_passed = 100 * np.sum(Sas_tot2 > percen_9975) / len(Sas_tot2)
    #TT_passed = 100 * np.sum(Sas_tot1 > percen_9975) / len(Sas_tot1)
    #print("AA_passed", AA_passed)
    #print("TT_passed", TT_passed)

    Ht_cut  = fixed_Ht_cut
    AD_cut = fixed_AD_cut


    # --- containers ---
    R1, R2 = [], []
    L_R3, L_R4, R3, R4 = [], [], [0.0], [0.0]
    L_R5, L_R6, R5, R6 = [], [], [0.0], [0.0]

    E1, E2 = [], []
    L_E3, L_E4, E3, E4 = [], [], [0.0], [0.0]
    L_E5, L_E6, E5, E6 = [], [], [0.0], [0.0]
    


    for I in range(N):
        if I < 10*chunk_size:
            continue

        if I % chunk_size == 0:
            start_idx = I
            end_idx   = min(I + chunk_size, N)
            indices   = list(range(start_idx, end_idx))

            # background
            bht   = Bht_tot[indices]
            bas  = Bas_tot[indices]
            b_npvs = B_npvs[indices]
            
        
            
            
            if bkgType=="MC":
                npv_min = np.min(b_npvs)
                npv_max = np.max(b_npvs)

                signal_mask1 = (S_npvs1 >= npv_min) & (S_npvs1 <= npv_max)
                signal_mask2 = (S_npvs2 >= npv_min) & (S_npvs2 <= npv_max)

                # ttbar
                sht1   = Sht_tot1[signal_mask1]
                sas1 = Sas_tot1[signal_mask1]

                # H→AATo4B
                sht2   = Sht_tot2[signal_mask2]
                sas2 = Sas_tot2[signal_mask2]
            else:
                #ttbar
                sht1   = Sht_tot1[indices]
                sas1 = Sas_tot1[indices]

                # H→AATo4B
                sht2   = Sht_tot2[indices]
                sas2 = Sas_tot2[indices]

                    

            # ---------- HT trigger ----------
            rate1 = Sing_Trigger(np.array(bht), fixed_Ht_cut)
            rate2 = Sing_Trigger(np.array(bht), Ht_cut)
            R1.append(rate1)
            R2.append(rate2)

            # ttbar signal cumulative + local
            rate3 = Sing_Trigger(np.array(sht1), fixed_Ht_cut)
            rate4 = Sing_Trigger(np.array(sht1), Ht_cut)
            L_R3.append(rate3)
            L_R4.append(rate4)
            a = ((len(R3) - 1) * R3[-1] + rate3) / len(R3)
            R3.append(a)
            b = ((len(R4) - 1) * R4[-1] + rate4) / len(R4)
            R4.append(b)

            # AA signal cumulative + local
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

            # ---------- AD trigger background ----------
            rate1 = Sing_Trigger(np.array(bas), fixed_AD_cut)
            rate2 = Sing_Trigger(np.array(bas), AD_cut)
            E1.append(rate1)
            E2.append(rate2)


            # ttbar dim=2
            rate3 = Sing_Trigger(np.array(sas1), fixed_AD_cut)
            rate4 = Sing_Trigger(np.array(sas1), AD_cut)
            L_E3.append(rate3)
            L_E4.append(rate4)
            a = ((len(E3) - 1) * E3[-1] + rate3) / len(E3)
            E3.append(a)
            b = ((len(E4) - 1) * E4[-1] + rate4) / len(E4)
            E4.append(b)


            # AA dim=2
            rate3 = Sing_Trigger(np.array(sas2), fixed_AD_cut)
            rate4 = Sing_Trigger(np.array(sas2), AD_cut)
            L_E5.append(rate3)
            L_E6.append(rate4)
            a = ((len(E5) - 1) * E5[-1] + rate3) / len(E5)
            E5.append(a)
            b = ((len(E6) - 1) * E6[-1] + rate4) / len(E6)
            E6.append(b)

            

            # update AS cuts
            AD_cut, pre_r2 = PD_controller2(E2[-1], pre_r2, AD_cut)

    # ----- convert to arrays + scale to kHz where needed -----
    E1 = np.array(E1) * 400
    E2 = np.array(E2) * 400


    R1 = np.array(R1) * 400
    R2 = np.array(R2) * 400

    # drop the initial 0 sentinel
    R3 = np.array(R3[1:])
    R4 = np.array(R4[1:])
    R5 = np.array(R5[1:])
    R6 = np.array(R6[1:])

    E3 = np.array(E3[1:])
    E4 = np.array(E4[1:])
    
    E5 = np.array(E5[1:])
    E6 = np.array(E6[1:])


    L_R3  = np.array(L_R3)
    L_R4  = np.array(L_R4)
    L_R5  = np.array(L_R5)
    L_R6  = np.array(L_R6)
    L_E3 = np.array(L_E3)
    L_E4 = np.array(L_E4)

    L_E5 = np.array(L_E5)
    L_E6 = np.array(L_E6)


    # ----- save everything to npz -----
    if bkgType=="MC" :
        OUT_NPZ = SCRIPT_DIR / "singletrigger_results_mc.npz"
    else:
        OUT_NPZ = SCRIPT_DIR / "singletrigger_results_realdata.npz"
        
    np.savez(
        OUT_NPZ,
        R1=R1, R2=R2,
        R3=R3, R4=R4, R5=R5, R6=R6,
        L_R3=L_R3, L_R4=L_R4, L_R5=L_R5, L_R6=L_R6,
        E1=E1, E2=E2,
        E3=E3, E4=E4,
        E5=E5, E6=E6,
        L_E3=L_E3, L_E4=L_E4,
        L_E5=L_E5, L_E6=L_E6,
        fixed_Ht_cut=fixed_Ht_cut,
        fixed_AD_cut=fixed_AD_cut
    )
    print(f"Saved results to {OUT_NPZ}")

if __name__ == "__main__":
    main()
