#!/usr/bin/env python
import sys
from pathlib import Path

# Make repo root importable 
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
from .mc_singletrigger_io import read_mc_data

PATH = "Data/Trigger_food_MC.h5"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_NPZ = SCRIPT_DIR / "mc_singletrigger_results.npz"
def main():
    (
        Sas01_tot1, Sas04_tot1, Sht_tot1, S_npvs1,
        Sas01_tot2, Sas04_tot2, Sht_tot2, S_npvs2,
        Bas01_tot,  Bas04_tot,  Bht_tot,  B_npvs
    ) = read_mc_data(PATH)

    Nb = len(B_npvs)
    N = Nb

    pre_r1   = 0.0
    pre_r2_1 = 0.0
    pre_r2_4 = 0.0

    fixed_Ht_cut  = np.percentile(Bht_tot[500000:600000], 99.75)
    fixed_AS_cut1 = np.percentile(Bas01_tot[500000:600000], 99.75)
    fixed_AS_cut4 = np.percentile(Bas04_tot[500000:600000], 99.75)

    print("fixed_Ht_cut",  fixed_Ht_cut)
    print("fixed_AS_cut1", fixed_AS_cut1)
    print("fixed_AS_cut4", fixed_AS_cut4)
    print("Bht 99.75 after 500k:", np.percentile(Bht_tot[500000:], 99.75))

    percen_9975 = np.percentile(Bas04_tot, 99.75)
    AA_passed = 100 * np.sum(Sas04_tot2 > percen_9975) / len(Sas04_tot2)
    TT_passed = 100 * np.sum(Sas04_tot1 > percen_9975) / len(Sas04_tot1)
    print("AA_passed", AA_passed)
    print("TT_passed", TT_passed)

    Ht_cut  = fixed_Ht_cut
    AS_cut1 = fixed_AS_cut1
    AS_cut4 = fixed_AS_cut4

    # --- containers ---
    R1, R2 = [], []
    L_R3, L_R4, R3, R4 = [], [], [0.0], [0.0]
    L_R5, L_R6, R5, R6 = [], [], [0.0], [0.0]

    E1_1, E2_1, E1_4, E2_4 = [], [], [], []
    L_E3_1, L_E4_1, E3_1, E4_1 = [], [], [0.0], [0.0]
    L_E3_4, L_E4_4, E3_4, E4_4 = [], [], [0.0], [0.0]
    L_E5_1, L_E6_1, E5_1, E6_1 = [], [], [0.0], [0.0]
    L_E5_4, L_E6_4, E5_4, E6_4 = [], [], [0.0], [0.0]

    chunk_size = 50000

    for I in range(N):
        if I < 500000:
            continue

        if I % chunk_size == 0:
            start_idx = I
            end_idx   = min(I + chunk_size, N)
            indices   = list(range(start_idx, end_idx))

            # background
            bht   = Bht_tot[indices]
            bas1  = Bas01_tot[indices]
            bas4  = Bas04_tot[indices]
            b_npvs = B_npvs[indices]

            npv_min = np.min(b_npvs)
            npv_max = np.max(b_npvs)

            signal_mask1 = (S_npvs1 >= npv_min) & (S_npvs1 <= npv_max)
            signal_mask2 = (S_npvs2 >= npv_min) & (S_npvs2 <= npv_max)

            # ttbar
            sht1   = Sht_tot1[signal_mask1]
            sas1_1 = Sas01_tot1[signal_mask1]
            sas1_4 = Sas04_tot1[signal_mask1]

            # H→AATo4B
            sht2   = Sht_tot2[signal_mask2]
            sas2_1 = Sas01_tot2[signal_mask2]
            sas2_4 = Sas04_tot2[signal_mask2]

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

            # AA signal
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
            rate1_1 = Sing_Trigger(np.array(bas1), fixed_AS_cut1)
            rate2_1 = Sing_Trigger(np.array(bas1), AS_cut1)
            E1_1.append(rate1_1)
            E2_1.append(rate2_1)

            rate1_4 = Sing_Trigger(np.array(bas4), fixed_AS_cut4)
            rate2_4 = Sing_Trigger(np.array(bas4), AS_cut4)
            E1_4.append(rate1_4)
            E2_4.append(rate2_4)

            # ttbar dim=1
            rate3_1 = Sing_Trigger(np.array(sas1_1), fixed_AS_cut1)
            rate4_1 = Sing_Trigger(np.array(sas1_1), AS_cut1)
            L_E3_1.append(rate3_1)
            L_E4_1.append(rate4_1)
            a = ((len(E3_1) - 1) * E3_1[-1] + rate3_1) / len(E3_1)
            E3_1.append(a)
            b = ((len(E4_1) - 1) * E4_1[-1] + rate4_1) / len(E4_1)
            E4_1.append(b)

            # ttbar dim=4
            rate3_4 = Sing_Trigger(np.array(sas1_4), fixed_AS_cut4)
            rate4_4 = Sing_Trigger(np.array(sas1_4), AS_cut4)
            L_E3_4.append(rate3_4)
            L_E4_4.append(rate4_4)
            a = ((len(E3_4) - 1) * E3_4[-1] + rate3_4) / len(E3_4)
            E3_4.append(a)
            b = ((len(E4_4) - 1) * E4_4[-1] + rate4_4) / len(E4_4)
            E4_4.append(b)

            # AA dim=1
            rate3_1 = Sing_Trigger(np.array(sas2_1), fixed_AS_cut1)
            rate4_1 = Sing_Trigger(np.array(sas2_1), AS_cut1)
            L_E5_1.append(rate3_1)
            L_E6_1.append(rate4_1)
            a = ((len(E5_1) - 1) * E5_1[-1] + rate3_1) / len(E5_1)
            E5_1.append(a)
            b = ((len(E6_1) - 1) * E6_1[-1] + rate4_1) / len(E6_1)
            E6_1.append(b)

            # AA dim=4
            rate3_4 = Sing_Trigger(np.array(sas2_4), fixed_AS_cut4)
            rate4_4 = Sing_Trigger(np.array(sas2_4), AS_cut4)
            L_E5_4.append(rate3_4)
            L_E6_4.append(rate4_4)
            a = ((len(E5_4) - 1) * E5_4[-1] + rate3_4) / len(E5_4)
            E5_4.append(a)
            b = ((len(E6_4) - 1) * E6_4[-1] + rate4_4) / len(E6_4)
            E6_4.append(b)

            # update AS cuts
            AS_cut1, pre_r2_1 = PD_controller2(E2_1[-1], pre_r2_1, AS_cut1)
            AS_cut4, pre_r2_4 = PD_controller2(E2_4[-1], pre_r2_4, AS_cut4)

    # ----- convert to arrays + scale to kHz where needed -----
    E1_1 = np.array(E1_1) * 400
    E1_4 = np.array(E1_4) * 400
    E2_1 = np.array(E2_1) * 400
    E2_4 = np.array(E2_4) * 400

    R1 = np.array(R1) * 400
    R2 = np.array(R2) * 400

    # drop the initial 0 sentinel
    R3 = np.array(R3[1:])
    R4 = np.array(R4[1:])
    R5 = np.array(R5[1:])
    R6 = np.array(R6[1:])

    E3_1 = np.array(E3_1[1:])
    E4_1 = np.array(E4_1[1:])
    E3_4 = np.array(E3_4[1:])
    E4_4 = np.array(E4_4[1:])
    E5_1 = np.array(E5_1[1:])
    E6_1 = np.array(E6_1[1:])
    E5_4 = np.array(E5_4[1:])
    E6_4 = np.array(E6_4[1:])

    L_R3  = np.array(L_R3)
    L_R4  = np.array(L_R4)
    L_R5  = np.array(L_R5)
    L_R6  = np.array(L_R6)
    L_E3_1 = np.array(L_E3_1)
    L_E4_1 = np.array(L_E4_1)
    L_E3_4 = np.array(L_E3_4)
    L_E4_4 = np.array(L_E4_4)
    L_E5_1 = np.array(L_E5_1)
    L_E6_1 = np.array(L_E6_1)
    L_E5_4 = np.array(L_E5_4)
    L_E6_4 = np.array(L_E6_4)

    # ----- save everything to npz -----
    np.savez(
        OUT_NPZ,
        R1=R1, R2=R2,
        R3=R3, R4=R4, R5=R5, R6=R6,
        L_R3=L_R3, L_R4=L_R4, L_R5=L_R5, L_R6=L_R6,
        E1_1=E1_1, E1_4=E1_4, E2_1=E2_1, E2_4=E2_4,
        E3_1=E3_1, E3_4=E3_4, E4_1=E4_1, E4_4=E4_4,
        E5_1=E5_1, E5_4=E5_4, E6_1=E6_1, E6_4=E6_4,
        L_E3_1=L_E3_1, L_E4_1=L_E4_1,
        L_E3_4=L_E3_4, L_E4_4=L_E4_4,
        L_E5_1=L_E5_1, L_E6_1=L_E6_1,
        L_E5_4=L_E5_4, L_E6_4=L_E6_4,
        fixed_Ht_cut=fixed_Ht_cut,
        fixed_AS_cut1=fixed_AS_cut1,
        fixed_AS_cut4=fixed_AS_cut4
    )
    print(f"Saved results to {OUT_NPZ}")

if __name__ == "__main__":
    main()
