# singletrigger_io.py
import h5py
import numpy as np
from pathlib import Path
DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "Trigger_food_MC.h5"

def read_mc_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        Bas01_tot = h5_file['mc_bkg_score01'][:]
        Bas04_tot = h5_file['mc_bkg_score04'][:]
        Bht_tot   = h5_file['mc_bkg_ht'][:]
        B_npvs    = h5_file['mc_bkg_Npv'][:]

        Sas01_tot1 = h5_file['mc_tt_score01'][:]
        Sas04_tot1 = h5_file['mc_tt_score04'][:]
        Sht_tot1   = h5_file['mc_tt_ht'][:]
        S_npvs1    = h5_file['tt_Npv'][:]

        Sas01_tot2 = h5_file['mc_aa_score01'][:]
        Sas04_tot2 = h5_file['mc_aa_score04'][:]
        Sht_tot2   = h5_file['mc_aa_ht'][:]
        S_npvs2    = h5_file['aa_Npv'][:]

    return (
        Sas01_tot1, Sas04_tot1, Sht_tot1, S_npvs1,
        Sas01_tot2, Sas04_tot2, Sht_tot2, S_npvs2,
        Bas01_tot,  Bas04_tot,  Bht_tot,  B_npvs
    )


import h5py, numpy as np

def load_trigger_food_mc(path: str): #used for Local_Multi.py or Multi_path.py
    with h5py.File(path, "r") as f:
        D = {
            # background
            "mc_bkg_score01": f["mc_bkg_score01"][:],
            "mc_bkg_ht":      f["mc_bkg_ht"][:],
            "mc_bkg_Npv":     f["mc_bkg_Npv"][:],
            "mc_bkg_njets":   f["mc_bkg_njets"][:],
            # signal 1 (tt)
            "mc_tt_score01":  f["mc_tt_score01"][:],
            "mc_tt_ht":       f["mc_tt_ht"][:],
            "tt_Npv":         f["tt_Npv"][:],
            "mc_tt_njets":    f["mc_tt_njets"][:],
            # signal 2 (aa)
            "mc_aa_score01":  f["mc_aa_score01"][:],
            "mc_aa_ht":       f["mc_aa_ht"][:],
            "aa_Npv":         f["aa_Npv"][:],
            "mc_aa_njets":    f["mc_aa_njets"][:],
        }
    return D

def load_trigger_food_summary_plots(path: str):   #used for summary.py
    with h5py.File(path, 'r') as h5_file:
        # Read datasets for background
        Bas01_tot = h5_file['mc_bkg_score04'][:]
        #Bas04_tot = h5_file['mc_bkg_score01'][:]
        #Bas_tot = h5_file['mc_bkg_Hmets'][:]
        Bht_tot = h5_file['mc_bkg_ht'][:]
        B_npvs = h5_file['mc_bkg_Npv'][:]
        B_njets = h5_file['mc_bkg_njets'][:]
        
        # Read datasets for signal
        Sas01_tot1 = h5_file['mc_tt_score04'][:]
        #Sas04_tot1 = h5_file['mc_dihiggs_score04'][:]
        #Sas_tot1 = h5_file['mc_dijet_Hmets'][:]
        Sht_tot1 = h5_file['mc_tt_ht'][:]
        S_npvs1 = h5_file['tt_Npv'][:]
        S_njets1 = h5_file['mc_tt_njets'][:]
        #sig_key1 = h5_file['dijet_key'][:]  # Read signal keys if needed

        Sas01_tot2 = h5_file['mc_aa_score04'][:]
        #Sas04_tot2 = h5_file['mc_tt_score04'][:]
        #Sas_tot2 = h5_file['mc_ttbar_Hmets'][:]
        Sht_tot2 = h5_file['mc_aa_ht'][:]
        S_npvs2 = h5_file['aa_Npv'][:]
        S_njets2 = h5_file['mc_aa_njets'][:]
        #sig_key2 = h5_file['tt_key'][:]  # Read signal keys if needed
        
        # Read datasets for data
        #data_ht = h5_file['data_ht'][:]
        #data_score = h5_file['data_score'][:]
        #data_npv = h5_file['data_Npv'][:]
        
    return Sas01_tot1, Sht_tot1, S_npvs1, S_njets1, Sas01_tot2, Sht_tot2, S_npvs2, S_njets2, Bas01_tot, Bht_tot, B_npvs, B_njets #,data_ht, data_score, data_npv

### Real data #####
def load_trigger_food_realdata(path: str): #used for real data
        # Bas01_tot = h5_file['data_scores01'][:] #Zixin: update to scores01
        # Bht_tot = h5_file['data_ht'][:]
        # B_npvs = h5_file['data_Npv'][:]
        # B_njets = h5_file['data_njets'][:]


        # # Read datasets for signal
        # # Sas01_tot1 = h5_file['matched_tt_scores'][:] #Giovanna: original
        # Sas01_tot1 = h5_file['matched_tt_scores01'][:] #Zixin: update to scores01
        # Sht_tot1 = h5_file['matched_tt_ht'][:]
        # S_npvs1 = h5_file['matched_tt_npvs'][:]
        # S_njets1 = h5_file['matched_tt_njets'][:]

        # # Sas01_tot2 = h5_file['matched_aa_scores'][:] #Giovanna: original
        # Sas01_tot2 = h5_file['matched_aa_scores01'][:] #Zixin: update to scores01
        # Sht_tot2 = h5_file['matched_aa_ht'][:]
        # S_npvs2 = h5_file['matched_aa_npvs'][:]
        # S_njets2 = h5_file['matched_aa_njets'][:]
    with h5py.File(path, "r") as f:
        D = {
            # background
            "matched_bkg_score01": f["data_scores01"][:],
            "matched_bkg_ht":      f["data_ht"][:],
            "matched_bkg_Npv":     f["data_Npv"][:],
            "matched_bkg_njets":   f["data_njets"][:],
            # signal 1 (tt)
            "matched_tt_score01":  f["matched_tt_scores01"][:],
            "matched_tt_ht":       f["matched_tt_ht"][:],
            "matched_tt_Npv":      f["matched_tt_npvs"][:],
            "matched_tt_njets":    f["matched_tt_njets"][:],
            # signal 2 (aa)
            "matched_aa_score01":  f["matched_aa_scores01"][:],
            "matched_aa_ht":       f["matched_aa_ht"][:],
            "matched_aa_Npv":      f["matched_aa_npvs"][:],
            "matched_aa_njets":    f["matched_aa_njets"][:],
        }
    return D