from __future__ import annotations
import numpy as np
import h5py
import hdf5plugin  # noqa: F401 (keeps h5py filters available)


def _sort_by_pt_desc_inplace(arr: np.ndarray) -> None:
    """Sort last axis=1 (jets) by pt descending. arr shape (N, n_jets, n_features)."""
    for i in range(arr.shape[0]):
        ele = arr[i].T
        idx = np.argsort(ele[2])[::-1]
        arr[i] = ele[:, idx].T


def process_h5_file_full(input_filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      jets: (N, 8, 4)  -> [eta(+5), phi(+pi), pt, npv] (npv duplicated across jets)
      ht:   (N,)
    """
    with h5py.File(input_filename, "r") as f:
        n_events = f["j0Eta"].shape[0]
        n_jets, n_features = 8, 4
        jets = np.zeros((n_events, n_jets, n_features), dtype=np.float32)

        for j in range(n_jets):
            jets[:, j, 0] = f[f"j{j}Eta"][:] + 5.0
            jets[:, j, 1] = f[f"j{j}Phi"][:] + np.pi
            jets[:, j, 2] = f[f"j{j}Pt"][:]

        npv = f["PV_npvsGood_smr1"][:]
        ht  = f["ht"][:]

    # filter npv != 0
    m = (npv != 0)
    jets, npv, ht = jets[m], npv[m], ht[m]

    # sort by pt desc
    _sort_by_pt_desc_inplace(jets)

    # add npv into feature-3 for every jet
    jets[:, :, 3] = npv[:, None]

    # sanitize zeros (eta/phi = -1 for zero-pt jets)
    zero_pt = (jets[:, :, 2] == 0)
    jets[:, :, 0][zero_pt] = -1
    jets[:, :, 1][zero_pt] = -1

    # drop first event (to match original script)
    jets, ht = jets[1:], ht[1:]
    return jets, ht


def process_h5_file0(input_filename, use_manual_ht=True):
    """
    Generalized version of process_h5_file0 in Data_testAE.py.

    Returns:
        jets: (N_eff, 8, 4) with
              [0] eta_shifted (+5),
              [1] phi_shifted (+pi),
              [2] pt (possibly zeroed for bad jets),
              [3] PV_npvsGood
        ht:   (N_eff,) manual or file 'ht' (depending on use_manual_ht)
    """
    with h5py.File(input_filename, "r") as h5_file:
        n_events = h5_file["j0Eta"].shape[0]
        print(f"[process_h5_file_full] {input_filename} n_events:", n_events)

        n_jets = 8
        n_features = 4  # eta, phi, pt, npv

        n_selected = n_events
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)

        # fill eta/phi/pt
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f"j{i}Eta"][:] + 5.0       # eta shifted
            data_array[:, i, 1] = h5_file[f"j{i}Phi"][:] + np.pi     # phi shifted
            data_array[:, i, 2] = h5_file[f"j{i}Pt"][:]              # pt

        # PV
        npvsGood_values = h5_file["PV_npvsGood"][:]  # same as in process_h5_file0

        # ---- HT ----
        if use_manual_ht:
            sorted_data_array = sort_obj0(data_array.copy())
            Ht_values = np.zeros(n_selected, dtype=np.float32)
            for i in range(n_selected):
                ht = 0.0
                for j in range(n_jets):
                    pt = sorted_data_array[i, j, 2]
                    eta = sorted_data_array[i, j, 0] - 5.0  # undo shift
                    if pt > 20.0 and abs(eta) < 2.5:
                        ht += pt
                    else:
                        # mask bad jets
                        sorted_data_array[i, j, 2] = 0.0
                        sorted_data_array[i, j, 0] = -1.0
                        sorted_data_array[i, j, 1] = -1.0
                Ht_values[i] = ht
        else:
            # use ht from file (if you trust it)
            Ht_values = h5_file["ht"][:]
            sorted_data_array = sort_obj0(data_array.copy())

        # ---- mask npv == 0 ----
        non_zero_mask = npvsGood_values > 0
        sorted_data_array = sorted_data_array[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        npvsGood_values = npvsGood_values[non_zero_mask]

        # put PV into last feature
        sorted_data_array[:, :, 3] = npvsGood_values[:, None]

        # if pt == 0, set eta/phi = -1
        zero_pt_mask = sorted_data_array[:, :, 2] == 0
        sorted_data_array[:, :, 0][zero_pt_mask] = -1.0
        sorted_data_array[:, :, 1][zero_pt_mask] = -1.0

        # drop first event (same as Data_testAE)
        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]

        print("[process_h5_file_full] example pt row:", sorted_data_array[0, :, 2])
        return sorted_data_array, Ht_values

def process_h5_file_full_data(input_filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for process_h5_file0 for Real Data.
    """
    
    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events) 
        n_jets = 8  
        n_features = 4  
        

        #selected_indices = list(range(0, n_events, 1000))
        #n_selected = len(selected_indices)
        n_selected = n_events
        
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)
        
        # Fill the array with the data
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:] + 5  # Eta
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:] + np.pi  # Phi
            data_array[:, i, 2] = h5_file[f'j{i}Pt'][:]   # Pt
        

        npvsGood_smr1_values = h5_file['PV_npvsGood'][:]#_smr1
        #Ht_values = h5_file['ht'][:]
        # Calcolo manuale di HT con selezioni su pt ed eta
        sorted_data_array = sort_obj0(data_array)

        Ht_values = np.zeros(n_selected)  # <- make sure this is before the for loop

        for i in range(n_selected):
            ht = 0
            for j in range(n_jets):
                pt = sorted_data_array[i, j, 2]
                eta = sorted_data_array[i, j, 0] - 5  # Undo shift
                if pt > 20 and abs(eta) < 2.5:
                    ht += pt
                else:
                    # Mask bad jets
                    sorted_data_array[i, j, 2] = 0.0
                    sorted_data_array[i, j, 0] = -1
                    sorted_data_array[i, j, 1] = -1
            Ht_values[i] = ht
  
        
        # Remove entries where npv == 0
        non_zero_mask = npvsGood_smr1_values > 0  
        sorted_data_array = sorted_data_array[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]

        
        # Add npvsGood_smr1 values to the last column (time column)
        sorted_data_array[:, :, 3] = npvsGood_smr1_values[:, np.newaxis]

        
        zero_pt_mask = sorted_data_array[:, :, 2] == 0  # Identify where pt == 0
        sorted_data_array[:, :, 0][zero_pt_mask] = -1   # Set eta to -1 where pt == 0
        sorted_data_array[:, :, 1][zero_pt_mask] = -1   # Set phi to -1 where pt == 0


        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]
        


        non_zero_ht_mask = Ht_values > 0

        # normalize the column 
        #sorted_data_array[:, :, 2][non_zero_ht_mask] /= Ht_values[non_zero_ht_mask, np.newaxis]
        print(sorted_data_array[0,:,2])

        for i in range(sorted_data_array.shape[0]):
            if(sorted_data_array[i, 0, 3] <= 0):
                print(sorted_data_array[i, 0, 3])

        return sorted_data_array, Ht_values


#### Autoencoder training ###
#process h5 file for 25 input features

def process_h5_file_Data(input_filename):
    """
    This is for new Data: 
    Reads NPV from PV_npvsGood.
    Does not read ht and computes Ht manually from jets.
    This uses non-_smr1 + recomputes HT.

    Computes Ht_values[i] by looping jets and adding pt only if:
    pt > 20 and abs(eta) < 2.5
    If a jet fails that selection, it actively masks the jet:
    pt = 0, eta = -1, phi = -1

    Outputs:
    (sorted_data_array, Ht_values, npvsGood_values)
    """
    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events) 
        n_jets = 8  
        n_features = 3  
        

        #selected_indices = list(range(0, n_events, 1000))
        #n_selected = len(selected_indices)
        n_selected = n_events
        
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)
        
        # Fill the array with the data
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:] + 5  # Eta
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:] + np.pi  # Phi
            data_array[:, i, 2] = h5_file[f'j{i}Pt'][:]   # Pt
        

        npvsGood_smr1_values = h5_file['PV_npvsGood'][:]#_smr1
        #Ht_values = h5_file['ht'][:]
        
        

        Ht_values = np.zeros(n_selected)  # <- make sure this is before the for loop

        for i in range(n_selected):
            ht = 0
            for j in range(n_jets):
                pt = data_array[i, j, 2]
                eta = data_array[i, j, 0] - 5  # Undo shift
                if pt > 20 and abs(eta) < 2.5:
                    ht += pt
                else:
                    # Mask bad jets
                    data_array[i, j, 2] = 0.0
                    data_array[i, j, 0] = -1
                    data_array[i, j, 1] = -1
            Ht_values[i] = ht
  
  
        sorted_data_array = data_array
        _sort_by_pt_desc_inplace(sorted_data_array)

        
        # Remove entries where npv == 0
        non_zero_mask = npvsGood_smr1_values > 0  
        sorted_data_array = sorted_data_array[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]

        
        # Add npvsGood_smr1 values to the last column (time column)
        #sorted_data_array[:, :, 3] = npvsGood_smr1_values[:, np.newaxis]

        
        zero_pt_mask = sorted_data_array[:, :, 2] == 0  # Identify where pt == 0
        sorted_data_array[:, :, 0][zero_pt_mask] = -1   # Set eta to -1 where pt == 0
        sorted_data_array[:, :, 1][zero_pt_mask] = -1   # Set phi to -1 where pt == 0


        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]
        npvsGood_smr1_values = npvsGood_smr1_values[1:]
        


        non_zero_ht_mask = Ht_values > 0

        # normalize the column 
        #sorted_data_array[:, :, 2][non_zero_ht_mask] /= Ht_values[non_zero_ht_mask, np.newaxis]
        print(sorted_data_array[0,:,2])

        #for i in range(sorted_data_array.shape[0]):
            #if(sorted_data_array[i, 0, 3] <= 0):
                #print(sorted_data_array[i, 0, 3])

        return sorted_data_array, Ht_values, npvsGood_smr1_values



def process_h5_file_MC(input_filename):
    """
    This is for new MC Data:
    Reads NPV from PV_npvsGood_smr1.
    Reads HT from ht.
    Takes Ht_values = h5_file['ht'][:] directly.
    Does not apply the physics selection (pt > 20, |eta| < 2.5) when forming HT.
    Only “masking” is: if pt == 0, set (eta, phi) = (-1, -1).

    OUTPUT:
    (sorted_data_array, Ht_values, npvsGood_smr1_values)
    """
    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:',n_events) 
        n_jets = 8  
        n_features = 3  
        

        #selected_indices = list(range(0, n_events, 1000))
        #n_selected = len(selected_indices)
        n_selected = n_events
        
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)
        
        # Fill the array with the data
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:] + 5  # Eta
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:] + np.pi  # Phi
            data_array[:, i, 2] = h5_file[f'j{i}Pt'][:]   # Pt
        

        npvsGood_smr1_values = h5_file['PV_npvsGood_smr1'][:]
        Ht_values = h5_file['ht'][:]
        
        # Remove entries where npv == 0
        non_zero_mask = npvsGood_smr1_values != 0  
        data_array = data_array[non_zero_mask]  
        
        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        
        
        sorted_data_array = data_array
        _sort_by_pt_desc_inplace(sorted_data_array)
        # Add npvsGood_smr1 values to the last column (time column)
        #sorted_data_array[:, :, 3] = npvsGood_smr1_values[:, np.newaxis]

        
        zero_pt_mask = sorted_data_array[:, :, 2] == 0  # Identify where pt == 0
        sorted_data_array[:, :, 0][zero_pt_mask] = -1   # Set eta to -1 where pt == 0
        sorted_data_array[:, :, 1][zero_pt_mask] = -1   # Set phi to -1 where pt == 0


        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]
        npvsGood_smr1_values = npvsGood_smr1_values[1:]

        non_zero_ht_mask = Ht_values > 0

        # normalize the column 
        #sorted_data_array[:, :, 2][non_zero_ht_mask] /= Ht_values[non_zero_ht_mask, np.newaxis]
        print(sorted_data_array[0,:,2])

        return sorted_data_array, Ht_values, npvsGood_smr1_values



def load_bkg_aa_tt(args):
    """
    Returns:
      bkg_jets, bkg_ht, bkg_npv,
      aa_jets,  aa_ht,  aa_npv,
      tt_jets,  tt_ht,  tt_npv
    "when we work with background real data we only care about jets with |eta|<2.5 and pt>20GeV"
    """
    if args.bkgType == "MC":
        bkg = process_h5_file_MC(args.MCBkg)
        aa = process_h5_file_MC(args.BSMSig)
        tt = process_h5_file_MC(args.SMSig)
    else:  # RealData
        bkg = process_h5_file_Data(args.dataBkg)
        aa = process_h5_file_Data(args.BSMSig)
        tt = process_h5_file_Data(args.SMSig)
    return bkg, aa, tt