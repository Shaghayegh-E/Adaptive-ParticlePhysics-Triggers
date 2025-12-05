# derived_info/preprocess.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from .data_io import read_columns
import h5py
JET_FEATURES = ("Eta", "Phi", "Pt")
N_JETS = 8
N_FEATURES = 4  # Eta, Phi, Pt, NPV (we fill NPV into [:,:,3])

def _sort_descending_by_pt(events: np.ndarray) -> np.ndarray:
    """
    Sort per-event jets by descending pt (column index 2) in-place style.
    events: (n_events, n_jets, n_features)
    """
    # Vectorized arg-sort across jets
    # shape: (n_events, n_jets)
    order = np.argsort(events[:, :, 2], axis=1)[:, ::-1]
    # Gather per row using take_along_axis
    # We need to expand order to (n_events, n_jets, 1) and tile across features
    idx = np.expand_dims(order, axis=-1).repeat(events.shape[-1], axis=-1)
    return np.take_along_axis(events, idx, axis=1)

def _mask_zero_pt_as_missing(events: np.ndarray) -> None:
    """Where pt==0, set eta/phi to -1 (in-place)."""
    zero_pt_mask = (events[:, :, 2] == 0)
    events[:, :, 0][zero_pt_mask] = -1.0
    events[:, :, 1][zero_pt_mask] = -1.0

def process_h5_file(input_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load jets & auxiliaries, filter npv==0, sort by pt desc, attach NPV as 4th feature.
    Returns:
      jets: (n_events_filt-1, N_JETS, 4)  (Eta,Phi,Pt,NPV)
      Ht:   (n_events_filt-1,)
    """
    # Load required columns
    jet_keys = [f"j{i}{feat}" for i in range(N_JETS) for feat in JET_FEATURES]
    cols = read_columns(input_filename, jet_keys + ["PV_npvsGood_smr1", "ht"])

    # Build (n_events, N_JETS, 3)
    n_events = cols["j0Eta"].shape[0]
    jets = np.zeros((n_events, N_JETS, 3), dtype=np.float32)
    for i in range(N_JETS):
        jets[:, i, 0] = cols[f"j{i}Eta"][:] + 5.0     # Eta normalized shift
        jets[:, i, 1] = cols[f"j{i}Phi"][:] + np.pi   # Phi normalized shift
        jets[:, i, 2] = cols[f"j{i}Pt"][:]            # Pt

    npv = cols["PV_npvsGood_smr1"].astype(np.float32)
    ht  = cols["ht"].astype(np.float32)

    # Filter out events with NPV==0
    mask = (npv != 0)
    jets = jets[mask]
    npv  = npv[mask]
    ht   = ht[mask]

    # Append NPV as 4th feature and sort per-event by Pt desc
    out = np.zeros((jets.shape[0], N_JETS, N_FEATURES), dtype=np.float32)
    out[:, :, :3] = jets
    out[:, :, 3]  = npv[:, None]

    out = _sort_descending_by_pt(out)
    _mask_zero_pt_as_missing(out)

    # Drop first event to match original behavior
    out = np.delete(out, 0, axis=0)
    ht  = ht[1:]

    return out, ht


def process_h5_file_data(input_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    This is fo real data reading jet_features_4plots.py:
      - read j{i}Eta/Phi/Pt
      - sort jets by pt
      - recompute HT with pt>20, |eta|<2.5 and mask bad jets
      - filter events with NPV>0 (PV_npvsGood)
      - attach NPV as 4th feature
      - drop first event
    Returns:
      jets: (n_events_filt-1, N_JETS, 4)
      Ht_values: (n_events_filt-1,)
    """
    with h5py.File(input_filename, "r") as h5_file:
        n_events = h5_file["j0Eta"].shape[0]
        n_jets = N_JETS

        # (n_events, n_jets, 4): Eta, Phi, Pt, (NPV placeholder)
        jets = np.zeros((n_events, n_jets, N_FEATURES), dtype=np.float32)
        for i in range(n_jets):
            jets[:, i, 0] = h5_file[f"j{i}Eta"][:] + 5.0
            jets[:, i, 1] = h5_file[f"j{i}Phi"][:] + np.pi
            jets[:, i, 2] = h5_file[f"j{i}Pt"][:]

        # NOTE: Data uses PV_npvsGood (no "_smr1" in your plotting script)
        npv = h5_file["PV_npvsGood"][:].astype(np.float32)

    # Sort by pt descending first
    jets = _sort_descending_by_pt(jets)

    # Recompute HT with pt>20 & |eta|<2.5,
    # and mask jets that fail the selection
    Ht_values = np.zeros(jets.shape[0], dtype=np.float32)
    for i in range(jets.shape[0]):
        ht = 0.0
        for j in range(n_jets):
            pt  = jets[i, j, 2]
            eta = jets[i, j, 0] - 5.0  # undo the +5 shift
            if pt > 20.0 and abs(eta) < 2.5:
                ht += pt
            else:
                # Mask bad jets
                jets[i, j, 2] = 0.0
                jets[i, j, 0] = -1.0
                jets[i, j, 1] = -1.0
        Ht_values[i] = ht

    # Filter out events where npv <= 0 (like your plotting script)
    mask = npv > 0
    jets      = jets[mask]
    Ht_values = Ht_values[mask]
    npv       = npv[mask]

    # Attach NPV as 4th feature (same as MC format)
    jets[:, :, 3] = npv[:, None]

    # Final masking of any zero-pt jets
    _mask_zero_pt_as_missing(jets)

    # Drop first event (to match your original behavior)
    jets      = np.delete(jets, 0, axis=0)
    Ht_values = Ht_values[1:]

    return jets, Ht_values
