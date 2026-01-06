# derived_info/data_io.py
from __future__ import annotations
import os
import numpy as np
from typing import Dict, Any, Tuple
import h5py
import hdf5plugin  # noqa: F401  (enables common HDF5 compressions)

JET_FEATURES = ("Eta", "Phi", "Pt")
N_JETS = 8
N_FEATURES = 4  # Eta, Phi, Pt, NPV (fill NPV into [:,:,3])

def read_columns(h5_path: str, keys: list[str]) -> Dict[str, Any]:
    """Read a list of datasets from an HDF5 file into dict[key]=np.ndarray."""
    out: Dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        for k in keys:
            if k not in f:
                raise KeyError(f"{h5_path} missing dataset '{k}'")
            out[k] = f[k][:]
    return out

def write_trigger_food(
    out_path: str,
    arrays: Dict[str, Any],
) -> None:
    """Write all numpy arrays into a single HDF5 (1-level, name->dataset)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=arr)
    #print(f"[OK] Results saved to {out_path}")
    




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
