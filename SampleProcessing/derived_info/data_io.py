# derived_info/data_io.py
from __future__ import annotations
import os
from typing import Dict, Any
import h5py
import hdf5plugin  # noqa: F401  (enables common HDF5 compressions)

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
    print(f"[OK] Results saved to {out_path}")