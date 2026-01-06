# derived_info/build_trigger_food_data.py
from __future__ import annotations
import argparse
import numpy as np
from typing import Dict, Any

# Reuse the same modular helpers you already built
from .preprocess import process_h5_file
from .scoring import (
    load_autoencoders,
    calculate_batch_loss,
    calculate_H_met,
    count_njets,
)
from .data_io import write_trigger_food


def run_pipeline(
    data_path: str,
    ae01_path: str,
    ae04_path: str,
    out_path: str,
) -> None:
    """
    Build Trigger_food-style HDF5 for *real data* only.
    """

    # 1) Load AE models
    ae01, ae04 = load_autoencoders((ae01_path, ae04_path))

    # 2) Load and preprocess real data jets
    #    process_h5_file should already do sorting, masking, HT computation, npv>0 cut, etc.
    data_jets, data_ht = process_h5_file(data_path)

    # 3) Derived quantities
    #    a) H_met computed from (jets, ht)
    data_Hmets = calculate_H_met(data_jets, data_ht)

    #    b) Npv from 4th feature of first jet (identical across jets per event)
    data_npvs = data_jets[:, 0, 3]

    #    c) AE scores
    data_scores01 = calculate_batch_loss(ae01, data_jets)
    data_scores04 = calculate_batch_loss(ae04, data_jets)

    #    d) Number of jets retained per event
    data_njets = count_njets(data_jets)

    # 4) Pack arrays in the same style as Trigger_food_MC,
    #    but using "data_*" prefixes instead of "mc_*".
    arrays: Dict[str, Any] = {
        "data_ht":      data_ht.astype(np.float32),
        "data_Hmets":   data_Hmets.astype(np.float32),
        "data_score01": data_scores01.astype(np.float32),
        "data_score04": data_scores04.astype(np.float32),
        "data_Npv":     data_npvs.astype(np.float32),
        "data_njets":   data_njets.astype(np.float32),
        # Optionally add a bookkeeping key, e.g. all 3's:
        # "data_key": np.full_like(data_npvs, 3, dtype=np.int32),
    }

    # 5) Write out HDF5
    write_trigger_food(out_path, arrays)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build Trigger_food HDF5 for CMS DATA + AE scores"
    )
    # Real data input
    p.add_argument(
        "--data",
        default="Data/data_sortedByEvtNo.h5",
        help="Real data jets file",
    )
    # Models
    p.add_argument(
        "--ae01",
        default="SampleProcessing/models/autoencoder_model0_1.keras",
        help="AE model (1-dim latent)",
    )
    p.add_argument(
        "--ae04",
        default="SampleProcessing/models/autoencoder_model0_4.keras",
        help="AE model (4-dim latent)",
    )
    # Output
    p.add_argument(
        "--out",
        default="Data/Trigger_food_Data.h5",
        help="Output HDF5 path for real data",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_pipeline(
        data_path=args.data,
        ae01_path=args.ae01,
        ae04_path=args.ae04,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
