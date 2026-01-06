# derived_info/build_trigger_food.py
from __future__ import annotations
import argparse
import numpy as np
from typing import Dict, Any, Tuple
import h5py  # for inline NPV pairing I/O
from .preprocess import process_h5_file
from .scoring import load_autoencoders, calculate_batch_loss, calculate_H_met, count_njets
from .data_io import write_trigger_food

def run_pipeline(
    mb2_path: str,
    htoaa_path: str,
    tt_path: str,
    ae01_path: str,
    ae04_path: str,
    out_path: str,
) -> None:
    # Load models
    ae01, ae04 = load_autoencoders((ae01_path, ae04_path))

    # Load and preprocess datasets
    mc_bkg_jets, mc_bkg_ht = process_h5_file(mb2_path)
    mc_aa_jets,  mc_aa_ht  = process_h5_file(htoaa_path)
    mc_tt_jets,  mc_tt_ht  = process_h5_file(tt_path)

    # Derived features
    mc_bkg_Hmets = calculate_H_met(mc_bkg_jets, mc_bkg_ht)
    mc_aa_Hmets  = calculate_H_met(mc_aa_jets,  mc_aa_ht)
    mc_tt_Hmets  = calculate_H_met(mc_tt_jets,  mc_tt_ht)

    # NPV: stored in 4th column of first jet (identical across jets per event)
    mc_bkg_npvs = mc_bkg_jets[:, 0, 3]
    mc_aa_npvs  = mc_aa_jets[:,  0, 3]
    mc_tt_npvs  = mc_tt_jets[:,  0, 3]

    # Keys for bookkeeping
    aa_key = np.ones_like(mc_aa_npvs, dtype=np.int32) * 1
    tt_key = np.ones_like(mc_tt_npvs, dtype=np.int32) * 2

    # Scores
    mc_bkg_scores01 = calculate_batch_loss(ae01, mc_bkg_jets)
    mc_aa_scores01  = calculate_batch_loss(ae01, mc_aa_jets)
    mc_tt_scores01  = calculate_batch_loss(ae01, mc_tt_jets)

    mc_bkg_scores04 = calculate_batch_loss(ae04, mc_bkg_jets)
    mc_aa_scores04  = calculate_batch_loss(ae04, mc_aa_jets)
    mc_tt_scores04  = calculate_batch_loss(ae04, mc_tt_jets)

    # Counts
    mc_bkg_njets = count_njets(mc_bkg_jets)
    mc_aa_njets  = count_njets(mc_aa_jets)
    mc_tt_njets  = count_njets(mc_tt_jets)

    # Pack & write
    arrays: Dict[str, Any] = {
        "mc_bkg_ht":      mc_bkg_ht.astype(np.float32),
        "mc_bkg_Hmets":   mc_bkg_Hmets.astype(np.float32),
        "mc_bkg_score01": mc_bkg_scores01.astype(np.float32),
        "mc_bkg_score04": mc_bkg_scores04.astype(np.float32),
        "mc_bkg_Npv":     mc_bkg_npvs.astype(np.float32),
        "mc_bkg_njets":   mc_bkg_njets.astype(np.float32),

        "mc_aa_ht":       mc_aa_ht.astype(np.float32),
        "mc_aa_Hmets":    mc_aa_Hmets.astype(np.float32),
        "mc_aa_score01":  mc_aa_scores01.astype(np.float32),
        "mc_aa_score04":  mc_aa_scores04.astype(np.float32),
        "aa_Npv":         mc_aa_npvs.astype(np.float32),
        "aa_key":         aa_key.astype(np.int32),
        "mc_aa_njets":    mc_aa_njets.astype(np.float32),

        "mc_tt_ht":       mc_tt_ht.astype(np.float32),
        "mc_tt_Hmets":    mc_tt_Hmets.astype(np.float32),
        "mc_tt_score01":  mc_tt_scores01.astype(np.float32),
        "mc_tt_score04":  mc_tt_scores04.astype(np.float32),
        "tt_Npv":         mc_tt_npvs.astype(np.float32),
        "tt_key":         tt_key.astype(np.int32),
        "mc_tt_njets":    mc_tt_njets.astype(np.float32),
    }
    write_trigger_food(out_path, arrays)
# -------------------------- REALDATA PAIRING HELPERS -------------------------


def _read_h5_for_pairing(file_path: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Equivalent to read_h5_data() from Data_pairing_NPV.py.

    Reads Trigger_food_Data.h5 and returns all arrays needed for NPV pairing.
    """
    with h5py.File(file_path, "r") as h5_file:
        # Background
        data_ht       = h5_file["mc_bkg_ht"][:]
        data_scores01 = h5_file["mc_bkg_score01"][:]
        data_scores04 = h5_file["mc_bkg_score04"][:]
        data_npvs     = h5_file["mc_bkg_Npv"][:]
        data_njets    = h5_file["mc_bkg_njets"][:]

        # TTbar
        tt_ht         = h5_file["mc_tt_ht"][:]
        tt_scores01   = h5_file["mc_tt_score01"][:]
        tt_scores04   = h5_file["mc_tt_score04"][:]
        tt_npvs       = h5_file["tt_Npv"][:]
        tt_njets      = h5_file["mc_tt_njets"][:]

        # H→AA→4b
        aa_ht         = h5_file["mc_aa_ht"][:]
        aa_scores01   = h5_file["mc_aa_score01"][:]
        aa_scores04   = h5_file["mc_aa_score04"][:]
        aa_npvs       = h5_file["aa_Npv"][:]
        aa_njets      = h5_file["mc_aa_njets"][:]

    return (
        data_ht, data_scores01, data_scores04, data_npvs, data_njets,
        tt_ht, tt_scores01, tt_scores04, tt_npvs, tt_njets,
        aa_ht, aa_scores01, aa_scores04, aa_npvs, aa_njets,
    )


def _match_to_data(
    data_npvs: np.ndarray,
    sig_ht: np.ndarray,
    sig_scores: np.ndarray,
    sig_npvs: np.ndarray,
    sig_njets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Equivalent to match_to_data() from Data_pairing_NPV.py.

    For each NPV value in the *background* distribution, pick a signal event
    with a nearby NPV, and copy over (ht, score, npv, njets).
    """
    sorted_indices = np.argsort(sig_npvs)
    sig_ht_sorted     = sig_ht[sorted_indices]
    sig_scores_sorted = sig_scores[sorted_indices]
    sig_npvs_sorted   = sig_npvs[sorted_indices]
    sig_njets_sorted  = sig_njets[sorted_indices]

    matched_ht = []
    matched_scores = []
    matched_npvs = []
    matched_njets = []

    for npv in data_npvs:
        idx_left = np.searchsorted(sig_npvs_sorted, npv, side="left")
        idx_right = np.searchsorted(sig_npvs_sorted, npv, side="right")

        if idx_left == len(sig_npvs_sorted):
            idx = len(sig_npvs_sorted) - 1
        elif idx_left == idx_right:
            idx = idx_left
        else:
            idx = np.random.randint(idx_left, idx_right)

        matched_ht.append(sig_ht_sorted[idx])
        matched_scores.append(sig_scores_sorted[idx])
        matched_npvs.append(sig_npvs_sorted[idx])
        matched_njets.append(sig_njets_sorted[idx])

    return (
        np.array(matched_ht),
        np.array(matched_scores),
        np.array(matched_npvs),
        np.array(matched_njets),
    )


def run_pairing_npv(input_file: str, output_file: str) -> None:
    """
    RealData-only step: replicate Data_pairing_NPV.py.

    Reads Trigger_food_Data.h5 (input_file) and writes
    Matched_data_2016_with04.h5 (or user-specified output_file).
    """
    (
        data_ht, data_scores01, data_scores04, data_npvs, data_njets,
        tt_ht, tt_scores01, tt_scores04, tt_npvs, tt_njets,
        aa_ht, aa_scores01, aa_scores04, aa_npvs, aa_njets,
    ) = _read_h5_for_pairing(input_file)

    # TTbar
    matched_tt_ht, matched_tt_scores01, matched_tt_npvs, matched_tt_njets = _match_to_data(
        data_npvs, tt_ht, tt_scores01, tt_npvs, tt_njets
    )
    _, matched_tt_scores04, _, _ = _match_to_data(
        data_npvs, tt_ht, tt_scores04, tt_npvs, tt_njets
    )

    # H→AA→4b
    matched_aa_ht, matched_aa_scores01, matched_aa_npvs, matched_aa_njets = _match_to_data(
        data_npvs, aa_ht, aa_scores01, aa_npvs, aa_njets
    )
    _, matched_aa_scores04, _, _ = _match_to_data(
        data_npvs, aa_ht, aa_scores04, aa_npvs, aa_njets
    )

    # Save output
    with h5py.File(output_file, "w") as h5_out:
        # Background
        h5_out.create_dataset("data_ht",        data=data_ht)
        h5_out.create_dataset("data_scores01",  data=data_scores01)
        h5_out.create_dataset("data_scores04",  data=data_scores04)
        h5_out.create_dataset("data_Npv",       data=data_npvs)
        h5_out.create_dataset("data_njets",     data=data_njets)

        # TTbar
        h5_out.create_dataset("matched_tt_ht",        data=matched_tt_ht)
        h5_out.create_dataset("matched_tt_scores01",  data=matched_tt_scores01)
        h5_out.create_dataset("matched_tt_scores04",  data=matched_tt_scores04)
        h5_out.create_dataset("matched_tt_npvs",      data=matched_tt_npvs)
        h5_out.create_dataset("matched_tt_njets",     data=matched_tt_njets)

        # H→AA→4b
        h5_out.create_dataset("matched_aa_ht",        data=matched_aa_ht)
        h5_out.create_dataset("matched_aa_scores01",  data=matched_aa_scores01)
        h5_out.create_dataset("matched_aa_scores04",  data=matched_aa_scores04)
        h5_out.create_dataset("matched_aa_npvs",      data=matched_aa_npvs)
        h5_out.create_dataset("matched_aa_njets",     data=matched_aa_njets)

    print(f"[RealData] Pairing completed and saved to: {output_file}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Trigger_food HDF5 from CMS jets + AE scores")
    # Inputs
    p.add_argument("--minbias2", default="Data/MinBias_2.h5", help="MC background file")
    p.add_argument("--htoaa",    default="Data/HToAATo4B.h5",  help="BSM signal file")
    p.add_argument("--tt",       default="Data/TT_1.h5",       help="SM ttbar signal file")
    # Models
    p.add_argument("--ae01", default="SampleProcessing/models/autoencoder_model0_1.keras")
    p.add_argument("--ae04", default="SampleProcessing/models/autoencoder_model0_4.keras")
    p.add_argument("--control", default="MC", choices = ["MC", "RealData"], help="Control sample type: MC or RealData")
    # Output
    p.add_argument("--out", default="Data/Trigger_food_MC.h5", help="Output HDF5 path") #building Data/trigger_food_MC
    p.add_argument(
        "--out-paired",
        default="Data/Matched_data_2016_with04.h5",
        help="Output HDF5 path for NPV-paired RealData (used only if control=RealData).",
    )
    return p

def main():
    args = build_argparser().parse_args()
    # ----- Model paths + output path handling -----
    if args.control == "RealData":
        # Swap to RealData-trained AEs if user kept MC defaults
        if args.ae01 == "SampleProcessing/models/autoencoder_model0_1_realdata.keras":
            ae01 = "Data/python/autoencoder_model0_1_realdata.keras"
        else:
            ae01 = args.ae01

        if args.ae04 == "SampleProcessing/models/autoencoder_model0_4_realdata.keras":
            ae04 = "Data/python/autoencoder_model0_4_realdata.keras"
        else:
            ae04 = args.ae04

        # Default Trigger_food output name for RealData
        if args.control == "RealData":
            out = "Data/Trigger_food_Data.h5"
        else:
            out = "Data/Trigger_food_MC.h5"

        # Also prepare the paired output path
        out_paired = args.out_paired
    else:
        # MC: behaviour unchanged
        ae01 = args.ae01
        ae04 = args.ae04
        out = args.out
        out_paired = None  # not used
    run_pipeline(
        mb2_path=args.minbias2,
        htoaa_path=args.htoaa,
        tt_path=args.tt,
        ae01_path=ae01,
        ae04_path=ae04,
        out_path=out,
    )
    # ----- Extra RealData: NPV pairing -----
    if args.control == "RealData":
        # mimics running Data_pairing_NPV.py after Data_Derived_info.py
        run_pairing_npv(input_file=out, output_file=out_paired)


if __name__ == "__main__":
    main()
