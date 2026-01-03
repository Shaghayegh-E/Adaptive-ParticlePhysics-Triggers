# derived_info/build_trigger_food_v2.py
# h5 returns (jest, ht, npv)
# X = [flatten(jets(8×3))=24, npv=1]

from __future__ import annotations 
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import h5py

# use the updated readers from ae/data.py
# (adjust import if your package layout differs)
from ..ae.data import process_h5_file_MC, process_h5_file_Data
#only read in load_autoencoder for dim=2 model
from .scoring import load_autoencoder, count_njets
from .data_io import write_trigger_food


# -------------------------
# Helpers
# -------------------------

def ensure_parent_dir(path_like: str) -> None:
    p = Path(path_like)
    if p.parent and str(p.parent) != "":
        p.parent.mkdir(parents=True, exist_ok=True)


def jets_npv_to_X(jets: np.ndarray, npv: np.ndarray) -> np.ndarray:
    """
    jets: (N, 8, 3) with [eta, phi, pt]
    npv:  (N,) or (N,1)

    returns X: (N, 25) = flatten(jets)->24 + npv->1
    """
    jets = np.asarray(jets, dtype=np.float32)
    npv = np.asarray(npv, dtype=np.float32)
    jets_flat = jets.reshape(jets.shape[0], -1)  # (N, 24)
    if npv.ndim == 1:
        npv = npv.reshape(-1, 1)
    return np.concatenate([jets_flat, npv], axis=1).astype(np.float32)


def ae_mse_scores(ae, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """
    Score AE with per-event MSE on X (N,25).
    Works for your ReLU AE trained as reconstruct(X)->X.
    """
    X = np.asarray(X, dtype=np.float32)
    Xhat = ae.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - Xhat) ** 2, axis=1).astype(np.float32)


def load_sample(bkgType: str, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns jets, ht, npv.
    Background uses RealData reader only if bkgType=RealData.
    AA/TT are always MC 
    """
    if bkgType == "RealData":
        jets, ht, npv = process_h5_file_Data(path)
    else:
        jets, ht, npv = process_h5_file_MC(path)
    return jets, ht, npv.flatten()


def plot_anomaly_score_distribution(
    h5_path: str,
    bkgType: str,
    ae_dim: int,
    out_dir: str | None = None,
    cut_quantile: float = 99.75,
    max_points: int = 300_000,
    bins: int = 90,
    show: bool = True,
) -> None:
    """
    Plot anomaly score (AE MSE) distributions for bkg vs tt vs aa
    from the Trigger_food_*.h5 written by this script.

    - Saves: anomaly_score_dist_raw_*.pdf/png and anomaly_score_dist_log10_*.pdf/png
    - Also shows interactively if show=True (may no-op on headless nodes).
    """
    score_key = f"score{ae_dim:02d}"  # e.g. score02


    k_bkg = f"bkg_{score_key}"
    k_tt  = f"tt_{score_key}"
    k_aa  = f"aa_{score_key}"


    def _read_downsample(dset, max_points: int):
        n = dset.shape[0]
        if max_points is None or max_points <= 0 or n <= max_points:
            return np.asarray(dset[:], dtype=np.float32)
        stride = int(np.ceil(n / max_points))
        return np.asarray(dset[::stride], dtype=np.float32)

    h5_path = str(h5_path)
    out_dir = out_dir or str(Path(h5_path).parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        for k in (k_bkg, k_tt, k_aa):
            if k not in h5:
                raise KeyError(f"Missing dataset '{k}' in {h5_path}. Keys: {list(h5.keys())}")

        bkg = _read_downsample(h5[k_bkg], max_points)
        tt  = _read_downsample(h5[k_tt],  max_points)
        aa  = _read_downsample(h5[k_aa],  max_points)

    tag = f"{Path(h5_path).stem}_{bkgType}_dim{ae_dim:02d}"

    # background cut
    cut = float(np.percentile(bkg, cut_quantile))
    pass_b = 100.0 * float(np.mean(bkg > cut))
    eff_tt = 100.0 * float(np.mean(tt  > cut))
    eff_aa = 100.0 * float(np.mean(aa  > cut))

    
def add_cms_header(fig, left_x=0.13, right_x=0.90, y=0.95):
    """
    Add 'CMS Open Data' on the left and 'Run 283876' on the right
    in figure coordinates.
    """
    fig.text(
        left_x, y, "CMS Open Data",
        ha="left", va="top",
        fontweight="bold", fontsize=24
    )
    fig.text(
        right_x, y, "Run 283876",
        ha="right", va="top",
        fontsize=24
    )



# -------------------------
# Main pipeline
# -------------------------

def _match_to_data(data_npvs, sig_ht, sig_score, sig_npvs, sig_njets):
        order = np.argsort(sig_npvs)
        sig_ht = sig_ht[order]
        sig_score = sig_score[order]
        sig_npvs = sig_npvs[order]
        sig_njets = sig_njets[order]

        m_ht, m_sc, m_npv, m_nj = [], [], [], []
        for npv in data_npvs:
            L = np.searchsorted(sig_npvs, npv, side="left")
            R = np.searchsorted(sig_npvs, npv, side="right")
            if L >= len(sig_npvs):
                idx = len(sig_npvs) - 1
            elif L == R:
                idx = L
            else:
                idx = np.random.randint(L, R)
            m_ht.append(sig_ht[idx]); m_sc.append(sig_score[idx])
            m_npv.append(sig_npvs[idx]); m_nj.append(sig_njets[idx])
        return np.array(m_ht), np.array(m_sc), np.array(m_npv), np.array(m_nj)


def run_pipeline(
    bkgType: str,
    bkg_path: str,
    aa_path: str,
    tt_path: str,
    ae_dim: int,
    ae_path: str,
    out_path: str,
) -> None:
    # Load model (load_autoencoders returns tuple; we pass 1 path for dim=2
    ae = load_autoencoder(ae_path)

    # Load and preprocess datasets (jets, ht, npv)
    bkg_jets, bkg_ht, bkg_npv = load_sample(bkgType, bkg_path)
    aa_jets,  aa_ht,  aa_npv  = load_sample(bkgType, aa_path)
    tt_jets,  tt_ht,  tt_npv  = load_sample(bkgType, tt_path)

    # # Derived features (should still work if calculate_H_met/count_njets assume pt is index 2)
    # bkg_Hmets = calculate_H_met(bkg_jets, bkg_ht)
    # aa_Hmets  = calculate_H_met(aa_jets,  aa_ht)
    # tt_Hmets  = calculate_H_met(tt_jets,  tt_ht)

    bkg_njets = count_njets(bkg_jets)
    aa_njets  = count_njets(aa_jets)
    tt_njets  = count_njets(tt_jets)

    # Keys for bookkeeping
    #aa_key = np.ones_like(aa_npv, dtype=np.int32) * 1
    #tt_key = np.ones_like(tt_npv, dtype=np.int32) * 2

    # AE scores on X = [jets_flat, npv]
    X_bkg = jets_npv_to_X(bkg_jets, bkg_npv)
    X_aa  = jets_npv_to_X(aa_jets,  aa_npv)
    X_tt  = jets_npv_to_X(tt_jets,  tt_npv)

    bkg_score = ae_mse_scores(ae, X_bkg)
    aa_score  = ae_mse_scores(ae, X_aa)
    tt_score  = ae_mse_scores(ae, X_tt)

    
    score_key = f"score{ae_dim:02d}"  # e.g. score02

    
    arrays: Dict[str, Any] = {
        "bkg_ht":        np.asarray(bkg_ht, dtype=np.float32),
        f"bkg_{score_key}": bkg_score,
        "bkg_Npv":       np.asarray(bkg_npv, dtype=np.float32),
        "bkg_njet":       np.asarray(bkg_njets, dtype=np.float32),

        "aa_ht":         np.asarray(aa_ht, dtype=np.float32),
        f"aa_{score_key}": aa_score,
        "aa_Npv":           np.asarray(aa_npv, dtype=np.float32),
        "aa_njet":       np.asarray(aa_njets, dtype=np.float32),

        "tt_ht":         np.asarray(tt_ht, dtype=np.float32),
        f"tt_{score_key}": tt_score,
        "tt_Npv":           np.asarray(tt_npv, dtype=np.float32),
        "tt_njet":       np.asarray(tt_njets, dtype=np.float32),
    }
    
    if bkgType == "RealData":
        arrays["tt_ht"], arrays[f"tt_{score_key}"], arrays["tt_Npv"], arrays["tt_njet"] = _match_to_data(
        arrays["bkg_Npv"], arrays["tt_ht"], arrays[f"tt_{score_key}"], arrays["tt_Npv"], arrays["tt_njet"]
        )
        
        arrays["aa_ht"], arrays[f"aa_{score_key}"], arrays["aa_Npv"], arrays["aa_njet"] = _match_to_data(
        arrays["bkg_Npv"], arrays["aa_ht"], arrays[f"aa_{score_key}"], arrays["aa_Npv"], arrays["aa_njet"]
        )
    

    

    ensure_parent_dir(out_path)
    write_trigger_food(out_path, arrays)
    print(f"[OK] Wrote Trigger_food to {out_path} (AE dim={ae_dim})")
    # --- plot anomaly score distributions for the above file ---
    plot_anomaly_score_distribution(
        h5_path=out_path,
        bkgType=bkgType,
        ae_dim=ae_dim,
        out_dir=str(Path(out_path).parent),  # save next to the H5
        cut_quantile=99.75,
        max_points=300_000,   # downsample if huge
        bins=90,
        show=True,            # set False on headless machines
    )


# -------------------------
# Pairing (update to new score key)
# -------------------------


# -------------------------
# CLI
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Trigger_food HDF5 using new readers + ReLU AE on X=(jets_flat,npv)")

    p.add_argument("--bkgType", default="MC", choices=["MC", "RealData"])

    # Inputs
    p.add_argument("--MCBkg", default="Data/MinBias_2.h5", help="MC background (if bkgType=MC)")
    p.add_argument("--dataBkg",    default="Data/data_Run_2016_283408_longest.h5", help="Real background (if bkgType=RealData)")
    p.add_argument("--BSMSig",   default="Data/HToAATo4B.h5")
    p.add_argument("--SMSig",      default="Data/TT_1.h5")

    # AE (default dim=2)
    p.add_argument("--ae-dim", type=int, default=2)

    # Outputs
    p.add_argument("--out", default="Data/Trigger_food_MC.h5")
    p.add_argument("--ae_path", default="SampleProcessing/models/autoencoder_model_mc_2.keras")
    p.add_argument("--force_ae_path", default=False)
    #p.add_argument("--out-paired", default="Data/Matched_data_2016.h5")

    return p


def main():
    args = build_argparser().parse_args()

    # Choose background file
    bkg_path = args.dataBkg if args.bkgType == "RealData" else args.MCBkg
    default_ae_path_mc="SampleProcessing/models/autoencoder_model_mc_2.keras"
    default_ae_path_data="SampleProcessing/models/autoencoder_model_realdata_2.keras"
    
    # Default output names 
    if args.out == "Data/Trigger_food_MC.h5" and args.bkgType == "RealData":
        out = "Data/Trigger_food_Data.h5"
        ae_path_string = default_ae_path_data

    else:
        out = args.out
        ae_path_string = default_ae_path_mc
        
    if args.force_ae_path :
        ae_path_string = args.ae_path

    run_pipeline(
        bkgType=args.bkgType,
        bkg_path=bkg_path,
        aa_path=args.BSMSig,
        tt_path=args.SMSig,
        ae_dim=args.ae_dim,
        ae_path=ae_path_string,
        out_path=out,
    )

    

if __name__ == "__main__":
    main()
