from __future__ import annotations
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os
from .data import process_h5_file_full, split_minbias_for_training, process_h5_file_full_data
from .models import build_autoencoder_flat
from .losses import masked_mse_loss  # available if you want to switch
from ..derived_info.scoring import (
    batch_mse_scores,
    percentile_threshold,
    pass_rate_above,
)
from .plots import plot_signal_pass_vs_dim, plot_hist_pair, plot_signal_pass_vs_dim_data
from pathlib import Path
try:
    import atlas_mpl_style as aplt
    aplt.use_atlas_style()
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent        # .../SampleProcessing/ae
PKG_ROOT = THIS_DIR.parent                        # .../SampleProcessing
DEFAULT_MODEL_DIR = PKG_ROOT / "models"           # .../SampleProcessing/models

def train_one_dim(X_train, X_val, img_shape, code_dim, loss_name="mse"):
    enc, dec = build_autoencoder_flat(img_shape, code_dim)
    inp = keras.Input(img_shape)
    z = enc(inp)
    out = dec(z)
    ae = keras.Model(inp, out)
    loss = loss_name if loss_name != "masked" else masked_mse_loss
    ae.compile(optimizer="adamax", loss=loss)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=0, restore_best_weights=True)
    ae.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=100, callbacks=[es], verbose=0)
    return ae

def run_realdata_legacy(args):
    """
    EXACT RealData behaviour matching the old Data_testAE.py script.
    Uses:
      - process_h5_file_full_data (clone of process_h5_file0)
      - same stride, same train/test split
      - same AE thresholding logic
      - same 'meeting style' plot via plot_signal_pass_vs_dim_data.
    """
    # ---- 1) Load RealData background + AA + TT with legacy pre-processing ----
    mb_jets_full, mb_ht_full = process_h5_file_full_data(args.data)
    aa_jets_full, aa_ht_full = process_h5_file_full_data(args.aa)
    tt_jets_full, tt_ht_full = process_h5_file_full_data(args.tt)

    stride = args.stride

    # Background: stride (same as X1, HT1 in Data_testAE.py)
    X1  = mb_jets_full[::stride]
    HT1 = mb_ht_full[::stride]

    # AA, TT: same slices as in Data_testAE.py
    X_AA  = aa_jets_full
    HTAA  = aa_ht_full[::stride]
    X_TT  = tt_jets_full
    HTtt  = tt_ht_full[::stride]

    # Train/test split exactly like old script
    X_train, X_test, HT_train, HT_test = train_test_split(
        X1, HT1, test_size=0.5, random_state=42
    )

    img_shape = X_train.shape[1:]  # (8,4)

    # ---- 2) HT baseline like old script: percentile of *HT_test* ----
    thr_ht = percentile_threshold(HT_test, 99.75)
    AA_ht_passed = 100.0 * (HTAA > thr_ht).sum() / HTAA.size
    TT_ht_passed = 100.0 * (HTtt > thr_ht).sum() / HTtt.size
    print("AA_ht_passed:", AA_ht_passed)
    print("TT_ht_passed:", TT_ht_passed)

    # ---- 3) Autoencoder loop over dimensions (Data_testAE style) ----
    dims = args.dims
    results_dim = {}
    aa_rates, tt_rates = [], []

    for d in dims:
        print(f"\n=== [RealData legacy] Training autoencoder for dim={d} ===")
        ae = train_one_dim(X_train, X_test, img_shape, code_dim=d, loss_name=args.loss)

        # Optionally save dim=1,4 models as before
        if d in args.save_model_dims:
            os.makedirs(args.model_out_dir, exist_ok=True)
            save_path = os.path.join(args.model_out_dir, f"{args.model_prefix}{d}_realdata.keras")
            print(f"Saving autoencoder for dim={d} to {save_path}")
            ae.save(save_path)

        # Scores on test-background and signal samples
        bkg_scores = batch_mse_scores(ae, X_test)
        aa_scores  = batch_mse_scores(ae, X_AA)
        tt_scores  = batch_mse_scores(ae, X_TT)

        # 99.75th percentile of background test scores (same as percen_9975)
        thr = percentile_threshold(bkg_scores, 99.75)

        # Pass rates (100 * (#scores > thr) / N) – mirrors old script
        aa_pass = pass_rate_above(aa_scores, thr)
        tt_pass = pass_rate_above(tt_scores, thr)
        print(f"dim={d:2d}  thr={thr:.4g}  AA={aa_pass:6.2f}%  TT={tt_pass:6.2f}%")

        results_dim[d] = dict(
            bkg_scores=bkg_scores,
            AA_scores=aa_scores,
            TT_scores=tt_scores,
        )
        aa_rates.append(aa_pass)
        tt_rates.append(tt_pass)

    # ---- 4) RealData “meeting style” plot (same semantics as Data_testAE) ----
    base, _ = os.path.splitext(args.out_pass_vs_dim)
    plot_signal_pass_vs_dim_data(
        dims,
        aa_rates,
        tt_rates,
        aa_ht_eff = 21.33, #hard coded values from Data_testAE.py
        tt_ht_eff = 97.26,
        out_prefix=base,
    )

    # ---- 5) Hist pair for d=1 and d=4, like modular path ----
    d1 = dims[0] if dims else 1
    d4 = 4 if 4 in results_dim else dims[min(3, len(dims) - 1)]

    plot_hist_pair(
        d1,
        d4,
        results_dim[d1],
        results_dim[d4],
        out_pair=args.out_hist_pair,
        out_a=args.out_hist_a,
        out_b=args.out_hist_b,
    )

def main():
    ap = argparse.ArgumentParser(description="Autoencoder test experiment (modularized)")
    ap.add_argument("--minbias", default="Data/MinBias_1.h5", help="MC MinBias background file (used if --control MC).")
    ap.add_argument("--aa",       default="Data/HToAATo4B.h5")
    ap.add_argument("--tt",       default="Data/TT_1.h5")
    ap.add_argument("--stride",   type=int, default=100)
    ap.add_argument("--dims",     type=int, nargs="+", default=list(range(1,16)))
    ap.add_argument("--loss",     choices=["mse","masked"], default="mse")
    ap.add_argument("--out_pass_vs_dim", default="outputs/signal_pass_vs_dimension.pdf")
    ap.add_argument("--out_hist_pair",   default="outputs/AS_hist_comparison2016.pdf")
    ap.add_argument("--out_hist_a",      default="outputs/AS_hist_comparison2016-a.pdf")
    ap.add_argument("--out_hist_b",      default="outputs/AS_hist_comparison2016-b.pdf")
    ap.add_argument(
        "--save-model-dims",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Latent dimensions for which to save the trained autoencoder models.",
    )
    ap.add_argument(
        "--model-out-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory to save autoencoder models for selected dimensions.",
    )
    ap.add_argument(
        "--model-prefix",
        default="autoencoder_model0_",
        help="Filename prefix for saved autoencoder models.",
    )
    ap.add_argument(
        "--control", default="MC", choices=["MC","RealData"], help="Monte Carlo Simulated data or Real Data with NPV pairing."
    )
    ap.add_argument(
        "--data",
        default="Data/data_Run_2016_283876.h5",
        help="Real data background file (used if --control RealData).",
    )
    args = ap.parse_args()
    # Ensure model output directory exists
    os.makedirs(args.model_out_dir, exist_ok=True)

    if args.control == "RealData":
        print("[INFO] RealData mode: using legacy Data_testAE-like pipeline.")
        run_realdata_legacy(args)
        return
    

    print(f"[INFO] Using MC MinBias background: {args.minbias}")
    # Load
    mb_jets_full, mb_ht_full = process_h5_file_full(args.minbias)
    aa_jets_full, aa_ht_full = process_h5_file_full(args.aa)
    tt_jets_full, tt_ht_full = process_h5_file_full(args.tt)

    # Slice like original
    X_full, Jets_only, Npv, HT = split_minbias_for_training(mb_jets_full, mb_ht_full, stride=args.stride)
    X_AA, Jets_AA, _, HT_AA = split_minbias_for_training(aa_jets_full, aa_ht_full, stride=1)
    X_TT, Jets_TT, _, HT_TT = split_minbias_for_training(tt_jets_full, tt_ht_full, stride=1)

    # Train/val split (only “full input model” was used in sweep in the original loop)
    X_tr, X_te = train_test_split(X_full, test_size=0.5, random_state=42)

    img_shape = X_tr.shape[1:]  # (8,4)

    # HT baselines
    thr_ht = percentile_threshold(HT, 99.75)
    AA_ht_pass = 100.0 * (HT_AA > thr_ht).sum() / HT_AA.size
    TT_ht_pass = 100.0 * (HT_TT > thr_ht).sum() / HT_TT.size
    print("AA_ht_passed:", AA_ht_pass)
    print("TT_ht_passed:", TT_ht_pass)

    results_dim = {}
    aa_rates, tt_rates = [], []

    for d in args.dims:
        print(f"\n=== Training autoencoder for dim={d} ===")
        ae = train_one_dim(X_tr, X_te, img_shape, code_dim=d, loss_name=args.loss)

        # ---- Default: 1 and 4 for saving autoencoder models ----
        if d in (1, 4):
            os.makedirs(args.model_out_dir, exist_ok=True)
            save_path = os.path.join(args.model_out_dir, f"{args.model_prefix}{d}.keras")
            print(f"Saving autoencoder for dim={d} to {save_path}")
            ae.save(save_path)

        bkg_scores = batch_mse_scores(ae, X_te)
        aa_scores  = batch_mse_scores(ae, X_AA)
        tt_scores  = batch_mse_scores(ae, X_TT)

        thr = percentile_threshold(bkg_scores, 99.75)
        aa_pass = pass_rate_above(aa_scores, thr)
        tt_pass = pass_rate_above(tt_scores, thr)

        results_dim[d] = dict(
            bkg_scores=bkg_scores,
            AA_scores=aa_scores,
            TT_scores=tt_scores,
        )
        aa_rates.append(aa_pass)
        tt_rates.append(tt_pass)
        print(f"dim={d:2d}  thr={thr:.4g}  AA={aa_pass:6.2f}%  TT={tt_pass:6.2f}%")


    if args.control == "MC":
        # Original clean plot (0–100%, one output file)
        plot_signal_pass_vs_dim(
            args.dims,
            aa_rates,
            tt_rates,
            AA_ht_pass,
            TT_ht_pass,
            args.out_pass_vs_dim,
        )
    else:
        # RealData: meeting-style plot (pdf + png) using out_pass_vs_dim as prefix
        base, _ = os.path.splitext(args.out_pass_vs_dim)
        plot_signal_pass_vs_dim_data(
            args.dims,
            aa_rates,
            tt_rates,
            aa_ht_eff = 21.33,
            tt_ht_eff = 97.26,
            out_prefix=base,
        )
    # hist pair for d=1 and d=4 (if available)
    d1 = args.dims[0] if args.dims else 1
    d4 = 4 if 4 in results_dim else args.dims[min(3, len(args.dims) - 1)]
    plot_hist_pair(
        d1,
        d4,
        results_dim[d1],
        results_dim[d4],
        out_pair=args.out_hist_pair,
        out_a=args.out_hist_a,
        out_b=args.out_hist_b,
    )



if __name__ == "__main__":
    main()
