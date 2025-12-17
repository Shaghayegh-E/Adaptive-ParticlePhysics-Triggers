import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def cummean(x):
    x = np.asarray(x, dtype=np.float64)
    return np.cumsum(x) / (np.arange(len(x)) + 1.0)

def rel_to_t0(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    return x / max(1e-12, float(x[0]))




# ------------------------- plotting helpers -------------------------
def add_cms_header(fig, left_x=0.13, right_x=0.90, y=0.94, run_label="Run 283408"):
    fig.text(left_x, y, "CMS Open Data",
             ha="left", va="top", fontweight="bold", fontsize=20)
    fig.text(right_x, y, run_label,
             ha="right", va="top", fontsize=20)

def save_pdf_png(fig, basepath, dpi_png=300):
    Path(basepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{basepath}.pdf", bbox_inches="tight")
    fig.savefig(f"{basepath}.png", bbox_inches="tight", dpi=dpi_png)

def plot_rate_with_tolerance(
    time,
    y_const,
    y_pd,
    y_dqn,
    outbase,
    *,
    run_label="Run 283408",
    legend_title="HT Trigger",
    ylim=(0, 200),
    ylabel="Background Rate [kHz]",
    xlabel="Time (Fraction of Run)",
    tol_upper=110.0,
    tol_lower=90.0,
    grid_alpha=0.6,
    # styles (match your paper family)
    const_style=dict(color="tab:blue", linestyle="dashed", linewidth=3.0),
    pd_style=dict(color="mediumblue", linestyle="solid", linewidth=2.5),
    dqn_style=dict(color="tab:purple", linestyle="solid", linewidth=2.5),
    const_label="Constant Menu",
    pd_label="PD Controller",
    dqn_label="DQN",
    add_cms_header=None,
    save_pdf_png=None,
    dpi_png=300,
):
    """
    Plot background rate vs time with:
      - main legend: Constant/PD/DQN under legend_title
      - reference legend: Upper/Lower tolerance lines
    outbase: path WITHOUT extension (str or Path)
    add_cms_header(fig, run_label=...) and save_pdf_png(fig, basepath, dpi_png=...)
      are passed in (so utils doesn't need to import them from your script).
    """
    time = np.asarray(time)
    y_const = np.asarray(y_const)
    y_pd = np.asarray(y_pd)
    y_dqn = np.asarray(y_dqn)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, y_const, **const_style)
    ax.plot(time, y_pd,    **pd_style)
    ax.plot(time, y_dqn,   **dqn_style)

    ax.axhline(y=tol_upper, color="gray", linestyle="--", linewidth=1.5)
    ax.axhline(y=tol_lower, color="gray", linestyle="--", linewidth=1.5)

    ax.set_xlabel(xlabel, loc="center")
    ax.set_ylabel(ylabel, loc="center")
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=grid_alpha)

    # ---- Main legend (Constant/PD/DQN) ----
    h_const = mlines.Line2D([], [], **const_style)
    h_pd    = mlines.Line2D([], [], **pd_style)
    h_dqn   = mlines.Line2D([], [], **dqn_style)

    leg_main = ax.legend(
        [h_const, h_pd, h_dqn],
        [const_label, pd_label, dqn_label],
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fontsize=14,
    )
    ax.add_artist(leg_main)

    # ---- Reference legend (tolerances) ----
    upper = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    lower = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    ax.legend(
        [upper, lower],
        [f"Upper Tolerance ({int(tol_upper)})", f"Lower Tolerance ({int(tol_lower)})"],
        title="Reference",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fontsize=14,
    )

    if add_cms_header is not None:
        # your add_cms_header(fig, run_label=...)
        add_cms_header(fig, run_label=run_label)

    if save_pdf_png is not None:
        save_pdf_png(fig, str(outbase), dpi_png=dpi_png)

    plt.close(fig)



def save_png(fig, basepath, dpi_png=300):
    """Save figure as PNG only. basepath: path without extension."""
    p = Path(basepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p.with_suffix(".png")), bbox_inches="tight", dpi=dpi_png)






#### H5 reading utility ####

# ------------------------- H5 reading (the unified reader) -------------------------
import h5py
import hdf5plugin  # noqa: F401
def _collect_datasets(h5):
    """Return dict: dataset_path -> h5py.Dataset (supports nested groups)."""
    dsets = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            dsets[name] = obj
    h5.visititems(visitor)
    return dsets

def _basename(x: str) -> str:
    return x.split("/")[-1]

def _find_key(keys, candidates):
    """
    Find a dataset key by trying:
      1) exact match
      2) basename match (for nested datasets)
    """
    for c in candidates:
        if c in keys:
            return c
    # basename match
    for c in candidates:
        for k in keys:
            if _basename(k) == c:
                return k
    return None
def _first_present(h5_keys, candidates):
    for k in candidates:
        if k in h5_keys:
            return k
    return None

def _read_score(h5, prefix: str, dim: int):
    """
    Trigger_food_MC.h5 or Trigger_food_Data.h5 or Matched_data_2016_dim2.h5 : f"{prefix}_score{dim:02d}" like mc_bkg_score02 or data_bkg_score02.
    """
    d2 = f"{int(dim):02d}"
    for k in (f"{prefix}_score{d2}", f"{prefix}_scores{d2}"):
        if k in h5:
            return h5[k][:]
    return None

def print_h5_tree(path: str, max_items: int | None = None) -> None:
    """
    Print ALL keys in an HDF5 file, including nested groups/datasets.
    Shows dataset shape + dtype. Use max_items to truncate.
    """
    print(f"\n[H5] Inspect: {path}")
    n = 0

    def visitor(name, obj):
        nonlocal n
        if max_items is not None and n >= max_items:
            return
        if isinstance(obj, h5py.Group):
            print(f"  [G] {name}/")
        elif isinstance(obj, h5py.Dataset):
            print(f"  [D] {name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"  [?] {name}  type={type(obj)}")
        n += 1

    with h5py.File(path, "r") as h5:
        # root keys (top-level)
        print("  Top-level:", list(h5.keys()))
        h5.visititems(visitor)

    if max_items is not None:
        print(f"  ... printed up to max_items={max_items}")
    print("")


def read_any_h5(path: str, score_dim_hint: int = 2):
    """
    Unified outputs (same keys as your DQN code expects):
      Bht, Bnpv, Bas2, #background Ht, Npv, anomaly score with dim 2
      Tht, Tnpv, Tas2, #ttbar Ht, Npv, anomaly score with dim 2
      Aht, Anpv, Aas2, #aa Ht, Npv, anomaly score with dim 2
      meta['matched_by_index']

    Supported input files:
      - Trigger_food_MC.h5          (MC control) Background Data/MinBias_2.h5 + aa Data/HToAATo4B.h5 + ttbar Data/TT_1.h5
      - Trigger_food_Data.h5        (RealData control, unpaired) Background Data/data_Run_2016_283408_longest.h5
      - Matched_data_2016.h5        (RealData paired) Matched_data_2016_dim2.h5
    """
    with h5py.File(path, "r") as h5:
        keys = set(h5.keys())
        hint = int(score_dim_hint)

        # ------------------------------------------------------------
        # Case A) MC Trigger_food (control="MC")
        # ------------------------------------------------------------
        if ("mc_bkg_ht" in keys) and ("mc_bkg_Npv" in keys):
            Bht  = h5["mc_bkg_ht"][:]
            Bnpv = h5["mc_bkg_Npv"][:]

            Tht  = h5["mc_tt_ht"][:]
            Aht  = h5["mc_aa_ht"][:]

            # tt_Npv / aa_Npv (not mc_tt_Npv / mc_aa_Npv)
            Tnpv = h5["tt_Npv"][:] if "tt_Npv" in keys else np.zeros_like(Tht, dtype=np.float32)
            Anpv = h5["aa_Npv"][:] if "aa_Npv" in keys else np.zeros_like(Aht, dtype=np.float32)
            # suppose dim = 2
            Bas2 = _read_score(h5, "mc_bkg", hint)

            Tas2 = _read_score(h5, "mc_tt",  hint)

            Aas2 = _read_score(h5, "mc_aa",  hint)

            if Bas2 is None or Tas2 is None or Aas2 is None:
                raise SystemExit(
                    f"[read_any_h5] MC file missing score{hint:02d}. "
                    f"Expected keys like mc_bkg_score{hint:02d}, mc_tt_score{hint:02d}, mc_aa_score{hint:02d}. "
                    f"Top-level keys: {sorted(list(keys))}"
                )



            return dict(
                Bht=Bht, Bnpv=Bnpv,
                Bas2=Bas2,
                Tht=Tht, Tnpv=Tnpv,
                Tas2=Tas2, 
                Aht=Aht, Anpv=Anpv,
                Aas2=Aas2,
                meta=dict(matched_by_index=False),
            )

        # ------------------------------------------------------------
        # Case B) RealData Trigger_food_Data (unpaired) OR paired Matched_data_2016_dim2.h5
        #   - Paired Matched_data has data_Npv not data_bkg_Npv
        #   - Unpaired Trigger_food_Data has data_bkg_Npv and also data_tt_Npv / data_aa_Npv
        # ------------------------------------------------------------
        has_bkg = ("data_bkg_ht" in keys)
        has_npvs_any = ("data_Npv" in keys) or ("data_bkg_Npv" in keys)
        has_tt = ("data_tt_ht" in keys)
        has_aa = ("data_aa_ht" in keys)

        if has_bkg and has_npvs_any and has_tt and has_aa:
            # Background arrays
            Bht = h5["data_bkg_ht"][:]
            # paired file uses data_Npv; unpaired uses data_bkg_Npv
            npv_key = _first_present(keys, ["data_Npv", "data_bkg_Npv"])
            Bnpv = h5[npv_key][:]

            # Signal arrays (already aligned to the background npv distribution if paired)
            Tht = h5["data_tt_ht"][:]
            Aht = h5["data_aa_ht"][:]

            # keep these for the "mask by npv range" branch
            # (in paired file they exist as data_tt_Npv / data_aa_Npv written by run_pairing_npv)
            Tnpv_k = _first_present(keys, ["data_tt_Npv", "data_tt_npv"])
            Anpv_k = _first_present(keys, ["data_aa_Npv", "data_aa_npv"])
            Tnpv = h5[Tnpv_k][:] if Tnpv_k else np.zeros_like(Tht, dtype=np.float32)
            Anpv = h5[Anpv_k][:] if Anpv_k else np.zeros_like(Aht, dtype=np.float32)

            Bas2 = _read_score(h5, "data_bkg", hint)

            Tas2 = _read_score(h5, "data_tt",  hint)

            Aas2 = _read_score(h5, "data_aa",  hint)

            if Bas2 is None or Tas2 is None or Aas2 is None:
                raise SystemExit(
                    f"[read_any_h5] Data file missing score{hint:02d}. "
                    f"Expected keys like data_bkg_score{hint:02d}, data_tt_score{hint:02d}, data_aa_score{hint:02d}. "
                    f"Top-level keys: {sorted(list(keys))}"
                )


            # IMPORTANT:
            # - If file has data_Npv, tt/aa were already matched: should start with Matched_data_2016_dim2.h5 -> treat as matched_by_index=True
            # - If file has data_bkg_Npv, it’s unpaired Trigger_food_Data.h5 -> matched_by_index=False
            matched_by_index = ("data_Npv" in keys)

            return dict(
                Bht=Bht, Bnpv=Bnpv,
                Bas2=Bas2, 
                Tht=Tht, Tnpv=Tnpv,
                Tas2=Tas2, 
                Aht=Aht, Anpv=Anpv,
                Aas2=Aas2, 
                meta=dict(matched_by_index=matched_by_index),
            )

        # ------------------------------------------------------------
        # Fall back: unknown layout
        # ------------------------------------------------------------
        raise SystemExit(
            "[read_any_h5] Unrecognized H5 layout. "
            "Run with --print-keys to inspect keys.\n"
            f"Top-level keys: {sorted(list(keys))}"
        )
