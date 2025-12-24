#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def analyze(csv_path, outdir="grpo_reward_analysis",
            sample_reward_col=None,
            exec_reward_col=None,
            group_col="chunk"):
    csv_path = Path(csv_path)
    outdir = Path(outdir) / csv_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"\n== {csv_path} ==")
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    # --- pick reward columns ---
    # sampled reward should be per-row (per-sample) -> prefer reward_raw
    if sample_reward_col is None:
        sample_reward_col = pick_first_existing(df, ["reward_raw", "reward_exec"])
    if sample_reward_col is None:
        raise ValueError("Couldn't find a sampled reward column. Expected one of: reward_raw, reward_exec")

    # executed reward is usually reward_exec on executed rows; fallback to reward_raw
    if exec_reward_col is None:
        exec_reward_col = pick_first_existing(df, ["reward_exec", "reward_raw"])

    # required columns
    if group_col not in df.columns:
        raise ValueError(f"Need group column '{group_col}'. Available: {list(df.columns)}")
    if "executed" not in df.columns:
        raise ValueError("Need column 'executed' (0/1) to identify chosen action.")
    if "a" not in df.columns:
        print("Note: no 'a' action column found; action histograms will be skipped.")

    # --- basic reward stats ---
    r = df[sample_reward_col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    print(f"Using sampled reward col: {sample_reward_col}")
    print("Reward min/mean/max:", float(np.min(r)), float(np.mean(r)), float(np.max(r)))
    print("Reward p1/p5/p50/p95/p99:", np.percentile(r, [1,5,50,95,99]).tolist())

    plt.figure()
    plt.hist(r, bins=80)
    plt.xlabel(sample_reward_col)
    plt.ylabel("count")
    plt.title("Sampled reward histogram (all rows)")
    plt.savefig(outdir / "reward_hist.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- group size sanity (how many samples per chunk) ---
    gsize = df.groupby(group_col).size()
    print("Samples per chunk: min/median/max =", int(gsize.min()), float(gsize.median()), int(gsize.max()))
    gsize.to_csv(outdir / "samples_per_chunk.csv")

    # --- best sampled reward per chunk (computed from sample_reward_col) ---
    best = df.groupby(group_col)[sample_reward_col].max().rename("r_best").reset_index()

    # --- executed reward per chunk ---
    executed = pd.to_numeric(df["executed"], errors="coerce").fillna(0).astype(int)
    exec_rows = df[executed == 1].copy()
    if len(exec_rows) == 0:
        print("WARNING: no executed==1 rows found; can't compute chosen-vs-best/regret.")
        exec_df = None
    else:
        # use exec_reward_col; if NaN, fallback to sample_reward_col
        exec_rows["r_exec"] = exec_rows[exec_reward_col]
        exec_rows["r_exec"] = exec_rows["r_exec"].where(np.isfinite(exec_rows["r_exec"]),
                                                        exec_rows[sample_reward_col])
        exec_df = exec_rows.groupby(group_col)["r_exec"].mean().reset_index()

    # --- regret = best - executed (should go DOWN if GRPO learns) ---
    if exec_df is not None:
        merged = best.merge(exec_df, on=group_col, how="inner")
        merged["regret"] = merged["r_best"] - merged["r_exec"]
        merged.to_csv(outdir / "best_exec_regret.csv", index=False)

        plt.figure()
        plt.plot(merged[group_col], merged["r_exec"], label="executed")
        plt.plot(merged[group_col], merged["r_best"], label="best sampled")
        plt.xlabel(group_col); plt.ylabel("reward")
        plt.title("Executed vs Best-sampled reward")
        plt.legend()
        plt.savefig(outdir / "executed_vs_best.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(merged[group_col], merged["regret"])
        plt.xlabel(group_col); plt.ylabel("regret = best - executed")
        plt.title("Regret proxy over time (want decreasing trend)")
        plt.savefig(outdir / "regret_vs_time.png", dpi=200, bbox_inches="tight")
        plt.close()

    # --- per-chunk reward mean/std/range (variance diagnosis) ---
    stats = df.groupby(group_col)[sample_reward_col].agg(["count","mean","std","min","max"]).reset_index()
    stats["range"] = stats["max"] - stats["min"]
    stats.to_csv(outdir / "per_chunk_reward_stats.csv", index=False)

    plt.figure()
    plt.plot(stats[group_col], stats["mean"])
    plt.fill_between(stats[group_col],
                     (stats["mean"] - stats["std"].fillna(0)),
                     (stats["mean"] + stats["std"].fillna(0)),
                     alpha=0.2)
    plt.xlabel(group_col); plt.ylabel("reward")
    plt.title("Sampled reward mean ± std per chunk")
    plt.savefig(outdir / "reward_mean_std_vs_chunk.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(stats[group_col], stats["range"])
    plt.xlabel(group_col); plt.ylabel("max-min reward")
    plt.title("Within-chunk reward range (variance / signal)")
    plt.savefig(outdir / "reward_range_vs_chunk.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- action coverage (sampled vs executed) ---
    if "a" in df.columns:
        plt.figure()
        df["a"].value_counts().sort_index().plot(kind="bar")
        plt.xlabel("action a"); plt.ylabel("count (sampled)")
        plt.title("Sampled action histogram")
        plt.savefig(outdir / "sampled_action_hist.png", dpi=200, bbox_inches="tight")
        plt.close()

        if exec_df is not None:
            plt.figure()
            exec_rows["a"].value_counts().sort_index().plot(kind="bar")
            plt.xlabel("action a"); plt.ylabel("count (executed)")
            plt.title("Executed action histogram")
            plt.savefig(outdir / "executed_action_hist.png", dpi=200, bbox_inches="tight")
            plt.close()

    # --- shielding rate (did your safety clamp activate?) ---
    if "shielded" in df.columns:
        sh = df.groupby(group_col)["shielded"].mean().reset_index().rename(columns={"shielded":"shield_rate"})
        sh.to_csv(outdir / "shield_rate_by_chunk.csv", index=False)
        plt.figure()
        plt.plot(sh[group_col], sh["shield_rate"])
        plt.xlabel(group_col); plt.ylabel("mean(shielded)")
        plt.title("Shielding rate over time")
        plt.savefig(outdir / "shield_rate_vs_chunk.png", dpi=200, bbox_inches="tight")
        plt.close()

    print("Wrote outputs to:", outdir)

if __name__ == "__main__":
    analyze("outputs/demo_sing_grpo_as_feature/grpo_as_sampled_rewards.csv")
    analyze("outputs/demo_sing_grpo_as_feature/grpo_ht_sampled_rewards.csv")
