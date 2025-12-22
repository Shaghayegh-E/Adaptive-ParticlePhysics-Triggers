#!/usr/bin/env python3
"""
demo_single_trigger_grpo_as_feature.py

AS-only single-trigger control: Constant vs PD vs GRPO (bandit-style).

Key GRPO idea:
  At each micro-step, given obs s:
    - sample G candidate actions from policy
    - evaluate reward for each candidate (counterfactual)
    - compute relative advantage A_i = r_i - mean(r_group)
    - update policy with PPO-style clipping + KL(pi||pi_ref)
    - execute the best candidate action (or sample) to advance the cut

This tends to be more stable than DQN for AD because:
  - no Q bootstrapping
  - direct policy optimization with group-relative advantages
  - easy to add stability penalties (occupancy/sensitivity) in reward
"""

import argparse
import random
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from controllers import PD_controller2
from triggers import Sing_Trigger
from RL.utils import (
    add_cms_header, save_png, print_h5_tree, read_any_h5
)
from RL.dqn_agent import make_event_seq_as, shield_delta, compute_reward  # reuse your existing code
from RL.grpo_agent import GRPOAgent, GRPOConfig


SEED = 20251221
random.seed(SEED)
np.random.seed(SEED)


def near_occupancy(x, cut, widths):
    x = np.asarray(x, dtype=np.float32)
    out = []
    for w in widths:
        out.append(float(np.mean(np.abs(x - cut) <= float(w))))
    return np.array(out, dtype=np.float32)


@dataclass
class RollingWindow:
    def __init__(self, max_events: int):
        self.max_events = int(max_events)
        self._bas = deque(maxlen=self.max_events)
        self._bnpv = deque(maxlen=self.max_events)

    def append(self, bas, bnpv):
        self._bas.extend(np.asarray(bas, dtype=np.float32).tolist())
        self._bnpv.extend(np.asarray(bnpv, dtype=np.float32).tolist())

    def get(self):
        return (
            np.fromiter(self._bas, dtype=np.float32),
            np.fromiter(self._bnpv, dtype=np.float32),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Data/Trigger_food_MC.h5", choices = ["Data/Trigger_food_MC.h5", "Data/Matched_data_2016_dim2.h5"])
    ap.add_argument("--outdir", default="outputs/demo_sing_grpo_as_feature")
    ap.add_argument("--control", default="MC", choices=["MC", "RealData"])
    ap.add_argument("--score-dim-hint", type=int, default=2)
    ap.add_argument("--as-dim", type=int, default=2, choices=[1, 2, 4])

    ap.add_argument("--as-deltas", type=str, default="-3,-1.5,0,1.5,3")
    ap.add_argument("--as-step", type=float, default=0.5)

    ap.add_argument("--print-keys", action="store_true")
    ap.add_argument("--print-keys-max", type=int, default=None)

    ap.add_argument("--window-events-chunk-size", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--inner-stride", type=int, default=10000)

    # GRPO knobs
    ap.add_argument("--group-size", type=int, default=16, help="GRPO group candidates per micro-step")
    ap.add_argument("--train-every", type=int, default=50, help="policy update frequency (micro-steps)")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)

    # objective/reward
    ap.add_argument("--target", type=float, default=0.25)  # percent
    ap.add_argument("--tol", type=float, default=0.02)      # percent (band)
    ap.add_argument("--alpha", type=float, default=0.4)     # signal bonus
    ap.add_argument("--beta", type=float, default=0.2)      # move penalty

    # optional stabilization (AD-specific)
    ap.add_argument("--occ-pen", type=float, default=0.0,
                    help="extra penalty weight for near-cut occupancy * |delta| (suggest 0.5~3.0)")
    args = ap.parse_args()

    if args.print_keys:
        print_h5_tree(args.input, max_items=args.print_keys_max)
        raise SystemExit(0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = read_any_h5(args.input, score_dim_hint=args.score_dim_hint)
    matched_by_index = bool(d["meta"].get("matched_by_index", False))

    Bnpv = d["Bnpv"]
    Tnpv = d["Tnpv"]
    Anpv = d["Anpv"]

    # choose AS
    if args.as_dim == 2:
        Bas, Tas, Aas = d["Bas2"], d["Tas2"], d["Aas2"]
    elif args.as_dim == 1:
        Bas, Tas, Aas = d["Bas1"], d["Tas1"], d["Aas1"]
    else:
        Bas, Tas, Aas = d["Bas4"], d["Tas4"], d["Aas4"]

    if Bas is None or Tas is None or Aas is None:
        raise SystemExit("AS arrays missing for requested --as-dim.")

    N = len(Bas)
    chunk_size = 50000 if args.control == "MC" else 20000
    start_event = chunk_size * 10
    start_event = max(0, (start_event // chunk_size) * chunk_size)
    if start_event + chunk_size > N:
        start_event = max(0, ((N - chunk_size) // chunk_size) * chunk_size)

    # fixed cut from calibration window
    win_lo = min(start_event, N - 1)
    win_hi = min(start_event + (100000 if args.control == "MC" else 10000), N)
    fixed_AS_cut = float(np.percentile(Bas[win_lo:win_hi], 99.75))

    # clip range
    ref_as = Bas[win_lo:win_hi]
    as_lo = float(np.min(ref_as))
    as_hi = float(np.max(ref_as))
    as_mid = 0.5 * (as_lo + as_hi)
    as_span = max(1e-6, as_hi - as_lo)

    print(f"[INFO] matched_by_index={matched_by_index} N={N} chunk={chunk_size} start={start_event}")
    print(f"[AS dim={args.as_dim}] fixed={fixed_AS_cut:.6f} clip=({as_lo:.6f},{as_hi:.6f}) as_step={args.as_step}")

    # PD init
    AS_cut_pd = fixed_AS_cut
    pre_as_err = 0.0

    # GRPO init
    AS_cut_grpo = fixed_AS_cut
    last_das = 0.0
    prev_bg_as = None

    # action space
    AS_DELTAS = np.array([float(x) for x in args.as_deltas.split(",")], dtype=np.float32)
    AS_STEP = float(args.as_step)
    MAX_DELTA_AS = float(np.max(np.abs(AS_DELTAS))) * AS_STEP

    # features
    K = int(args.seq_len)
    near_widths_as = (0.25, 0.5, 1.0)
    feat_dim_as = 10 + len(near_widths_as)  # must match make_event_seq_as()

    # GRPO agent
    cfg = GRPOConfig(
        lr=args.lr,
        beta_kl=args.beta_kl,
        ent_coef=args.ent_coef,
        device="cpu",
        batch_size=256,
        train_epochs=2,
        ref_update_interval=200,
    )
    agent = GRPOAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=cfg, seed=SEED)

    # rolling window for event features
    roll = RollingWindow(max_events=int(args.window_events_chunk_size * chunk_size))

    # logs
    RATE_SCALE_KHZ = 400.0
    target = float(args.target)
    tol = float(args.tol)

    R_const, R_pd, R_grpo = [], [], []
    Cut_pd, Cut_grpo = [], []
    TT_const, TT_pd, TT_grpo = [], [], []
    AA_const, AA_pd, AA_grpo = [], [], []
    losses = []
    rewards = []

    # loop
    batch_starts = list(range(start_event, N, chunk_size))
    micro_counter = 0

    for t, I in enumerate(batch_starts):
        end = min(I + chunk_size, N, len(Bnpv))
        if end <= I:
            break

        idx = np.arange(I, end)
        bas = Bas[idx]
        bnpv = Bnpv[idx]

        # signals for the chunk
        if matched_by_index:
            end_sig = min(end, len(Tas), len(Aas), len(Tnpv), len(Anpv))
            idx_sig = np.arange(I, end_sig)
            sas_tt = Tas[idx_sig]
            sas_aa = Aas[idx_sig]
        else:
            npv_min = float(np.min(bnpv))
            npv_max = float(np.max(bnpv))
            mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
            mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)
            sas_tt = Tas[mask_tt]
            sas_aa = Aas[mask_aa]

        stride = max(500, int(args.inner_stride))
        n_micro = max(1, int(np.ceil((end - I) / stride)))

        micro_rewards = []

        for j in range(n_micro):
            j_lo = I + j * stride
            j_hi = min(I + (j + 1) * stride, end)
            if j_hi <= j_lo:
                continue

            idxj = np.arange(j_lo, j_hi)
            bas_j = Bas[idxj]
            bnpv_j = Bnpv[idxj]

            roll.append(bas_j, bnpv_j)
            bas_w, bnpv_w = roll.get()

            # bg rate before
            bg_before = Sing_Trigger(bas_j, AS_cut_grpo)
            if prev_bg_as is None:
                prev_bg_as = bg_before

            # build obs
            obs = make_event_seq_as(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_before,
                prev_bg_rate=prev_bg_as,
                cut=AS_cut_grpo,
                as_mid=as_mid, as_span=as_span,
                target=target,
                K=K,
                last_delta=last_das,
                max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            # ---- GRPO group sampling ----
            G = int(args.group_size)
            acts, old_logps = agent.sample_group_actions(obs, group_size=G, temperature=float(args.temperature))

            # Evaluate each candidate action counterfactually
            cand_rewards = np.zeros(G, dtype=np.float32)
            cand_deltas = np.zeros(G, dtype=np.float32)
            cand_bg_after = np.zeros(G, dtype=np.float32)

            # compute near-occupancy (chunk-level proxy) for optional penalty
            occ_mid = float(near_occupancy(bas_j, AS_cut_grpo, near_widths_as)[1])  # w=0.5

            for k in range(G):
                a = int(acts[k])
                das = float(AS_DELTAS[a] * AS_STEP)
                cand_deltas[k] = das

                # candidate next cut
                cut_next = float(np.clip(AS_cut_grpo + das, as_lo, as_hi))

                bg_after = Sing_Trigger(bas_j, cut_next)
                cand_bg_after[k] = bg_after

                # signal accepts (micro-step proxy: use sas_tt/sas_aa from chunk as approximation)
                tt_after = Sing_Trigger(sas_tt, cut_next)
                aa_after = Sing_Trigger(sas_aa, cut_next)

                r = compute_reward(
                    bg_rate=bg_after,
                    target=target,
                    tol=tol,
                    sig_rate_1=tt_after,
                    sig_rate_2=aa_after,
                    delta_applied=das,
                    max_delta=MAX_DELTA_AS,
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    prev_bg_rate=bg_before,
                    gamma_stab=0.3,
                )

                # optional stabilization: penalize moving when occupancy is high
                if args.occ_pen > 0:
                    r -= float(args.occ_pen) * occ_mid * (abs(das) / (MAX_DELTA_AS + 1e-6))

                cand_rewards[k] = r

            r_mean = float(np.mean(cand_rewards))
            adv = cand_rewards - r_mean

            # push group samples to buffer
            for k in range(G):
                agent.buf.push(obs, int(acts[k]), float(old_logps[k]), float(adv[k]))

            # pick executed action: best reward in group (stable)
            k_best = int(np.argmax(cand_rewards))
            a_exec = int(acts[k_best])
            das_exec = float(AS_DELTAS[a_exec] * AS_STEP)

            # safety shield (optional but recommended)
            sd = shield_delta(bg_before, target, tol, MAX_DELTA_AS)
            if sd is not None:
                das_exec = float(sd)

            cut_next = float(np.clip(AS_cut_grpo + das_exec, as_lo, as_hi))
            bg_after_exec = Sing_Trigger(bas_j, cut_next)

            # execute transition
            AS_cut_grpo = cut_next
            prev_bg_as = bg_after_exec
            last_das = das_exec

            micro_rewards.append(float(cand_rewards[k_best]))
            micro_counter += 1

            # update periodically
            if micro_counter % int(args.train_every) == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

        # ---- end micro loop ----

        # chunk-level logging
        bg_const = Sing_Trigger(bas, fixed_AS_cut)
        bg_pd = Sing_Trigger(bas, AS_cut_pd)
        bg_grpo = Sing_Trigger(bas, AS_cut_grpo)

        # PD update once per chunk
        AS_cut_pd, pre_as_err = PD_controller2(bg_pd, pre_as_err, AS_cut_pd)
        AS_cut_pd = float(np.clip(AS_cut_pd, as_lo, as_hi))

        tt_const = Sing_Trigger(sas_tt, fixed_AS_cut)
        tt_pd = Sing_Trigger(sas_tt, AS_cut_pd)
        tt_grpo = Sing_Trigger(sas_tt, AS_cut_grpo)

        aa_const = Sing_Trigger(sas_aa, fixed_AS_cut)
        aa_pd = Sing_Trigger(sas_aa, AS_cut_pd)
        aa_grpo = Sing_Trigger(sas_aa, AS_cut_grpo)

        R_const.append(bg_const * RATE_SCALE_KHZ)
        R_pd.append(bg_pd * RATE_SCALE_KHZ)
        R_grpo.append(bg_grpo * RATE_SCALE_KHZ)

        Cut_pd.append(AS_cut_pd)
        Cut_grpo.append(AS_cut_grpo)

        TT_const.append(tt_const); TT_pd.append(tt_pd); TT_grpo.append(tt_grpo)
        AA_const.append(aa_const); AA_pd.append(aa_pd); AA_grpo.append(aa_grpo)

        rewards.append(float(np.mean(micro_rewards)) if micro_rewards else np.nan)

        if t % 5 == 0:
            print(f"[chunk {t:4d}] "
                  f"AS bg% const={bg_const:.3f} pd={bg_pd:.3f} grpo={bg_grpo:.3f} "
                  f"| cut pd={AS_cut_pd:.5f} grpo={AS_cut_grpo:.5f} "
                  f"| reward={rewards[-1]} loss={losses[-1] if losses else None}")

    # ---- plotting ----
    R_const = np.asarray(R_const)
    R_pd = np.asarray(R_pd)
    R_grpo = np.asarray(R_grpo)
    Cut_pd = np.asarray(Cut_pd)
    Cut_grpo = np.asarray(Cut_grpo)

    time = np.linspace(0, 1, len(R_const))
    run_label = "MC" if args.control == "MC" else "283408"

    upper_tol_khz = 110.0
    lower_tol_khz = 90.0

    import matplotlib.pyplot as plt

    # rate plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, R_const, linestyle="--", linewidth=2.4, label="Constant")
    ax.plot(time, R_pd, linewidth=2.4, label="PD")
    ax.plot(time, R_grpo, linewidth=3.0, linestyle=(0, (8, 2, 2, 2)), label="GRPO")

    ax.axhline(upper_tol_khz, linestyle="--", linewidth=1.2)
    ax.axhline(lower_tol_khz, linestyle="--", linewidth=1.2)
    ax.fill_between(time, lower_tol_khz, upper_tol_khz, alpha=0.12, label="Tolerance band")

    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Background rate [kHz]")
    ax.set_ylim(0, 200)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, title="AD Trigger")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "bas_rate_pidData_grpo"))
    plt.close(fig)

    # cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, Cut_pd, linewidth=2.4, label="PD")
    ax.plot(time, Cut_grpo, linewidth=3.0, linestyle=(0, (8, 2, 2, 2)), label="GRPO")
    ax.axhline(y=fixed_AS_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_AS_cut")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("AS_cut")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, title="AD Cut")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "as_cut_pidData_grpo"))
    plt.close(fig)

    # reward trace
    if rewards:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, np.asarray(rewards, dtype=np.float32), linewidth=1.5)
        ax.set_xlabel("Time (Fraction of Run)")
        ax.set_ylabel("Mean micro reward")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "reward_as_pidData_grpo"))
        plt.close(fig)

    # loss trace
    if losses:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses)), losses, linewidth=1.5)
        ax.set_xlabel("Policy update index")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "grpo_loss_as"))
        plt.close(fig)

    print(f"\n[OK] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
