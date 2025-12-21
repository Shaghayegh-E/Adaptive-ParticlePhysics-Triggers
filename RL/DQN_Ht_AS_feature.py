#!/usr/bin/env python3
"""
DQN_Ht_AS_feature.py

SingleTrigger: 
Constant vs PD vs DQN
- HT trigger: accept = (HT >= Ht_cut)
- AS trigger: accept = (AS >= AS_cut)

We train two independent DQNs:
  (1) DQN_HT controls Ht_cut using HT-only rates
  (2) DQN_AS controls AS_cut using AS-only rates For AS only, we use binned steps to ensure stability.

Outputs:
outdir = RL_outputs/demo_sing_dqn_separate_feature
HT trigger outputs:
  - bht_rate_pidData_dqn_feature.png          (HT background rate [kHz])
  - ht_cut_pidData_dqn_feature.png            (Ht_cut evolution)
  - sht_rate_pidData2data_dqn_feature.png     (cumulative signal eff, relative to t0)
  - L_sht_rate_pidData2data_dqn_feature.png   (local signal eff, relative to t0)
  - dqn_loss_ht_feature.png                   (HT DQN loss)

AS trigger outputs:
  - bas_rate_pidData_dqn_feature.png          (AS background rate [kHz])
  - as_cut_pidData_dqn_feature.png            (AS_cut evolution)
  - sas_rate_pidData2data_dqn_feature.png     (cumulative signal eff, relative to t0)
  - L_sas_rate_pidData2data_dqn_feature.png   (local signal eff, relative to t0)
  - dqn_loss_as_feature.png                   (AS DQN loss)

"""
from dataclasses import dataclass
import random
import argparse
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin  # noqa: F401
import csv
from pathlib import Path
from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
from RL.utils import cummean, rel_to_t0, add_cms_header, plot_rate_with_tolerance, save_png, print_h5_tree, read_any_h5, compute_auroc_windows_separate, compute_operating_point_windows_separate #save_pdf_png,
from RL.dqn_agent import DQNAgent, make_obs, shield_delta, compute_reward, DQNConfig, SeqDQNAgent, make_event_seq_as, make_event_seq_ht

# ------------------------- Fixing seed for reproducibility -------------------------
SEED = 20251213
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class RollingWindow: #sliding window for event-level features
    def __init__(self, max_events: int):
        self.max_events = int(max_events)
        self._bht  = deque(maxlen=self.max_events)
        self._bas  = deque(maxlen=self.max_events)
        self._bnpv = deque(maxlen=self.max_events)

    def append(self, bht, bas, bnpv):
        self._bht.extend(np.asarray(bht,  dtype=np.float32).tolist())
        self._bas.extend(np.asarray(bas,  dtype=np.float32).tolist())
        self._bnpv.extend(np.asarray(bnpv, dtype=np.float32).tolist())

    def get(self):
        return (
            np.fromiter(self._bht,  dtype=np.float32),
            np.fromiter(self._bas,  dtype=np.float32),
            np.fromiter(self._bnpv, dtype=np.float32),
        )


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Data/Matched_data_2016_dim2.h5",
                    help="Matched_data_*.h5 (data) or Trigger_food_*.h5 (MC)")
    ap.add_argument("--outdir", default="RL_outputs/demo_sing_dqn_separate_feature", help="output dir")
    ap.add_argument("--control", default="MC", choices=["MC", "RealData"],
                    help="Control type: MC or RealData")
    ap.add_argument("--score-dim-hint", type=int, default=2,
                    help="If file has only scoreXX, use this dim (e.g. 2 -> score02).")
    ap.add_argument("--as-dim", type=int, default=2, choices=[1, 4, 2],
                    help="Which AS dimension to use (1->score01, 4->score04).")

    ap.add_argument("--ht-deltas", type=str, default="-2,-1,0,1,2",
                    help="HT DQN deltas (in HT cut units, like your HT script).")
    ap.add_argument("--as-deltas", type=str, default="-3,-1.5,0,1.5,3",
                    help="AS DQN delta multipliers.")
    ap.add_argument("--as-step", type=float, default=0.5,
                    help="AS step: final delta or step we make for AS trigger = as_delta * as_step (tune the AS scale).")
    ap.add_argument("--print-keys", action="store_true",
                help="Print all HDF5 groups/datasets (with shapes/dtypes) and exit.")
    ap.add_argument("--print-keys-max", type=int, default=None,
                help="Optional cap on number of printed items.")
    # Let's predefine AS bins to ensure better dqn stability
    ap.add_argument("--as-bins", type=int, default=20, choices=[10, 20, 30, 40, 50],
                help="Number of bins used to define AS step a in the cut-range.")
    ap.add_argument("--as-p-lo", type=float, default=99.0,
                help="Low percentile for AS cut range.")
    ap.add_argument("--as-p-hi", type=float, default=99.995,
                help="High percentile for AS cut range.")
    ap.add_argument("--window-events-chunk-size", type=int, default=3,
                help="How many most-recent background events of chunk size to keep for features (sliding window).")
    ap.add_argument("--seq-len", type=int, default=128,
                help="Sequence length K passed into make_event_seq_* (downsample/pad to this).")
    ap.add_argument("--inner-stride", type=int, default=10000,
        help="Micro-step size inside each chunk (events). 50k chunk with 10k stride -> 5 transitions per chunk.")
    ap.add_argument("--train-steps-per-micro", type=int, default=1,
        help="How many gradient updates to do per micro-step (usually 1).")
    
    args = ap.parse_args()
    if args.control == "MC":
        run_label = "MC"
    else:
        run_label = "283408"
    if args.print_keys:
        print_h5_tree(args.input, max_items=args.print_keys_max)
        raise SystemExit(0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = read_any_h5(args.input, score_dim_hint=args.score_dim_hint)
    matched_by_index = bool(d["meta"].get("matched_by_index", False))

    Bht, Bnpv = d["Bht"], d["Bnpv"]
    Tht, Tnpv = d["Tht"], d["Tnpv"]
    Aht, Anpv = d["Aht"], d["Anpv"]

    # Pick AS dim
    # For now only support dim=2 (score02) in this script
    if args.as_dim == 2:
        Bas, Tas, Aas = d["Bas2"], d["Tas2"], d["Aas2"]

    if Bas is None or Tas is None or Aas is None:
        raise SystemExit("AS arrays missing for requested --as-dim. Check your input file.")

    N = len(Bht)
    if args.control == "MC":
        chunk_size = 50000
    else:
        #real data
        chunk_size = 20000
    start_event = chunk_size * 10

    # Align start_event to chunk boundary
    start_event = max(0, (start_event // chunk_size) * chunk_size)
    if start_event + chunk_size > N:
        start_event = max(0, ((N - chunk_size) // chunk_size) * chunk_size)

    # Fixed cuts from a reference window (mimic Data_SingleTrigger.py)
    win_lo = min(start_event, N - 1)
    if args.control == "MC":
        # Run_SingleTrigger.py uses 100k for MC DEBUG 
        win_hi = min(start_event + 100000, N)
    else:
        # Data_SingleTrigger.py uses 10k for RealData DEBUG
        win_hi = min(start_event + 10000, N)


    fixed_Ht_cut = float(np.percentile(Bht[win_lo:win_hi], 99.75))
    fixed_AS_cut = float(np.percentile(Bas[win_lo:win_hi], 99.75))

    # Clip ranges
    ht_lo = float(np.percentile(Bht[start_event:], 95.0))
    ht_hi = float(np.percentile(Bht[start_event:], 99.99))
    ht_mid = 0.5 * (ht_lo + ht_hi)
    ht_span = max(1.0, ht_hi - ht_lo)

    ref_as = Bas[win_lo:win_hi]

    as_lo = float(np.min(ref_as))
    as_hi = float(np.max(ref_as))
    as_mid  = 0.5 * (as_lo + as_hi)
    as_span = max(1e-6, as_hi - as_lo)


    print(f"[INFO] matched_by_index={matched_by_index} N={N} chunk={chunk_size} start_event={start_event}")
    print(f"[HT] fixed={fixed_Ht_cut:.3f} clip=({ht_lo:.3f},{ht_hi:.3f}) window=[{win_lo}:{win_hi}]")
    print(f"[AS dim={args.as_dim}] fixed={fixed_AS_cut:.6f} clip=({as_lo:.6f},{as_hi:.6f}) as_step={args.as_step}")

    # ------------------------- init cuts -------------------------
    # HT
    Ht_cut_pd  = fixed_Ht_cut
    Ht_cut_dqn = fixed_Ht_cut
    pre_ht_err = 0.0

    # AS
    AS_cut_pd  = fixed_AS_cut
    AS_cut_dqn = fixed_AS_cut
    pre_as_err = 0.0

    # ------------------------- DQN configs -------------------------
    target = 0.25  # %
    tol = 0.02     # background - target/tolerance for reward?
    alpha = 0.4    # signal bonus
    beta  = 0.2   # move penalty

    HT_DELTAS = np.array([float(x) for x in args.ht_deltas.split(",")], dtype=np.float32)
    HT_STEP = 1.0
    MAX_DELTA_HT = float(np.max(np.abs(HT_DELTAS))) * HT_STEP

    AS_DELTAS = np.array([float(x) for x in args.as_deltas.split(",")], dtype=np.float32)
    AS_STEP = float(args.as_step)
    MAX_DELTA_AS = float(np.max(np.abs(AS_DELTAS))) * AS_STEP
    print("MAX_DELTA_AS=", MAX_DELTA_AS)

    cfg = DQNConfig(lr=5e-4, gamma=0.95, batch_size=32, target_update=200)
    
    # Make AS agent slower learning rate for faster adaptation
    cfg_as = DQNConfig(lr=1e-4, gamma=0.95, batch_size=32, target_update=200)
    K = int(args.seq_len)
    near_widths_ht = (5.0, 10.0, 20.0)
    feat_dim_ht = 10 + len(near_widths_ht)   # 13       

    near_widths_as = (0.01, 0.02, 0.05)
    feat_dim_as = 10 + len(near_widths_as)   # 13

    agent_ht = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS), cfg=cfg, seed=SEED)
    agent_as = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=cfg_as, seed=SEED)

    roll = RollingWindow(max_events=int(args.window_events_chunk_size * chunk_size))

    # state trackers (HT)
    prev_obs_ht = None
    prev_act_ht = None
    prev_bg_ht = None
    last_dht = 0.0
    losses_ht = []

    # state trackers (AS)
    prev_obs_as = None
    prev_act_as = None
    prev_bg_as = None
    last_das = 0.0
    losses_as = []

    rewards_ht = []   # reward used to train HT agent per chunk
    rewards_as = []   # reward used to train AS agent per chunk

    # ------------------------- logs (HT) -------------------------
    R1_ht, R2_ht, R3_ht = [], [], []                  # background % (const, PD, DQN)
    Ht_pd_hist, Ht_dqn_hist = [], []
    L_tt_ht_const, L_tt_ht_pd, L_tt_ht_dqn = [], [], []
    L_aa_ht_const, L_aa_ht_pd, L_aa_ht_dqn = [], [], []

    # ------------------------- logs (AS) -------------------------
    R1_as, R2_as, R3_as = [], [], []                  # background % (const, PD, DQN)
    As_pd_hist, As_dqn_hist = [], []
    L_tt_as_const, L_tt_as_pd, L_tt_as_dqn = [], [], []
    L_aa_as_const, L_aa_as_pd, L_aa_as_dqn = [], [], []

    # ------------------------- batching loop -------------------------
    batch_starts = list(range(start_event, N, chunk_size))

    for t, I in enumerate(batch_starts):
        end = min(I + chunk_size, N, len(Bnpv), len(Bas))
        if end <= I:
            break
        idx = np.arange(I, end)

        # chunk background arrays
        bht  = Bht[idx]
        bas  = Bas[idx]
        bnpv = Bnpv[idx]

        # micro-step setup
        stride = max(500, int(args.inner_stride))
        chunk_len = end - I
        n_micro = max(1, int(np.ceil(chunk_len / stride)))

        micro_rewards_ht = []
        micro_rewards_as = []

        # -------------------------
        # micro loop: rl transitions
        # -------------------------
        bg_acc_ht = 0; bg_tot_ht = 0
        bg_acc_as = 0; bg_tot_as = 0
        for j in range(n_micro):
            j_lo = I + j * stride
            j_hi = min(I + (j + 1) * stride, end)
            if j_hi <= j_lo:
                continue

            idxj = np.arange(j_lo, j_hi)

            bht_j  = Bht[idxj]
            bas_j  = Bas[idxj]
            bnpv_j = Bnpv[idxj]

            # sliding-window update
            roll.append(bht_j, bas_j, bnpv_j)
            bht_w, bas_w, bnpv_w = roll.get()

            # signal subsets for THIS micro-step
            if matched_by_index:
                end_sig_j = min(j_hi, len(Tht), len(Aht), len(Tas), len(Aas))
                if j_lo >= end_sig_j:
                    # no signal available for this micro-step
                    sht_tt_j = np.empty(0, dtype=np.float32); sas_tt_j = np.empty(0, dtype=np.float32)
                    sht_aa_j = np.empty(0, dtype=np.float32); sas_aa_j = np.empty(0, dtype=np.float32)
                else:
                    idx_sig_j = np.arange(j_lo, end_sig_j)
                    sht_tt_j = Tht[idx_sig_j];  sas_tt_j = Tas[idx_sig_j]
                    sht_aa_j = Aht[idx_sig_j];  sas_aa_j = Aas[idx_sig_j]
            else:
                # Pick signal events with similar NPV range as the current micro background window
                npv_min = float(np.min(bnpv_j))
                npv_max = float(np.max(bnpv_j))

                mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
                mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)

                sht_tt_j = Tht[mask_tt];  sas_tt_j = Tas[mask_tt]
                sht_aa_j = Aht[mask_aa];  sas_aa_j = Aas[mask_aa]
            # global micro-step index for epsilon
            step = t * n_micro + j
            eps = max(0.05, 1.0 * (0.98 ** step))

            # =========================================================
            # HT micro-step: (s, a, r, s')
            # =========================================================
            bg_before_ht = Sing_Trigger(bht_j, Ht_cut_dqn)
            if prev_bg_ht is None:
                prev_bg_ht = bg_before_ht

            obs_ht = make_event_seq_ht(
                bht=bht_w, bnpv=bnpv_w,
                bg_rate=bg_before_ht,
                prev_bg_rate=prev_bg_ht,
                cut=Ht_cut_dqn,
                ht_mid=ht_mid, ht_span=ht_span,
                target=target,
                K=K,
                last_delta=last_dht,
                max_delta=MAX_DELTA_HT,
                near_widths=near_widths_ht,
            )

            act_ht = agent_ht.act(obs_ht, eps=eps)
            dht = float(HT_DELTAS[act_ht])

            # shield based on micro-step bg
            sd = shield_delta(bg_before_ht, target, tol, MAX_DELTA_HT)
            if sd is not None:
                dht = float(sd)

            Ht_cut_next = float(np.clip(Ht_cut_dqn + dht, ht_lo, ht_hi))

            bg_after_ht = Sing_Trigger(bht_j, Ht_cut_next)
            tt_after_ht = Sing_Trigger(sht_tt_j, Ht_cut_next)
            aa_after_ht = Sing_Trigger(sht_aa_j, Ht_cut_next)

            obs_ht_next = make_event_seq_ht(
                bht=bht_w, bnpv=bnpv_w,
                bg_rate=bg_after_ht,
                prev_bg_rate=bg_before_ht,
                cut=Ht_cut_next,
                ht_mid=ht_mid, ht_span=ht_span,
                target=target,
                K=K,
                last_delta=dht,
                max_delta=MAX_DELTA_HT,
                near_widths=near_widths_ht,
            )   

            r_ht = compute_reward(
                bg_rate=bg_after_ht,
                target=target,
                tol=tol,
                sig_rate_1=tt_after_ht,
                sig_rate_2=aa_after_ht,
                delta_applied=dht,
                max_delta=MAX_DELTA_HT,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=bg_before_ht,
                gamma_stab=0.3,
            )

            agent_ht.buf.push(obs_ht, act_ht, r_ht, obs_ht_next, done=False)
            for _ in range(int(args.train_steps_per_micro)):
                loss = agent_ht.train_step()
                if loss is not None:
                    losses_ht.append(loss)

            micro_rewards_ht.append(r_ht)

            # advance HT state
            Ht_cut_dqn = Ht_cut_next
            prev_bg_ht = bg_after_ht
            last_dht = dht

            # =========================================================
            # AS micro-step: (s, a, r, s')
            # =========================================================
            bg_before_as = Sing_Trigger(bas_j, AS_cut_dqn)
            if prev_bg_as is None:
                prev_bg_as = bg_before_as

            obs_as = make_event_seq_as(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_before_as,
                prev_bg_rate=prev_bg_as,
                cut=AS_cut_dqn,
                as_mid=as_mid, as_span=as_span,
                target=target,
                K=K,
                last_delta=last_das,
                max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            act_as = agent_as.act(obs_as, eps=eps)
            das = float(AS_DELTAS[act_as] * AS_STEP)

            sd = shield_delta(bg_before_as, target, tol, MAX_DELTA_AS)
            if sd is not None:
                das = float(sd)

            AS_cut_next = float(np.clip(AS_cut_dqn + das, as_lo, as_hi))

            bg_after_as = Sing_Trigger(bas_j, AS_cut_next)
            tt_after_as = Sing_Trigger(sas_tt_j, AS_cut_next)
            aa_after_as = Sing_Trigger(sas_aa_j, AS_cut_next)

            obs_as_next = make_event_seq_as(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_after_as,
                prev_bg_rate=bg_before_as,
                cut=AS_cut_next,
                as_mid=as_mid, as_span=as_span,
                target=target,
                K=K,
                last_delta=das,
                max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            r_as = compute_reward(
                bg_rate=bg_after_as,
                target=target,
                tol=tol,
                sig_rate_1=tt_after_as,
                sig_rate_2=aa_after_as,
                delta_applied=das,
                max_delta=MAX_DELTA_AS,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=bg_before_as,
                gamma_stab=0.3,
            )

            agent_as.buf.push(obs_as, act_as, r_as, obs_as_next, done=False)
            for _ in range(int(args.train_steps_per_micro)):
                loss = agent_as.train_step()
                if loss is not None:
                    losses_as.append(loss)

            micro_rewards_as.append(r_as)

            # advance AS state
            AS_cut_dqn = AS_cut_next
            prev_bg_as = bg_after_as
            last_das = das
        # -------------------------
        # AFTER MICRO LOOP (once per chunk): build chunk-level signals
        # -------------------------
        if matched_by_index:
            end_sig = min(end, len(Tht), len(Aht), len(Tas), len(Aas))
            idx_sig = np.arange(I, end_sig)
            sht_tt_j = Tht[idx_sig];  sas_tt_j = Tas[idx_sig]
            sht_aa_j = Aht[idx_sig];  sas_aa_j = Aas[idx_sig]
        else:
            npv_min = float(np.min(bnpv))
            npv_max = float(np.max(bnpv))
            mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
            mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)
            sht_tt_j = Tht[mask_tt];  sas_tt_j = Tas[mask_tt]
            sht_aa_j = Aht[mask_aa];  sas_aa_j = Aas[mask_aa]

        # -------------------------
        # CHUNK RATES for plots/logging
        # -------------------------
        # HT
        bg_const_ht = Sing_Trigger(bht, fixed_Ht_cut)
        bg_pd_ht = Sing_Trigger(bht, Ht_cut_pd)
        bg_dqn_ht = Sing_Trigger(bht, Ht_cut_dqn)   

        tt_const_ht = Sing_Trigger(sht_tt_j, fixed_Ht_cut)
        tt_pd_ht = Sing_Trigger(sht_tt_j, Ht_cut_pd)
        tt_dqn_ht = Sing_Trigger(sht_tt_j, Ht_cut_dqn)

        aa_const_ht = Sing_Trigger(sht_aa_j, fixed_Ht_cut)
        aa_pd_ht = Sing_Trigger(sht_aa_j, Ht_cut_pd)
        aa_dqn_ht = Sing_Trigger(sht_aa_j, Ht_cut_dqn)

        # PD update (once per chunk)
        Ht_cut_pd, pre_ht_err = PD_controller1(bg_pd_ht, pre_ht_err, Ht_cut_pd)
        Ht_cut_pd = float(np.clip(Ht_cut_pd, ht_lo, ht_hi))

        # chunk reward = mean of micro rewards
        reward_ht_t = float(np.mean(micro_rewards_ht)) if micro_rewards_ht else np.nan
        rewards_ht.append(reward_ht_t)

        # log once per chunk
        R1_ht.append(bg_const_ht); R2_ht.append(bg_pd_ht); R3_ht.append(bg_dqn_ht)

        Ht_pd_hist.append(Ht_cut_pd); Ht_dqn_hist.append(Ht_cut_dqn)
        L_tt_ht_const.append(tt_const_ht); L_tt_ht_pd.append(tt_pd_ht); L_tt_ht_dqn.append(tt_dqn_ht)
        L_aa_ht_const.append(aa_const_ht); L_aa_ht_pd.append(aa_pd_ht); L_aa_ht_dqn.append(aa_dqn_ht)

        # AS
        bg_const_as = Sing_Trigger(bas, fixed_AS_cut)
        bg_pd_as    = Sing_Trigger(bas, AS_cut_pd)
        bg_dqn_as   = Sing_Trigger(bas, AS_cut_dqn)   # <-- ADD THIS

        tt_const_as = Sing_Trigger(sas_tt_j, fixed_AS_cut)
        tt_pd_as    = Sing_Trigger(sas_tt_j, AS_cut_pd)
        tt_dqn_as   = Sing_Trigger(sas_tt_j, AS_cut_dqn)

        aa_const_as = Sing_Trigger(sas_aa_j, fixed_AS_cut)
        aa_pd_as    = Sing_Trigger(sas_aa_j, AS_cut_pd)
        aa_dqn_as   = Sing_Trigger(sas_aa_j, AS_cut_dqn)

        # PD update (once per chunk)
        AS_cut_pd, pre_as_err = PD_controller2(bg_pd_as, pre_as_err, AS_cut_pd)
        AS_cut_pd = float(np.clip(AS_cut_pd, as_lo, as_hi))

        reward_as_t = float(np.mean(micro_rewards_as)) if micro_rewards_as else np.nan
        rewards_as.append(reward_as_t)

        R1_as.append(bg_const_as); R2_as.append(bg_pd_as); R3_as.append(bg_dqn_as)
        As_pd_hist.append(AS_cut_pd); As_dqn_hist.append(AS_cut_dqn)
        L_tt_as_const.append(tt_const_as); L_tt_as_pd.append(tt_pd_as); L_tt_as_dqn.append(tt_dqn_as)
        L_aa_as_const.append(aa_const_as); L_aa_as_pd.append(aa_pd_as); L_aa_as_dqn.append(aa_dqn_as)

        # DEBUG print per 5 chunks
        if t % 5 == 0:
            lh = losses_ht[-1] if losses_ht else None
            la = losses_as[-1] if losses_as else None
            print(f"[batch {t:4d}] "
                f"HT bg% c={bg_const_ht:.3f} pd={bg_pd_ht:.3f} dqn={bg_dqn_ht:.3f} "
                f"| ht_cut pd={Ht_cut_pd:.1f} dqn={Ht_cut_dqn:.1f} loss={lh} "
                f"|| AS bg% c={bg_const_as:.3f} pd={bg_pd_as:.3f} dqn={bg_dqn_as:.3f} "
                f"| as_cut pd={AS_cut_pd:.4f} dqn={AS_cut_dqn:.4f} loss={la} "
                f"| reward_ht={reward_ht_t} reward_as={reward_as_t}")


    # ------------------------- convert + scale -------------------------
    RATE_SCALE_KHZ = 400.0
    upper_tol_khz = 0.275 * RATE_SCALE_KHZ
    lower_tol_khz = 0.225 * RATE_SCALE_KHZ

    # HT
    R1_ht = np.array(R1_ht) * RATE_SCALE_KHZ
    R2_ht = np.array(R2_ht) * RATE_SCALE_KHZ
    R3_ht = np.array(R3_ht) * RATE_SCALE_KHZ
    Ht_pd_hist = np.array(Ht_pd_hist)
    Ht_dqn_hist = np.array(Ht_dqn_hist)
    L_tt_ht_const = np.array(L_tt_ht_const)
    L_tt_ht_pd    = np.array(L_tt_ht_pd)
    L_tt_ht_dqn   = np.array(L_tt_ht_dqn)
    L_aa_ht_const = np.array(L_aa_ht_const)
    L_aa_ht_pd    = np.array(L_aa_ht_pd)
    L_aa_ht_dqn   = np.array(L_aa_ht_dqn)

    # AS
    R1_as = np.array(R1_as) * RATE_SCALE_KHZ
    R2_as = np.array(R2_as) * RATE_SCALE_KHZ
    R3_as = np.array(R3_as) * RATE_SCALE_KHZ
    As_pd_hist = np.array(As_pd_hist)
    As_dqn_hist = np.array(As_dqn_hist)
    L_tt_as_const = np.array(L_tt_as_const)
    L_tt_as_pd    = np.array(L_tt_as_pd)
    L_tt_as_dqn   = np.array(L_tt_as_dqn)
    L_aa_as_const = np.array(L_aa_as_const)
    L_aa_as_pd    = np.array(L_aa_as_pd)
    L_aa_as_dqn   = np.array(L_aa_as_dqn)

    CONST_STYLE = dict(linestyle="--", linewidth=2.8, alpha=0.85, zorder=2)
    PD_STYLE    = dict(linestyle="-",  linewidth=2.4, alpha=0.90, zorder=3)

    # DQN: thick + custom dash pattern + markers
    DQN_STYLE   = dict(
        linestyle=(0, (8, 2, 2, 2)),   # long dash, gap, short dash, gap
        linewidth=3.2,
        marker="o",
        markersize=4,
        markevery=6,                  # marker every ~6 points
        alpha=0.95,
        zorder=5,
    )
    time = np.linspace(0, 1, len(R1_ht))

    # ------------------------- reward plots -------------------------
    rewards_ht = np.asarray(rewards_ht, dtype=np.float32)
    rewards_as = np.asarray(rewards_as, dtype=np.float32)

    def moving_avg_nan(x, w=5):
        x = np.asarray(x, dtype=np.float32)
        m = np.isfinite(x).astype(np.float32)
        x0 = np.nan_to_num(x, nan=0.0)
        k = np.ones(w, dtype=np.float32)
        num = np.convolve(x0, k, mode="same")
        den = np.convolve(m,  k, mode="same")
        return num / np.maximum(den, 1e-8)



    # HT reward vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, rewards_ht, linewidth=1.2, alpha=0.35, label="HT reward (per chunk)")
    ax.plot(time, moving_avg_nan(rewards_ht, w=5), linewidth=2.2, label="HT reward (moving avg)")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Reward", loc="center")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "reward_ht_pidData_dqn_feature"))
    plt.close(fig)
    
    time_as = np.linspace(0, 1, len(R1_as))
    # AS reward vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_as, rewards_as, linewidth=1.2, alpha=0.35, label="AS reward (per chunk)")
    ax.plot(time_as, moving_avg_nan(rewards_as, w=5), linewidth=2.2, label="AS reward (moving avg)")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Reward", loc="center")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "reward_as_pidData_dqn_feature"))
    plt.close(fig)





    plot_rate_with_tolerance(
        time, R1_ht, R2_ht, R3_ht,
        outbase=outdir / "bht_rate_pidData_dqn",
        run_label=run_label,
        legend_title="HT Trigger",
        ylim=(0, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        const_style=dict(color="tab:blue", **CONST_STYLE),
        pd_style=dict(color="mediumblue", **PD_STYLE),
        dqn_style=dict(color="tab:purple", **DQN_STYLE),
        # pass your functions from utils import
        add_cms_header=add_cms_header,
        save_pdf_png=save_png,
    )

    # ------------------------- common styles -------------------------
    styles = {
        "Constant": CONST_STYLE,
        "PD":       PD_STYLE,
        "DQN":      DQN_STYLE,
    }


    # =========================================================
    # HT plots
    # =========================================================
    # (2) HT cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, Ht_pd_hist,  color="mediumblue", linewidth=2.0, label="PD Controller")
    ax.plot(time, Ht_dqn_hist, color="tab:purple", label="DQN", **DQN_STYLE)
    ax.axhline(y=fixed_Ht_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_Ht_cut")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Ht_cut [GeV]", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="HT Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "ht_cut_pidData_dqn"))
    plt.close(fig)

    # (3) HT cumulative eff (relative to t0)
    tt_c_const = cummean(L_tt_ht_const)
    tt_c_pd    = cummean(L_tt_ht_pd)
    tt_c_dqn   = cummean(L_tt_ht_dqn)
    aa_c_const = cummean(L_aa_ht_const)
    aa_c_pd    = cummean(L_aa_ht_pd)
    aa_c_dqn   = cummean(L_aa_ht_dqn)

    colors_ht = {"ttbar": "goldenrod", "HToAATo4B": "seagreen"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(tt_c_const), color=colors_ht["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={tt_c_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(aa_c_const), color=colors_ht["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={aa_c_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(tt_c_pd), color=colors_ht["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={tt_c_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(aa_c_pd), color=colors_ht["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={aa_c_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(tt_c_dqn), color=colors_ht["ttbar"],
            label=fr"DQN, ttbar ($\epsilon[t_0]={tt_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time, rel_to_t0(aa_c_dqn), color=colors_ht["HToAATo4B"],
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={aa_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Cumulative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.5, 2.5)
    ax.legend(title="HT Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "sht_rate_pidData2data_dqn"))
    plt.close(fig)

    # (4) HT local eff (relative to t0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(L_tt_ht_const), color=colors_ht["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={L_tt_ht_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_ht_const), color=colors_ht["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={L_aa_ht_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_tt_ht_pd), color=colors_ht["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={L_tt_ht_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_ht_pd), color=colors_ht["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={L_aa_ht_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_tt_ht_dqn), color=colors_ht["ttbar"], 
            label=fr"DQN, ttbar ($\epsilon[t_0]={L_tt_ht_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time, rel_to_t0(L_aa_ht_dqn), color=colors_ht["HToAATo4B"], 
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={L_aa_ht_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.0, 2.5)
    ax.legend(title="HT Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "L_sht_rate_pidData2data_dqn"))
    plt.close(fig)

    # (5) HT loss
    if losses_ht:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses_ht)), losses_ht, linewidth=1.5)
        ax.set_title("HT DQN training loss")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "dqn_loss_ht"))
        plt.close(fig)

    # =========================================================
    # AD plots
    # =========================================================
    time_as = np.linspace(0, 1, len(R1_as))
    plot_rate_with_tolerance(
        time_as, R1_as, R2_as, R3_as,
        outbase=outdir / "bas_rate_pidData_dqn",
        run_label=run_label,
        legend_title="AD Trigger",
        ylim=(0, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        const_style=dict(color="tab:blue", **CONST_STYLE),
        pd_style=dict(color="mediumblue", **PD_STYLE),
        dqn_style=dict(color="tab:purple", **DQN_STYLE),
        add_cms_header=add_cms_header,
        save_pdf_png=save_png,
    )

    # (A2) AS cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, As_pd_hist,  color="mediumblue", linewidth=2.0, label="PD Controller")
    ax.plot(time_as, As_dqn_hist, color="tab:purple", linewidth=2.0, label="DQN")
    ax.axhline(y=fixed_AS_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_AS_cut")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Anomaly Score Cut", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="AD Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "as_cut_pidData_dqn"))
    plt.close(fig)

    # (A3) AD cumulative eff (relative to t0)
    tt_c_const = cummean(L_tt_as_const)
    tt_c_pd    = cummean(L_tt_as_pd)
    tt_c_dqn   = cummean(L_tt_as_dqn)
    aa_c_const = cummean(L_aa_as_const)
    aa_c_pd    = cummean(L_aa_as_pd)
    aa_c_dqn   = cummean(L_aa_as_dqn)

    colors_ad = {"ttbar": "goldenrod", "HToAATo4B": "limegreen"}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, rel_to_t0(tt_c_const), color=colors_ad["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={tt_c_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(aa_c_const), color=colors_ad["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={aa_c_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(tt_c_pd), color=colors_ad["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={tt_c_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(aa_c_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={aa_c_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(tt_c_dqn), color=colors_ad["ttbar"],
            label=fr"DQN, ttbar ($\epsilon[t_0]={tt_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time_as, rel_to_t0(aa_c_dqn), color=colors_ad["HToAATo4B"], 
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={aa_c_dqn[0]:.2f}\%$)", **DQN_STYLE)

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Cumulative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.5, 2.5)
    ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "sas_rate_pidData2data_dqn"))
    plt.close(fig)

    # (A4) AD local eff (relative to t0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, rel_to_t0(L_tt_as_const), color=colors_ad["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={L_tt_as_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_const), color=colors_ad["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={L_aa_as_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_tt_as_pd), color=colors_ad["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={L_tt_as_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={L_aa_as_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_tt_as_dqn), color=colors_ad["ttbar"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, ttbar ($\epsilon[t_0]={L_tt_as_dqn[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_dqn), color=colors_ad["HToAATo4B"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={L_aa_as_dqn[0]:.2f}\%$)")

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.5, 2.5)
    ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "L_sas_rate_pidData2data_dqn"))
    plt.close(fig)

    # (A5) AS loss
    if losses_as:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses_as)), losses_as, linewidth=1.5)
        ax.set_title("AD DQN training loss")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "dqn_loss_as"))
        plt.close(fig)


    # =========================================================
    # Extra paper plots + summary tables (PD vs DQN baseline)
    # =========================================================
    plots_dir = outdir / "extra_plots"
    tables_dir = outdir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    TARGET_PCT = float(target)               # 0.25 (percent)
    TOL_PCT = float(tol)                     # 0.02 (percent)
    TARGET_KHZ = TARGET_PCT * RATE_SCALE_KHZ
    TOL_KHZ = TOL_PCT * RATE_SCALE_KHZ

    # ---------------------------------------------------------
    # Build percent-rate arrays (since R*_ht/as are in kHz now)
    # ---------------------------------------------------------
    r_const_ht_pct = R1_ht / RATE_SCALE_KHZ
    r_pd_ht_pct    = R2_ht / RATE_SCALE_KHZ
    r_dqn_ht_pct   = R3_ht / RATE_SCALE_KHZ

    r_const_as_pct = R1_as / RATE_SCALE_KHZ
    r_pd_as_pct    = R2_as / RATE_SCALE_KHZ
    r_dqn_as_pct   = R3_as / RATE_SCALE_KHZ

    # Constant cut arrays (for jitter metrics)
    Ht_const_hist = np.full_like(Ht_pd_hist, fixed_Ht_cut, dtype=np.float64)
    As_const_hist = np.full_like(As_pd_hist, fixed_AS_cut, dtype=np.float64)

    # ---------------- helpers ----------------
    def ecdf(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.array([]), np.array([])
        x = np.sort(x)
        y = (np.arange(1, x.size + 1) / x.size)
        return x, y
    
    def summarize_metrics(r_pct, s_tt, s_aa, cut, target_pct=0.25, tol_pct=0.02):
        r = np.asarray(r_pct, dtype=np.float64)
        inband = np.abs(r - target_pct) <= tol_pct

        def safe_mean(x, m):
            x = np.asarray(x, dtype=np.float64)
            return float(np.mean(x[m])) if np.any(m) else np.nan

        err = r - target_pct
        out = {}
        out["mae"] = float(np.mean(np.abs(err)))
        out["rmse"] = float(np.sqrt(np.mean(err**2)))
        out["p95_abs_err"] = float(np.percentile(np.abs(err), 95))
        out["inband_frac"] = float(np.mean(inband))
        out["upper_viol_frac"] = float(np.mean(r > (target_pct + tol_pct)))
        out["lower_viol_frac"] = float(np.mean(r < (target_pct - tol_pct)))
        out["viol_mag"] = float(np.mean(np.maximum(0.0, np.abs(err) - tol_pct)))

        c = np.asarray(cut, dtype=np.float64)
        dc = np.diff(c) if c.size >= 2 else np.array([], dtype=np.float64)
        out["cut_TV"] = float(np.sum(np.abs(dc))) if dc.size else 0.0
        out["cut_step_rms"] = float(np.sqrt(np.mean(dc**2))) if dc.size else 0.0
        out["cut_step_max"] = float(np.max(np.abs(dc))) if dc.size else 0.0

        out["tt_inband"] = safe_mean(s_tt, inband)
        out["aa_inband"] = safe_mean(s_aa, inband)
        out["score_50_50"] = safe_mean(0.5*np.asarray(s_tt) + 0.5*np.asarray(s_aa), inband)
        out["score_80_AA"] = safe_mean(0.2*np.asarray(s_tt) + 0.8*np.asarray(s_aa), inband)
        return out

    def write_tables(rows, csv_path: Path, tex_path: Path):
        # CSV
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        # LaTeX (simple Overleaf-friendly)
        cols = [
            "Trigger","Method",
            "mae","rmse","p95_abs_err",
            "inband_frac","upper_viol_frac","lower_viol_frac","viol_mag",
            "cut_TV","cut_step_rms","cut_step_max",
            "tt_inband","aa_inband","score_50_50","score_80_AA",
        ]
        def fmt(v):
            if isinstance(v, (float, np.floating)):
                return f"{v:.4g}"
            return str(v)

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{llrrrrrrrrrrrrrr}")
        lines.append(r"\hline")
        lines.append(r"Trigger & Method & MAE & RMSE & P95$|e|$ & InBand & UpViol & LoViol & ViolMag & TV & StepRMS & StepMax & TT & AA & 50/50 & 80/20 \\")
        lines.append(r"\hline")
        for r in rows:
            rr = {k: r.get(k, "") for k in cols}
            lines.append(
                f"{fmt(rr['Trigger'])} & {fmt(rr['Method'])} & "
                f"{fmt(rr['mae'])} & {fmt(rr['rmse'])} & {fmt(rr['p95_abs_err'])} & "
                f"{fmt(rr['inband_frac'])} & {fmt(rr['upper_viol_frac'])} & {fmt(rr['lower_viol_frac'])} & {fmt(rr['viol_mag'])} & "
                f"{fmt(rr['cut_TV'])} & {fmt(rr['cut_step_rms'])} & {fmt(rr['cut_step_max'])} & "
                f"{fmt(rr['tt_inband'])} & {fmt(rr['aa_inband'])} & {fmt(rr['score_50_50'])} & {fmt(rr['score_80_AA'])} \\\\"
            )
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{PD vs DQN summary. Rates are evaluated in percent units (target=0.25, tol=0.02).}")
        lines.append(r"\label{tab:pd_vs_dqn_summary}")
        lines.append(r"\end{table}")

        with open(tex_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _save(fig, outbase: Path):
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outbase))
        plt.close(fig)

    # ---------------- build rows ----------------
    sum_const_ht = summarize_metrics(r_const_ht_pct, L_tt_ht_const, L_aa_ht_const, Ht_const_hist, TARGET_PCT, TOL_PCT)
    sum_pd_ht    = summarize_metrics(r_pd_ht_pct,    L_tt_ht_pd,    L_aa_ht_pd,    Ht_pd_hist,   TARGET_PCT, TOL_PCT)
    sum_dqn_ht   = summarize_metrics(r_dqn_ht_pct,   L_tt_ht_dqn,   L_aa_ht_dqn,   Ht_dqn_hist,  TARGET_PCT, TOL_PCT)

    sum_const_as = summarize_metrics(r_const_as_pct, L_tt_as_const, L_aa_as_const, As_const_hist, TARGET_PCT, TOL_PCT)
    sum_pd_as    = summarize_metrics(r_pd_as_pct,    L_tt_as_pd,    L_aa_as_pd,    As_pd_hist,   TARGET_PCT, TOL_PCT)
    sum_dqn_as   = summarize_metrics(r_dqn_as_pct,   L_tt_as_dqn,   L_aa_as_dqn,   As_dqn_hist,  TARGET_PCT, TOL_PCT)

    
    rows = []
    def add_row(trigger, method, d):
        r = {"Trigger": trigger, "Method": method}
        r.update(d)
        rows.append(r)

    add_row("HT", "Constant", sum_const_ht)
    add_row("HT", "PD",       sum_pd_ht)
    add_row("HT", "DQN",      sum_dqn_ht)
    add_row("AS", "Constant", sum_const_as)
    add_row("AS", "PD",       sum_pd_as)
    add_row("AS", "DQN",      sum_dqn_as)

    # Save tables here:
    #   outdir/tables/pd_vs_dqn_summary.csv
    #   outdir/tables/pd_vs_dqn_summary.tex
    write_tables(
        rows,
        csv_path=tables_dir / "pd_vs_dqn_summary.csv",
        tex_path=tables_dir / "pd_vs_dqn_summary.tex",
    )
    print(f"[OK] wrote {tables_dir/'pd_vs_dqn_summary.csv'}")
    print(f"[OK] wrote {tables_dir/'pd_vs_dqn_summary.tex'}")

    # ---------------------------------------------------------
    # Extra Plot 1: CDF of |rate error| (kHz)  (HT + AS)
    # ---------------------------------------------------------
    def plot_cdf_abs_err(r_khz_pd, r_khz_dqn, outpath: Path, title: str):
        e_pd  = np.abs(np.asarray(r_khz_pd)  - TARGET_KHZ)
        e_dqn = np.abs(np.asarray(r_khz_dqn) - TARGET_KHZ)
        x1, y1 = ecdf(e_pd)
        x2, y2 = ecdf(e_dqn)

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.plot(x1, y1, linewidth=2.2, label="PD")
        ax.plot(x2, y2, linewidth=2.2, label="DQN")
        ax.axvline(TOL_KHZ, linestyle="--", linewidth=1.6, label=f"Tolerance = {TOL_KHZ:.1f} kHz")
        ax.set_xlabel(r"$|r - r^*|$  [kHz]")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        _save(fig, outpath)

    plot_cdf_abs_err(R2_ht, R3_ht, plots_dir / "cdf_abs_err_ht", "HT: CDF of absolute background-rate error")
    plot_cdf_abs_err(R2_as, R3_as, plots_dir / "cdf_abs_err_as", "AS: CDF of absolute background-rate error")

    # ---------------------------------------------------------
    # Extra Plot 2: In-band efficiency bars (PD vs DQN)  (HT + AS)
    # ---------------------------------------------------------
    def plot_inband_bars(sum_pd, sum_dqn, outpath: Path, title: str):
        labels = ["ttbar", "HToAATo4B"]
        pd_vals = [sum_pd["tt_inband"], sum_pd["aa_inband"]]
        dqn_vals = [sum_dqn["tt_inband"], sum_dqn["aa_inband"]]

        x = np.arange(len(labels))
        w = 0.35

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.bar(x - w/2, pd_vals, width=w, label="PD")
        ax.bar(x + w/2, dqn_vals, width=w, label="DQN")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Signal efficiency (mean, in-band)")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        _save(fig, outpath)

    plot_inband_bars(sum_pd_ht, sum_dqn_ht, plots_dir / "inband_eff_bars_ht", "HT: in-band mean efficiency (PD vs DQN)")
    plot_inband_bars(sum_pd_as, sum_dqn_as, plots_dir / "inband_eff_bars_as", "AS: in-band mean efficiency (PD vs DQN)")

    # ---------------------------------------------------------
    # Extra Plot 3: Cut-step magnitude histogram (jitter) (PD vs DQN)
    # ---------------------------------------------------------
    def plot_cut_step_hist(cut_pd, cut_dqn, outbase: Path, title: str, xlabel: str):
        dp = np.diff(np.asarray(cut_pd, dtype=np.float64))
        dd = np.diff(np.asarray(cut_dqn, dtype=np.float64))
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.hist(np.abs(dp), bins=30, alpha=0.55, label="PD")
        ax.hist(np.abs(dd), bins=30, alpha=0.55, label="DQN")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", frameon=True)
        _save(fig, outbase)

    plot_cut_step_hist(Ht_pd_hist, Ht_dqn_hist, plots_dir / "cut_step_hist_ht",
                       "HT: |Δ cut| distribution (jitter)", xlabel=r"$|\Delta Ht\_cut|$ [GeV]")
    plot_cut_step_hist(As_pd_hist, As_dqn_hist, plots_dir / "cut_step_hist_as",
                       "AS: |Δ cut| distribution (jitter)", xlabel=r"$|\Delta AS\_cut|$")

    # ---------------------------------------------------------
    # Extra Plot 4: Running in-band fraction over time (PD vs DQN)
    # ---------------------------------------------------------
    def running_mean_bool(mask, w=5):
        m = np.asarray(mask, dtype=np.float64)
        k = np.ones(w, dtype=np.float64)
        return np.convolve(m, k, mode="same") / np.convolve(np.ones_like(m), k, mode="same")

    def plot_running_inband(r_khz_pd, r_khz_dqn, outbase: Path, title: str, w=5):
        rpd = np.asarray(r_khz_pd, dtype=np.float64) / RATE_SCALE_KHZ  
        rdq = np.asarray(r_khz_dqn, dtype=np.float64) / RATE_SCALE_KHZ
        in_pd = np.abs(rpd - TARGET_PCT) <= TOL_PCT
        in_dq = np.abs(rdq - TARGET_PCT) <= TOL_PCT

        tgrid = np.linspace(0, 1, len(rpd))
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.plot(tgrid, running_mean_bool(in_pd, w=w), linewidth=2.2, label=f"PD (w={w})")
        ax.plot(tgrid, running_mean_bool(in_dq, w=w), linewidth=2.2, label=f"DQN (w={w})")
        ax.set_xlabel("Time (Fraction of Run)")
        ax.set_ylabel("Running in-band fraction")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        _save(fig, outbase)


    plot_running_inband(R2_ht, R3_ht, plots_dir / "running_inband_ht", "HT: running in-band fraction (PD vs DQN)", w = 5)
    plot_running_inband(R2_as, R3_as, plots_dir / "running_inband_as", "AS: running in-band fraction (PD vs DQN)", w = 5)

    print("\nSaved to:", outdir)
    for p in sorted(outdir.glob("*.png")):
        print(" -", p.name)
    

    # Example usage for HT (replace with AS by passing Bas/Tas/Aas and cut hists)
    t_mid, auc_tt_pd, auc_tt_dqn, auc_aa_pd, auc_aa_dqn = compute_auroc_windows_separate(
        start_event=start_event,
        window_events=chunk_size,              # chunk size
        update_chunk_size=chunk_size,     # controller update interval (your big chunk)
        matched_by_index=matched_by_index,
        Bnpv=Bnpv, Tnpv=Tnpv, Anpv=Anpv,
        Bx=Bht, Tx=Tht, Ax=Aht,
        cut_hist_pd=Ht_pd_hist,
        cut_hist_dqn=Ht_dqn_hist,
        max_n=200000,
        seed=SEED,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_mid, auc_tt_pd,  label="AUROC TT vs BKG (PD)",  linewidth=2.0)
    ax.plot(t_mid, auc_tt_dqn, label="AUROC TT vs BKG (DQN)", linewidth=2.0)
    ax.plot(t_mid, auc_aa_pd,  label="AUROC AA vs BKG (PD)",  linewidth=2.0)
    ax.plot(t_mid, auc_aa_dqn, label="AUROC AA vs BKG (DQN)", linewidth=2.0)
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "auroc_tt_aa_vs_time_ht"))
    plt.close(fig)


    # Operating point (accept if x > cut) — the anomaly decision
    t_mid2, fpr_pd, fpr_dqn, tpr_tt_pd, tpr_tt_dqn, tpr_aa_pd, tpr_aa_dqn = compute_operating_point_windows_separate(
        start_event=start_event,
        window_events=50000,
        update_chunk_size=chunk_size,
        matched_by_index=matched_by_index,
        Bnpv=Bnpv, Tnpv=Tnpv, Anpv=Anpv,
        Bx=Bht, Tx=Tht, Ax=Aht,
        cut_hist_pd=Ht_pd_hist,
        cut_hist_dqn=Ht_dqn_hist,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_mid2, fpr_pd,  label="BKG accept fraction (PD)",  linewidth=2.0)
    ax.plot(t_mid2, fpr_dqn, label="BKG accept fraction (DQN)", linewidth=2.0)
    ax.plot(t_mid2, tpr_tt_pd,  label="TT accept fraction (PD)",  linewidth=2.0)
    ax.plot(t_mid2, tpr_tt_dqn, label="TT accept fraction (DQN)", linewidth=2.0)
    ax.plot(t_mid2, tpr_aa_pd,  label="AA accept fraction (PD)",  linewidth=2.0)
    ax.plot(t_mid2, tpr_aa_dqn, label="AA accept fraction (DQN)", linewidth=2.0)
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Accept fraction at margin > 0")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "operating_point_accept_frac_vs_time_ht"))
    plt.close(fig)

if __name__ == "__main__":
    main()