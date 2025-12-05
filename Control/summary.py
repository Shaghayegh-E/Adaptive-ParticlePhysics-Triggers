# summaryPlots.py
# putting the summary binning + plotting
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Callable

from .mc_singletrigger_io import load_trigger_food_summary_plots
from .metrics import rate_above_threshold, comp_costs
try:
    import atlas_mpl_style as aplt
    aplt.use_atlas_style()
except Exception:
    pass


# ---------- small utilities (binners) ----------

def average_perf_bins(perf: list[float], cost: list[float], n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(perf))
    bins = np.array_split(idx, n_bins)
    avg_p, avg_c = [], []
    for b in bins:
        avg_p.append(np.mean([perf[i] for i in b]))
        avg_c.append(np.mean([cost[i] for i in b]))
    return np.asarray(avg_p), np.asarray(avg_c)

def average_pair_over_bins(x_list: list[float], y_list: list[float], n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(x_list))
    bins = np.array_split(idx, n_bins)
    x_avg, y_avg = [], []
    for b in bins:
        x_avg.append(np.mean([x_list[i] for i in b]))
        y_avg.append(np.mean([y_list[i] for i in b]))
    return np.asarray(x_avg), np.asarray(y_avg)


# ---------- trigger eval primitive (no grid scan) ----------

def eval_fixed_cuts(
    bht: np.ndarray, bas: np.ndarray, bnjets: np.ndarray,
    sht1: np.ndarray, sas1: np.ndarray,
    sht2: np.ndarray, sas2: np.ndarray,
    ht_cut: float, as_cut: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Returns: r_b, r_s, r_bht, r_bas, r_sht, r_sas, b_Ecost, b_Tcost
    (keeps the same semantics from your Trigger() in summaryPlots.py)
    """
    s1_ht = sht1 >= ht_cut
    s2_ht = sht2 >= ht_cut
    s1_as = sas1 >= as_cut
    s2_as = sas2 >= as_cut
    b_ht = bht >= ht_cut
    b_as = bas >= as_cut

    s1_evt = s1_ht.sum() + s1_as.sum() - (s1_ht & s1_as).sum()
    s2_evt = s2_ht.sum() + s2_as.sum() - (s2_ht & s2_as).sum()
    b_evt  = b_ht.sum()  + b_as.sum()  - (b_ht  & b_as).sum()

    r_s = 100.0 * (s1_evt + s2_evt) / (sht1.size + sht2.size + 1e-10)
    r_b = 100.0 * b_evt / (bht.size + 1e-10)

    r_sht = 100.0 * (s1_ht.sum() + s2_ht.sum()) / (sht1.size + sht2.size + 1e-10)
    r_bht = 100.0 * b_ht.sum() / (bht.size + 1e-10)
    r_sas = 100.0 * (s1_as.sum() + s2_as.sum()) / (sas1.size + sas2.size + 1e-10)
    r_bas = 100.0 * b_as.sum() / (bas.size + 1e-10)

    # comp costs (same as in your Trigger() impl)
    ht_w, as_w = 1.0, 4.0
    b_both = (b_ht & b_as).sum()
    b_Tcost = (ht_w * (b_ht.sum() - b_both) + as_w * b_as.sum()) / (b_evt + 1e-10)
    b_Ecost = (((b_ht | b_as) * bnjets).sum()) / (b_evt + 1e-10)

    return r_b, r_s, r_bht, r_bas, r_sht, r_sas, float(b_Ecost), float(b_Tcost)



def V1(bht, sht1, sht2, bas, sas1, sas2, bnjets):
    """
    V1_Trigger_Agent from A: same body, bnjets is unused but kept
    for a consistent API.
    """
    max1 = np.percentile(sht1, 99.99)
    max2 = np.percentile(sht2, 99.99)
    MAX = max(max1, max2)
    MAX = np.percentile(bht, 99.99)

    ht_vals = np.linspace(np.percentile(bht, 0.01), MAX, 100)

    max1 = np.percentile(sas1, 99.99)
    max2 = np.percentile(sas2, 99.99)
    MAX = max(max1, max2)
    MAX = np.percentile(bas, 99.999)
    as_vals = np.linspace(np.percentile(bas, 0.01), MAX, 100)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing="ij")

    # signal
    s1_accepted_ht = sht1[:, None, None] >= HT[None, :, :]
    s1_accepted_as = sas1[:, None, None] >= AS[None, :, :]
    s2_accepted_ht = sht2[:, None, None] >= HT[None, :, :]
    s2_accepted_as = sas2[:, None, None] >= AS[None, :, :]

    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s1_as_count = s1_accepted_as.sum(axis=0)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)

    s2_ht_count = s2_accepted_ht.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)

    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count

    r1_s = 100.0 * s1_accepted_events / sht1.shape[0]
    r2_s = 100.0 * s2_accepted_events / sht2.shape[0]

    total_s_events = s1_accepted_events + s2_accepted_events
    total_s_rate = 100.0 * total_s_events / (sht1.shape[0] + sht2.shape[0])

    # background
    b_accepted_ht = bht[:, None, None] >= HT[None, :, :]
    b_accepted_as = bas[:, None, None] >= AS[None, :, :]
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)

    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100.0 * b_accepted_events / bht.shape[0]

    b_overlap = 100.0 * (b_both_count + 1e-10) / (b_accepted_events + 1e-10)
    s1_overlap = 100.0 * (s1_both_count + 1e-10) / (s1_accepted_events + 1e-10)
    s2_overlap = 100.0 * (s2_both_count + 1e-10) / (s2_accepted_events + 1e-10)

    r_bht = 100.0 * b_ht_count / bht.shape[0]
    r_bas = 100.0 * b_as_count / bht.shape[0]
    r1_sht = 100.0 * s1_ht_count / sht1.shape[0]
    r1_sas = 100.0 * s1_as_count / sas1.shape[0]
    r2_sht = 100.0 * s2_ht_count / sht2.shape[0]
    r2_sas = 100.0 * s2_as_count / sas2.shape[0]

    t_b = 0.25
    a0, a1 = 100.0, 0.2
    cost = a0 * np.abs(r_b - t_b) + a1 * np.abs(total_s_rate - 100.0)
    log_cost = np.log10(cost.clip(min=1e-10))

    return (
        log_cost,
        r_b,
        r1_s,
        r2_s,
        b_overlap,
        s1_overlap,
        s2_overlap,
        r_bht,
        r_bas,
        r1_sht,
        r2_sht,
        r1_sas,
        r2_sas,
        HT,
        AS,
    )

def V2(bht, sht1, sht2, bas, sas1, sas2, bnjets):
    """
    V2_Trigger_Agent from A.
    """
    max1 = np.percentile(sht1, 99.99)
    max2 = np.percentile(sht2, 99.99)
    MAX = max(max1, max2)
    MAX = np.percentile(bht, 99.99)
    ht_vals = np.linspace(np.percentile(bht, 0.01), MAX, 100)

    max1 = np.percentile(sas1, 99.99)
    max2 = np.percentile(sas2, 99.99)
    MAX = max(max1, max2)
    MAX = np.percentile(bas, 99.99)
    as_vals = np.linspace(np.percentile(bas, 0.01), MAX, 100)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing="ij")

    s1_accepted_ht = sht1[:, None, None] >= HT[None, :, :]
    s1_accepted_as = sas1[:, None, None] >= AS[None, :, :]
    s2_accepted_ht = sht2[:, None, None] >= HT[None, :, :]
    s2_accepted_as = sas2[:, None, None] >= AS[None, :, :]

    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s1_as_count = s1_accepted_as.sum(axis=0)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)

    s2_ht_count = s2_accepted_ht.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)

    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    r1_s = 100.0 * s1_accepted_events / sht1.shape[0]
    r2_s = 100.0 * s2_accepted_events / sht2.shape[0]

    total_s_events = s1_accepted_events + s2_accepted_events
    total_s_rate = 100.0 * total_s_events / (sht1.shape[0] + sht2.shape[0])

    b_accepted_ht = bht[:, None, None] >= HT[None, :, :]
    b_accepted_as = bas[:, None, None] >= AS[None, :, :]
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    b_accepted_events = b_ht_count + b_as_count - b_both_count

    r_b = 100.0 * b_accepted_events / bht.shape[0]
    r_as_ex = 100.0 * (b_as_count - b_both_count) / bht.shape[0]

    b_overlap = 100.0 * (b_both_count + 1e-10) / (b_accepted_events + 1e-10)
    s1_overlap = 100.0 * (s1_both_count + 1e-10) / (s1_accepted_events + 1e-10)
    s2_overlap = 100.0 * (s2_both_count + 1e-10) / (s2_accepted_events + 1e-10)

    r_bht = 100.0 * b_ht_count / bht.shape[0]
    r_bas = 100.0 * b_as_count / bht.shape[0]
    r1_sht = 100.0 * s1_ht_count / sht1.shape[0]
    r1_sas = 100.0 * s1_as_count / sas1.shape[0]
    r2_sht = 100.0 * s2_ht_count / sht2.shape[0]
    r2_sas = 100.0 * s2_as_count / sas2.shape[0]

    a0, a1, a2 = 100.0, 0.2, 25.0
    t_b = 0.25
    percentage = 0.3
    cost = (
        a0 * np.abs(r_b - t_b)
        + a1 * np.abs(r1_s - 90.0)
        + a2 * np.abs(r_as_ex - percentage * t_b)
    )
    log_cost = np.log10(cost.clip(min=1e-10))

    return (
        log_cost,
        r_b,
        r1_s,
        r2_s,
        b_overlap,
        s1_overlap,
        s2_overlap,
        r_bht,
        r_bas,
        r1_sht,
        r2_sht,
        r1_sas,
        r2_sas,
        HT,
        AS,
    )


def V3(bht, sht1, sht2, bas, sas1, sas2, bnjets):
    """
    V3_Trigger_Agent from A.
    """
    MAX = np.percentile(bht, 99.99)
    ht_vals = np.linspace(np.percentile(bht, 0.01), MAX, 100)
    MAX = np.percentile(bas, 99.99)
    as_vals = np.linspace(np.percentile(bas, 0.01), MAX, 100)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing="ij")

    s1_accepted_ht = sht1[:, None, None] >= HT[None, :, :]
    s1_accepted_as = sas1[:, None, None] >= AS[None, :, :]
    s2_accepted_ht = sht2[:, None, None] >= HT[None, :, :]
    s2_accepted_as = sas2[:, None, None] >= AS[None, :, :]

    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s1_as_count = s1_accepted_as.sum(axis=0)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)

    s2_ht_count = s2_accepted_ht.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)

    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count

    total_s_events = s1_accepted_events + s2_accepted_events
    r_s = 100.0 * total_s_events / (sht1.shape[0] + sht2.shape[0])

    b_accepted_ht = bht[:, None, None] >= HT[None, :, :]
    b_accepted_as = bas[:, None, None] >= AS[None, :, :]
    b_accepted = b_accepted_ht | b_accepted_as

    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    b_accepted_events = b_ht_count + b_as_count - b_both_count

    r_b = 100.0 * b_accepted_events / bht.shape[0]

    b_overlap = 100.0 * (b_both_count + 1e-10) / (b_accepted_events + 1e-10)
    s1_overlap = 100.0 * (s1_both_count + 1e-10) / (s1_accepted_events + 1e-10)
    s2_overlap = 100.0 * (s2_both_count + 1e-10) / (s2_accepted_events + 1e-10)

    r_bht = 100.0 * b_ht_count / bht.shape[0]
    r_bas = 100.0 * b_as_count / bht.shape[0]
    r1_sht = 100.0 * s1_ht_count / sht1.shape[0]
    r1_sas = 100.0 * s1_as_count / sas1.shape[0]
    r2_sht = 100.0 * s2_ht_count / sht2.shape[0]
    r2_sas = 100.0 * s2_as_count / sas2.shape[0]

    a0, a1, a2, a3 = 100.0, 0.2, 1 / 0.5, 1 / 0.5
    t_b = 0.25

    b_Ecomp_cost, b_Tcomp_cost = comp_costs(
        b_accepted,
        b_accepted_events,
        b_ht_count,
        b_both_count,
        b_as_count,
        bnjets,
    )

    cost = (
        a0 * np.abs(r_b - t_b)
        + a1 * np.abs(r_s - 100.0)
        + a2 * np.maximum(b_Ecomp_cost - 5.5, 0)
        + a3 * np.maximum(b_Tcomp_cost - 2.0, 0)
    )
    log_cost = np.log10(cost.clip(min=1e-10))

    return (
        log_cost,
        r_b,
        r_s,
        r_bht,
        r_bas,
        r1_sht,
        r2_sht,
        r1_sas,
        r2_sas,
        b_overlap,
        s1_overlap,
        s2_overlap,
        HT,
        AS,
    )

def local_V1(
    bht,
    sht1,
    sht2,
    bas,
    sas1,
    sas2,
    bnjets,
    ht0,
    as0,
    ht_win=20.0,
    as_win=20.0,
    num=10,
):
    """
    V1_Trigger_localAgent from A.
    NOTE: The crucial part is the parameter order: bnjets, ht0, as0.
    """
    ht_min, ht_max = ht0 - ht_win, ht0 + ht_win
    as_min, as_max = as0 - as_win, as0 + as_win

    MAX = np.percentile(bht, 99.99)
    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num)

    MAX = np.percentile(bas, 99.99)
    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing="ij")

    s1_accepted_ht = sht1[:, None, None] >= HT[None, :, :]
    s1_accepted_as = sas1[:, None, None] >= AS[None, :, :]
    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s1_as_count = s1_accepted_as.sum(axis=0)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)

    s2_accepted_ht = sht2[:, None, None] >= HT[None, :, :]
    s2_accepted_as = sas2[:, None, None] >= AS[None, :, :]
    s2_ht_count = s2_accepted_ht.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)

    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count

    r_s = 100.0 * (s1_accepted_events + s2_accepted_events) / (
        sht1.shape[0] + sht2.shape[0]
    )

    b_accepted_ht = bht[:, None, None] >= HT[None, :, :]
    b_accepted_as = bas[:, None, None] >= AS[None, :, :]
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    b_accepted_events = b_ht_count + b_as_count - b_both_count

    r_b = 100.0 * b_accepted_events / bht.shape[0]
    r_bht = 100.0 * b_ht_count / bht.shape[0]
    r_bas = 100.0 * b_as_count / bht.shape[0]
    r_sht = 100.0 * (s1_ht_count + s2_ht_count) / (sht1.shape[0] + sht2.shape[0])
    r_sas = 100.0 * (s1_as_count + s2_as_count) / (sas1.shape[0] + sas2.shape[0])

    a0, a1 = 100.0, 0.2
    t_b = 0.25
    cost = a0 * np.abs(r_b - t_b) + a1 * np.abs(r_s - 100.0)
    log_cost = np.log10(cost.clip(min=1e-10))

    return log_cost, r_b, r_s, r_bht, r_bas, r_sht, r_sas, HT, AS


def local_V2(
    bht,
    sht1,
    sht2,
    bas,
    sas1,
    sas2,
    bnjets,
    ht0,
    as0,
    ht_win=20.0,
    as_win=20.0,
    num=10,
):
    """
    V2_Trigger_localAgent from A.
    """
    ht_min, ht_max = ht0 - ht_win, ht0 + ht_win
    as_min, as_max = as0 - as_win, as0 + as_win

    MAX = np.percentile(bht, 99.99)
    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num)

    MAX = np.percentile(bas, 99.99)
    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing="ij")

    s1_accepted_ht = sht1[:, None, None] >= HT[None, :, :]
    s1_accepted_as = sas1[:, None, None] >= AS[None, :, :]
    s2_accepted_ht = sht2[:, None, None] >= HT[None, :, :]
    s2_accepted_as = sas2[:, None, None] >= AS[None, :, :]

    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s1_as_count = s1_accepted_as.sum(axis=0)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)

    s2_ht_count = s2_accepted_ht.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)

    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count

    r1_s = 100.0 * s1_accepted_events / sht1.shape[0]
    total_s_events = s1_accepted_events + s2_accepted_events
    r_s = 100.0 * total_s_events / (sht1.shape[0] + sht2.shape[0])

    b_accepted_ht = bht[:, None, None] >= HT[None, :, :]
    b_accepted_as = bas[:, None, None] >= AS[None, :, :]
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    b_accepted_events = b_ht_count + b_as_count - b_both_count

    r_b = 100.0 * b_accepted_events / bht.shape[0]
    r_as_ex = 100.0 * (b_as_count - b_both_count) / bht.shape[0]

    r_bht = 100.0 * b_ht_count / bht.shape[0]
    r_bas = 100.0 * b_as_count / bht.shape[0]
    r_sht = 100.0 * (s1_ht_count + s2_ht_count) / (sht1.shape[0] + sht2.shape[0])
    r_sas = 100.0 * (s1_as_count + s2_as_count) / (sas1.shape[0] + sas2.shape[0])

    a0, a1, a2 = 100.0, 0.2, 25.0
    t_b = 0.25
    percentage = 0.3
    cost = (
        a0 * np.abs(r_b - t_b)
        + a1 * np.abs(r1_s - 90.0)
        + a2 * np.abs(r_as_ex - percentage * t_b)
    )
    log_cost = np.log10(cost.clip(min=1e-10))

    return log_cost, r_b, r_s, r_bht, r_bas, r_sht, r_sas, HT, AS


def local_V3(
    bht,
    sht1,
    sht2,
    bas,
    sas1,
    sas2,
    bnjets,
    ht0,
    as0,
    ht_win=20.0,
    as_win=20.0,
    num=10,
):
    """
    V3_Trigger_localAgent from A.
    """
    ht_min, ht_max = ht0 - ht_win, ht0 + ht_win
    as_min, as_max = as0 - as_win, as0 + as_win

    MAX = np.percentile(bht, 99.99)
    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num)
    MAX = np.percentile(bas, 99.99)
    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing="ij")

    s1_accepted_ht = sht1[:, None, None] >= HT[None, :, :]
    s1_accepted_as = sas1[:, None, None] >= AS[None, :, :]
    s2_accepted_ht = sht2[:, None, None] >= HT[None, :, :]
    s2_accepted_as = sas2[:, None, None] >= AS[None, :, :]

    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s1_as_count = s1_accepted_as.sum(axis=0)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)

    s2_ht_count = s2_accepted_ht.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)

    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count

    total_s_events = s1_accepted_events + s2_accepted_events
    r_s = 100.0 * total_s_events / (sht1.shape[0] + sht2.shape[0])

    b_accepted_ht = bht[:, None, None] >= HT[None, :, :]
    b_accepted_as = bas[:, None, None] >= AS[None, :, :]
    b_accepted = b_accepted_ht | b_accepted_as

    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    b_accepted_events = b_ht_count + b_as_count - b_both_count

    r_b = 100.0 * b_accepted_events / bht.shape[0]
    r_bht = 100.0 * b_ht_count / bht.shape[0]
    r_bas = 100.0 * b_as_count / bht.shape[0]
    r_sht = 100.0 * (s1_ht_count + s2_ht_count) / (sht1.shape[0] + sht2.shape[0])
    r_sas = 100.0 * (s1_as_count + s2_as_count) / (sas1.shape[0] + sas2.shape[0])

    a0, a1, a2, a3 = 100.0, 0.2, 1 / 0.5, 1 / 0.5
    t_b = 0.25

    b_Ecomp_cost, b_Tcomp_cost = comp_costs(
        b_accepted,
        b_accepted_events,
        b_ht_count,
        b_both_count,
        b_as_count,
        bnjets,
    )

    cost = (
        a0 * np.abs(r_b - t_b)
        + a1 * np.abs(r_s - 100.0)
        + a2 * np.maximum(b_Ecomp_cost - 5.5, 0)
        + a3 * np.maximum(b_Tcomp_cost - 2.0, 0)
    )
    log_cost = np.log10(cost.clip(min=1e-10))

    return log_cost, r_b, r_s, r_bht, r_bas, r_sht, r_sas, HT, AS

# ---------- summary runner (library entry) ----------

def build_initial_cuts_from_first_chunk(
    agent_funcs: dict[str, Callable],
    bht: np.ndarray, bas: np.ndarray, bnjets: np.ndarray,
    sht1: np.ndarray, sas1: np.ndarray,
    sht2: np.ndarray, sas2: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """
    For each agent, scan its grid once and take the (HT, AS) at min cost.
    """
    initial_cuts = {}
    for name, agent in agent_funcs.items():
        cost, *_ , HT, AS = agent(bht, sht1, sht2, bas, sas1, sas2, bnjets)
        i0, j0 = np.unravel_index(np.argmin(cost), cost.shape)
        initial_cuts[name] = (float(HT[i0, j0]), float(AS[i0, j0]))
    return initial_cuts


def run_summary_stream(
    path: str,
    start_offset: int = 500_000,
    chunk: int = 50_000,
    fixed_calib_len: int = 100_000,
) -> dict:
    """
    Replicates the logic in summaryPlots.py but using the modular agents & utils.
    Returns a dict keyed by case name, each holding lists of metrics.
    """
    sas1, sht1, snpv1, snjets, sas2, sht2, snpv2, snjets2, bas, bht, bnpv, bnjets = load_trigger_food_summary_plots(path)

    # (Using dim=4 scores everywhere here because the original summaryPlots.py read dim=4)
    # bas = D["mc_bkg_score04"]; bht = D["mc_bkg_ht"]; bnpv = D["mc_bkg_Npv"]; bnjets = D["mc_bkg_njets"]
    # sas1 = D["mc_tt_score04"]; sht1 = D["mc_tt_ht"]; snpv1 = D["tt_Npv"]
    # sas2 = D["mc_aa_score04"]; sht2 = D["mc_aa_ht"]; snpv2 = D["aa_Npv"]

    # stream subset
    bas = bas[start_offset:]; bht = bht[start_offset:]; bnpv = bnpv[start_offset:]; bnjets = bnjets[start_offset:]
    N = bnpv.size

    # cases storage
    cases_lists = {
        'Fixed Menu':        {'cost': [], 'absrb': [], 'rs': [], 'w0absrb': [], 'w1rs': [], 'Tcost': [], 'Ecost': []},
        'Standard':          {'cost': [], 'absrb': [], 'rs': [], 'w0absrb': [], 'w1rs': [], 'Tcost': [], 'Ecost': []},
        'Anomaly Focused':   {'cost': [], 'absrb': [], 'rs': [], 'w0absrb': [], 'w1rs': [], 'Tcost': [], 'Ecost': []},
        'Low-Comp Focused':  {'cost': [], 'absrb': [], 'rs': [], 'w0absrb': [], 'w1rs': [], 'Tcost': [], 'Ecost': []},
    }

    # initial grid (first chunk) to get cuts per agent
    init = slice(0, min(chunk, N))
    bht_i, bas_i, bnjets_i = bht[init], bas[init], bnjets[init]
    npv_i = bnpv[init]
    m1_i = (snpv1 >= npv_i.min()) & (snpv1 <= npv_i.max())
    m2_i = (snpv2 >= npv_i.min()) & (snpv2 <= npv_i.max())
    sht1_i, sas1_i = sht1[m1_i], sas1[m1_i]
    sht2_i, sas2_i = sht2[m2_i], sas2[m2_i]

    agent_for_init = {
        'Standard': V1,
        'Anomaly Focused': V2,
        'Low-Comp Focused': V3,
    }
    init_cuts = build_initial_cuts_from_first_chunk(agent_for_init, bht_i, bas_i, bnjets_i, sht1_i, sas1_i, sht2_i, sas2_i)

    # Fixed menu: use percentiles over the first fixed_calib_len
    calib = slice(0, min(fixed_calib_len, N))
    Ht_fixed = float(np.percentile(bht[calib], 99.8))
    AS_fixed = float(np.percentile(bas[calib], 99.9))

    # live cuts per case
    cuts = {
        'Standard':         {'Ht': init_cuts['Standard'][0],         'AS': init_cuts['Standard'][1]},
        'Anomaly Focused':  {'Ht': init_cuts['Anomaly Focused'][0],  'AS': init_cuts['Anomaly Focused'][1]},
        'Low-Comp Focused': {'Ht': init_cuts['Low-Comp Focused'][0], 'AS': init_cuts['Low-Comp Focused'][1]},
    }

    # local agents used for iterative updates
    local_agents = {
        'Standard': local_V1,
        'Anomaly Focused': local_V2,
        'Low-Comp Focused': local_V3,
    }

    # stream over chunks
    for I in range(0, N, chunk):
        idx = slice(I, min(I + chunk, N))
        bht_c, bas_c, bnjets_c, bnpv_c = bht[idx], bas[idx], bnjets[idx], bnpv[idx]
        npv_min, npv_max = float(bnpv_c.min()), float(bnpv_c.max())
        m1 = (snpv1 >= npv_min) & (snpv1 <= npv_max)
        m2 = (snpv2 >= npv_min) & (snpv2 <= npv_max)

        # Fixed Menu
        r_b, r_s, _, _, _, _, Ecost, Tcost = eval_fixed_cuts(
            bht_c, bas_c, bnjets_c, sht1[m1], sas1[m1], sht2[m2], sas2[m2], Ht_fixed, AS_fixed
        )
        cases_lists['Fixed Menu']['absrb'].append(abs(400.0 * r_b - 100.0))
        cases_lists['Fixed Menu']['w0absrb'].append(100.0 * abs(r_b - 0.25))
        cases_lists['Fixed Menu']['rs'].append(r_s)
        cases_lists['Fixed Menu']['w1rs'].append(0.2 * (100.0 - r_s))
        cases_lists['Fixed Menu']['Tcost'].append(Tcost)
        cases_lists['Fixed Menu']['Ecost'].append(Ecost)
        cases_lists['Fixed Menu']['cost'].append(Ecost + Tcost)

        # Adaptive cases (V1/V2/V3 local agents)
        for case, agent in local_agents.items():
            ht_cut = cuts[case]['Ht']; as_cut = cuts[case]['AS']
            r_b, r_s, _, _, _, _, Ecost, Tcost = eval_fixed_cuts(
                bht_c, bas_c, bnjets_c, sht1[m1], sas1[m1], sht2[m2], sas2[m2], ht_cut, as_cut
            )
            cases_lists[case]['absrb'].append(abs(400.0 * r_b - 100.0))
            cases_lists[case]['w0absrb'].append(100.0 * abs(r_b - 0.25))
            cases_lists[case]['rs'].append(r_s)
            cases_lists[case]['w1rs'].append(0.2 * (100.0 - r_s))
            cases_lists[case]['Tcost'].append(Tcost)
            cases_lists[case]['Ecost'].append(Ecost)
            cases_lists[case]['cost'].append(Ecost + Tcost)

            # local scan update
            cost_grid, *_ , HT, AS = agent(bht_c, sht1[m1], sht2[m2], bas_c, sas1[m1], sas2[m2], bnjets_c, ht_cut, as_cut)
            # agent(
            #     bht_c, sht1[m1], sht2[m2], bas_c, sas1[m1], sas2[m2], bnjets_c,
            #     ht_cut, as_cut
            # )
            ii, jj = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
            cuts[case]['Ht'], cuts[case]['AS'] = float(HT[ii, jj]), float(AS[ii, jj])

        print(f"Processed chunk starting at {I}")

    return cases_lists



# trigger/run_summary_plots.py
import argparse, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable



def plot_case_comparison1(cases_data, n_bins=10, save_path=None):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.ravel()

    titles  = [r'$(1-\epsilon)/\sigma_s$ vs $|r_b-r_t|/\sigma_b$',
               r'Total Computational Cost vs $|r_b-r_t|$',
               r'Total Computational Cost vs $\epsilon(\%)$',
               'Average Trigger Cost vs Average Event Cost']
    x_keys  = ['w0absrb', 'absrb', 'rs', 'Tcost']
    y_keys  = ['w1rs',    'cost',  'cost', 'Ecost']
    xlabels = [r'$|r_b-r_t|/\sigma_b$',
               r'$|r_b-r_t|$, $r_b =$ Background Rate(kHz)',
               r'$\epsilon$: Total Signal Efficiency $(\%)$',
               'Average Trigger Cost']
    ylabels = [r'$(1-\epsilon)/\sigma_s$',
               'Total Computational Cost',
               'Total Computational Cost',
               'Average Event Cost']

    markers = ['o', 's', 'P', 'D']
    base_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    legend_handles, legend_labels = [], []

    for c_idx, (case_name, data) in enumerate(cases_data.items()):
        base_color = mpl.colors.to_rgb(base_colors[c_idx % len(base_colors)])
        for p in range(4):
            x_vals, y_vals = average_pair_over_bins(data[x_keys[p]], data[y_keys[p]], n_bins=n_bins)
            if len(x_vals) < 2:
                continue
            points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            t_vals = np.linspace(0, 1, len(segments))
            colors = [(1 - t) * np.array(base_color) + t * np.ones(3) for t in t_vals]

            lc = LineCollection(segments, colors=colors, linewidth=min(2*(4-c_idx),4), alpha=1, label=case_name)
            axs[p].add_collection(lc)
            axs[p].plot(x_vals[0], y_vals[0], marker='*', color=base_color, markersize=12)
            axs[p].plot(x_vals[1:], y_vals[1:], marker=markers[c_idx % len(markers)],
                        linestyle='None', color=base_color, markersize=5)

            # legend “template” handle
            line, = axs[0].plot([], [], color=base_color, marker=markers[c_idx % len(markers)],
                                linestyle='-', linewidth=2)
            if p == 0:
                legend_handles.append(line); legend_labels.append(case_name)

    for p in range(4):
        axs[p].set_xlabel(xlabels[p], fontsize=15)
        axs[p].set_ylabel(ylabels[p], fontsize=15)
        axs[p].tick_params(axis='both', labelsize=15)
        axs[p].grid(True)
        yt = axs[p].get_yticks()
        axs[p].set_yticks(yt[1:])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.8, wspace=0.3, hspace=0.3)
    fig.suptitle("Summary Plots for Performance Comparison", fontsize=22, x=0.4, y=0.95)

    # global colorbar (time direction)
    sm = ScalarMappable(cmap='Greys', norm=plt.Normalize(0, 1)); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Relative Time (Dark → Light)', fontsize=18); cbar.set_ticks([])

    from matplotlib.lines import Line2D
    star_legend = Line2D([], [], marker='*', color='black', linestyle='None', markersize=10)
    legend_handles.append(star_legend); legend_labels.append("Start of the Run")
    fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(0.99, 0.92),
               fontsize=16, ncol=1, frameon=False)

    if save_path:
        plt.savefig(save_path)
    else:
        print("No save_path provided")


def main():
    ap = argparse.ArgumentParser(description="Summary plots across strategies (Fixed / V1 / V2 / V3)")
    ap.add_argument("--path", default="Data/Trigger_food_MC.h5")
    ap.add_argument("--start", type=int, default=500000)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--calib", type=int, default=100000)
    ap.add_argument("--out", default="outputs/Summary4Panel(dim=4).pdf")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    cases_lists = run_summary_stream(args.path, start_offset=args.start, chunk=args.chunk, fixed_calib_len=args.calib)

    # shape into plotting dict (same keys as your original)
    cases_data = {
        name: {
            'w0absrb': d['w0absrb'],
            'w1rs':    d['w1rs'],
            'cost':    d['cost'],
            'absrb':   d['absrb'],
            'rs':      d['rs'],
            'Tcost':   d['Tcost'],
            'Ecost':   d['Ecost'],
        } for name, d in cases_lists.items()
    }

    plot_case_comparison1(cases_data, n_bins=args.bins, save_path=args.out)

if __name__ == "__main__":
    main()
