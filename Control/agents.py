# MC/trigger/agents.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from metrics import comp_costs  # must return (b_E, b_T)

# ------------------ shared utilities ------------------

def _grid_from_percentiles(
    vec_ht: np.ndarray,
    vec_as: np.ndarray,
    n_ht: int = 100,
    n_as: int = 100,
    p_lo: float = 0.01,
    p_hi_ht: float = 99.99,
    p_hi_as: float = 99.99,
) -> Tuple[np.ndarray, np.ndarray]:
    ht_vals = np.linspace(np.percentile(vec_ht, p_lo), np.percentile(vec_ht, p_hi_ht), n_ht)
    as_vals = np.linspace(np.percentile(vec_as, p_lo), np.percentile(vec_as, p_hi_as), n_as)
    return np.meshgrid(ht_vals, as_vals, indexing="ij")  # HT, AS

def _accepted_counts(
    ht: np.ndarray, as_: np.ndarray, HT: np.ndarray, AS: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    acc_ht = (ht[:, None, None] >= HT[None, :, :])
    acc_as = (as_[:, None, None] >= AS[None, :, :])
    both = (acc_ht & acc_as)
    return acc_ht.sum(0), acc_as.sum(0), both.sum(0), acc_ht, acc_as  # (n_ht, n_as), ...

def _rates_from_counts(n_ht: np.ndarray, n_as: np.ndarray, n_both: np.ndarray, denom: int) -> Tuple[np.ndarray, np.ndarray]:
    accepted = n_ht + n_as - n_both
    rate = 100.0 * accepted / max(1, denom)
    return accepted, rate

def _log_cost(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(x, 1e-10, None))

# ------------------ global agents (uniform returns) ------------------

def V0(bht, sht1, sht2, bas, sas1, sas2):
    """Minimize |r_b - 0.25| only (wide grid on AS)."""
    HT, AS = _grid_from_percentiles(bht, bas, n_ht=120, n_as=50, p_hi_as=99.999)
    s1_ht, s1_as, s1_both, *_ = _accepted_counts(sht1, sas1, HT, AS)
    s2_ht, s2_as, s2_both, *_ = _accepted_counts(sht2, sas2, HT, AS)
    _, r1_s = _rates_from_counts(s1_ht, s1_as, s1_both, sht1.shape[0])
    _, r2_s = _rates_from_counts(s2_ht, s2_as, s2_both, sht2.shape[0])
    b_ht, b_as, b_both, *_ = _accepted_counts(bht, bas, HT, AS)
    _, r_b = _rates_from_counts(b_ht, b_as, b_both, bht.shape[0])
    t_b = 0.25
    cost = np.abs(r_b - t_b)
    return _log_cost(cost), r_b, r1_s, r2_s, HT, AS

def V1(bht, sht1, sht2, bas, sas1, sas2, bnjets):
    HT, AS = _grid_from_percentiles(bht, bas)
    s1_ht, s1_as, s1_both, *_ = _accepted_counts(sht1, sas1, HT, AS)
    s2_ht, s2_as, s2_both, *_ = _accepted_counts(sht2, sas2, HT, AS)
    s1_acc, r1_s = _rates_from_counts(s1_ht, s1_as, s1_both, sht1.shape[0])
    s2_acc, r2_s = _rates_from_counts(s2_ht, s2_as, s2_both, sht2.shape[0])
    b_ht, b_as, b_both, *_ = _accepted_counts(bht, bas, HT, AS)
    _, r_b = _rates_from_counts(b_ht, b_as, b_both, bht.shape[0])

    total_s_rate = 100.0 * (s1_acc + s2_acc) / max(1, sht1.shape[0] + sht2.shape[0])
    a = [100.0, 0.2]; t_b = 0.25
    cost = a[0] * np.abs(r_b - t_b) + a[1] * np.abs(total_s_rate - 100.0)
    return _log_cost(cost), r_b, r1_s, r2_s, HT, AS

def V2(bht, sht1, sht2, bas, sas1, sas2, bnjets):
    HT, AS = _grid_from_percentiles(bht, bas)
    s1_ht, s1_as, s1_both, *_ = _accepted_counts(sht1, sas1, HT, AS)
    s2_ht, s2_as, s2_both, *_ = _accepted_counts(sht2, sas2, HT, AS)
    _, r1_s = _rates_from_counts(s1_ht, s1_as, s1_both, sht1.shape[0])
    _, r2_s = _rates_from_counts(s2_ht, s2_as, s2_both, sht2.shape[0])
    b_ht, b_as, b_both, b_acc_ht, b_acc_as = _accepted_counts(bht, bas, HT, AS)
    b_acc, r_b = _rates_from_counts(b_ht, b_as, b_both, bht.shape[0])

    r_as_ex = 100.0 * (b_acc_as.sum(0) - b_both) / max(1, bht.shape[0])  # exclusive AS rate
    a = [100.0, 0.2, 25.0]; t_b = 0.25; pct = 0.3
    cost = a[0]*np.abs(r_b - t_b) + a[1]*np.abs(r1_s - 90.0) + a[2]*np.abs(r_as_ex - pct*t_b)
    return _log_cost(cost), r_b, r1_s, r2_s, HT, AS

def V3(bht, sht1, sht2, bas, sas1, sas2, bnjets): #, snjets1, snjets2
    """Adds jet-aware comp costs via comp_costs."""
    HT, AS = _grid_from_percentiles(bht, bas)
    s1_ht, s1_as, s1_both, *_ = _accepted_counts(sht1, sas1, HT, AS)
    s2_ht, s2_as, s2_both, *_ = _accepted_counts(sht2, sas2, HT, AS)
    s1_acc, r1_s = _rates_from_counts(s1_ht, s1_as, s1_both, sht1.shape[0])
    s2_acc, r2_s = _rates_from_counts(s2_ht, s2_as, s2_both, sht2.shape[0])
    total_s_rate = 100.0 * (s1_acc + s2_acc) / max(1, sht1.shape[0] + sht2.shape[0])

    b_ht, b_as, b_both, b_acc_ht, b_acc_as = _accepted_counts(bht, bas, HT, AS)
    b_acc, r_b = _rates_from_counts(b_ht, b_as, b_both, bht.shape[0])

    b_accepted = (b_acc_ht | b_acc_as)         # (N_bkg, n_ht, n_as)
    b_E, b_T = comp_costs(b_accepted, b_acc, b_ht, b_both, b_as, bnjets)  # returns arrays (n_ht, n_as)

    a = [100.0, 0.2, 1/0.5, 1/0.5]; t_b = 0.25
    cost = (a[0]*np.abs(r_b - t_b) +
            a[1]*np.abs(total_s_rate - 100.0) +
            a[2]*np.maximum(b_E - 5.5, 0.0) +
            a[3]*np.maximum(b_T - 2.5, 0.0))
    return _log_cost(cost), r_b, r1_s, r2_s, HT, AS

def V4(bht, sht1, sht2, bas, sas1, sas2):
    """Includes simple per-event computational cost term."""
    HT, AS = _grid_from_percentiles(bht, bas, n_ht=120, n_as=50, p_hi_as=99.999)
    s1_ht, s1_as, s1_both, *_ = _accepted_counts(sht1, sas1, HT, AS)
    s2_ht, s2_as, s2_both, *_ = _accepted_counts(sht2, sas2, HT, AS)
    _, r1_s = _rates_from_counts(s1_ht, s1_as, s1_both, sht1.shape[0])
    _, r2_s = _rates_from_counts(s2_ht, s2_as, s2_both, sht2.shape[0])
    b_ht, b_as, b_both, *_ = _accepted_counts(bht, bas, HT, AS)
    _, r_b = _rates_from_counts(b_ht, b_as, b_both, bht.shape[0])

    event_cost_ht, event_cost_as = 1.0, 10.0
    b_comp = (event_cost_ht * (b_ht - b_both)
              + event_cost_as * (b_as - b_both)
              + (event_cost_ht + event_cost_as) * b_both) / max(1, bht.shape[0])
    a = [1.0, 10.0, 3.0]; t_b = 0.25
    cost = a[0]*np.abs(r_b - t_b) + a[1]*np.abs(r1_s - 90.0) + (a[2]*b_comp)**2
    return _log_cost(cost), r_b, r1_s, r2_s, HT, AS

def V5(bht, sht1, sht2, bas, sas1, sas2):
    """Strong r_b term with comp-cost inside, plus total signal rate."""
    HT, AS = _grid_from_percentiles(bht, bas, p_lo=0.001, p_hi_ht=99.999, p_hi_as=99.999)
    s1_ht, s1_as, s1_both, *_ = _accepted_counts(sht1, sas1, HT, AS)
    s2_ht, s2_as, s2_both, *_ = _accepted_counts(sht2, sas2, HT, AS)
    s1_acc, r1_s = _rates_from_counts(s1_ht, s1_as, s1_both, sht1.shape[0])
    s2_acc, r2_s = _rates_from_counts(s2_ht, s2_as, s2_both, sht2.shape[0])
    total_s_rate = 100.0 * (s1_acc + s2_acc) / max(1, sht1.shape[0] + sht2.shape[0])

    b_ht, b_as, b_both, *_ = _accepted_counts(bht, bas, HT, AS)
    _, r_b = _rates_from_counts(b_ht, b_as, b_both, bht.shape[0])

    event_cost_ht, event_cost_as = 1.0, 10.0
    b_comp_cost = (
        event_cost_ht * (b_ht - b_both)
        + event_cost_as * (b_as - b_both)
        + (event_cost_ht + event_cost_as) * b_both
    ) / max(1, bht.shape[0])

    a = [1.0, 1000.0, 3.0]; t_b = 0.25
    cost = a[0] * (1000 * (np.abs(r_b - t_b) + 10.0 * b_comp_cost))**4 + a[1] * np.abs(total_s_rate - 100.0)
    return _log_cost(cost), r_b, r1_s, r2_s, HT, AS

def V6(bht, sht1, sht2, bas, sas1, sas2):
    """rb^4 + total_s_rate + (overlap proxies)"""
    HT, AS = _grid_from_percentiles(bht, bas, p_lo=0.001, p_hi_ht=99.999, p_hi_as=99.999)
    s1_ht, s1_as, s1_both, *_ = _accepted_counts(sht1, sas1, HT, AS)
    s2_ht, s2_as, s2_both, *_ = _accepted_counts(sht2, sas2, HT, AS)
    s1_acc, r1_s = _rates_from_counts(s1_ht, s1_as, s1_both, sht1.shape[0])
    s2_acc, r2_s = _rates_from_counts(s2_ht, s2_as, s2_both, sht2.shape[0])
    total_s_rate = 100.0 * (s1_acc + s2_acc) / max(1, sht1.shape[0] + sht2.shape[0])

    b_ht, b_as, b_both, *_ = _accepted_counts(bht, bas, HT, AS)
    _, r_b = _rates_from_counts(b_ht, b_as, b_both, bht.shape[0])

    # simple overlap proxies (normalized)
    b_overlap = (b_both + 1e-10) / (b_ht + b_as - b_both + 1e-10)
    s1_overlap = (s1_both + 1e-10) / (s1_ht + s1_as - s1_both + 1e-10)
    s2_overlap = (s2_both + 1e-10) / (s2_ht + s2_as - s2_both + 1e-10)

    a0 = 1.0
    cost = (a0 * (1000.0 * np.abs(r_b - 0.25))**4
            + np.abs(total_s_rate - 100.0)
            + (b_overlap + s1_overlap + s2_overlap)**2)
    return _log_cost(cost), r_b, r1_s, r2_s, HT, AS

# ------------------ local/windowed variants ------------------

def local_V1(bht, sht1, sht2, bas, sas1, sas2, ht_value, as_value, ht_win=20, as_win=20, num=10):
    # ht_vals = np.linspace(max(np.min(bht), ht0 - ht_win), min(np.percentile(bht, 99.99), ht0 + ht_win), num)
        # Define the local range around the given ht_value and as_value
    print("bht:", np.array(bht).shape)
    # print("ht_min:", np.array(ht_min).shape)
    print("ht0:", np.array(ht_value).shape)
    # print("ht_win:", np.array(ht_win).shape)
    ht_min, ht_max = ht_value - ht_win, ht_value + ht_win
    as_min, as_max = as_value - as_win, as_value + as_win

    MAX = np.percentile(bht, 99.99)
    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num)

    MAX = np.percentile(bas, 99.99)
    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing="ij")


    s1_ht, s1_as, s1_both, *_ = _accepted_counts(sht1, sas1, HT, AS)
    s2_ht, s2_as, s2_both, *_ = _accepted_counts(sht2, sas2, HT, AS)
    s1_acc, _ = _rates_from_counts(s1_ht, s1_as, s1_both, sht1.shape[0])
    s2_acc, _ = _rates_from_counts(s2_ht, s2_as, s2_both, sht2.shape[0])
    r_s = 100.0 * (s1_acc + s2_acc) / max(1, sht1.shape[0] + sht2.shape[0])

    b_ht, b_as, b_both, *_ = _accepted_counts(bht, bas, HT, AS)
    _, r_b = _rates_from_counts(b_ht, b_as, b_both, bht.shape[0])

    a = [100.0, 0.2]; t_b = 0.25
    cost = a[0]*np.abs(r_b - t_b) + a[1]*np.abs(r_s - 100.0)
    return _log_cost(cost), r_b, r_s, HT, AS

def local_V2(bht, sht1, sht2, bas, sas1, sas2, bnjets, ht0, as0, ht_win=20, as_win=20, num_points=10):
    

    # # Define the local range around the given ht_value and as_value
    # ht_min, ht_max = ht_value - ht_window, ht_value + ht_window
    # as_min, as_max = as_value - as_window, as_value + as_window

    # # Generate local ht and as values
    # MAX = np.percentile(bht,99.99)

    # ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num_points)
    
    # MAX = np.percentile(bht,99.99)

    # as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num_points)

    # # Create a grid in the local window
    # HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij')

    ht_min, ht_max = ht0 - ht_win, ht0 + ht_win
    as_min, as_max = as0 - as_win, as0 + as_win

    MAX = np.percentile(bht, 99.99)
    ht_vals = np.linspace(max(np.min(bht), ht_min), min(MAX, ht_max), num)

    MAX = np.percentile(bas, 99.99)
    as_vals = np.linspace(max(np.min(bas), as_min), min(MAX, as_max), num)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing="ij")

    # Signal computations
    s1_accepted_ht = (sht1[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s1_accepted_as = (sas1[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)

    s2_accepted_ht = (sht2[:, None, None] >= HT[None, :, :])   # shape: (N_signal, n_ht, n_as)
    s2_accepted_as = (sas2[:, None, None] >= AS[None, :, :])     # shape: (N_signal, n_ht, n_as)
    

    s1_ht_count = s1_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s1_as_count = s1_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)  # events passing both

    s2_ht_count = s2_accepted_ht.sum(axis=0)      # shape: (n_ht, n_as)
    s2_as_count = s2_accepted_as.sum(axis=0)        # shape: (n_ht, n_as)
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)  # events passing both
    
    # Total signal accepted events = ht + as - both 
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    r1_s = 100 * s1_accepted_events / sht1.shape[0]
    
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count
    #r2_s = 100 * s2_accepted_events / sht2.shape[0]
    
    
    total_s_accepted_events = s2_accepted_events + s1_accepted_events
    r_s = 100 * (total_s_accepted_events)/ (sht1.shape[0] + sht2.shape[0])

    
    # -----------------------------
    # Background computations
    b_accepted_ht = (bht[:, None, None] >= HT[None, :, :])
    b_accepted_as = (bas[:, None, None] >= AS[None, :, :])
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    
    
    
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]

    
    
    r_as_ex = 100 * (b_as_count - b_both_count)/(bht.shape[0])

    
    

    
    # Additional rates if needed
    r_bht = 100 * b_ht_count / bht.shape[0]
    r_bas = 100 * b_as_count / bht.shape[0]
    r_sht = 100 * (s1_ht_count + s2_ht_count)/ (sht1.shape[0]+sht2.shape[0])
    r_sas = 100 * (s1_as_count + s2_as_count)/ (sas1.shape[0]+sas2.shape[0])


    
    a = [100, .2, 25]
    t_b = 0.25
    percentage = .3
    cost =  (a[0] *np.abs(r_b - t_b)) + (a[1] *np.abs(r1_s - 90)) + (a[2] * np.abs(r_as_ex - percentage*t_b))

    log_Cost = np.log10(cost.clip(min=1e-10))

    return _log_cost(cost), r_b, r_s, HT, AS

def local_V3(bht, sht1, sht2, bas, sas1, sas2, bnjets, ht0, as0, ht_win=20, as_win=20, num=10):
    ht_vals = np.linspace(max(np.min(bht), ht0 - ht_win), min(np.percentile(bht, 99.99), ht0 + ht_win), num)
    as_vals = np.linspace(max(np.min(bas), as0 - as_win), min(np.percentile(bas, 99.99), as0 + as_win), num)
    HT, AS = np.meshgrid(ht_vals, as_vals, indexing="ij")

    s1_ht, s1_as, s1_both, *_ = _accepted_counts(sht1, sas1, HT, AS)
    s2_ht, s2_as, s2_both, *_ = _accepted_counts(sht2, sas2, HT, AS)
    s1_acc, _ = _rates_from_counts(s1_ht, s1_as, s1_both, sht1.shape[0])
    s2_acc, _ = _rates_from_counts(s2_ht, s2_as, s2_both, sht2.shape[0])
    r_s = 100.0 * (s1_acc + s2_acc) / max(1, sht1.shape[0] + sht2.shape[0])

    b_ht, b_as, b_both, b_acc_ht, b_acc_as = _accepted_counts(bht, bas, HT, AS)
    _, r_b = _rates_from_counts(b_ht, b_as, b_both, bht.shape[0])

    b_E, b_T = comp_costs((b_acc_ht | b_acc_as), (b_ht + b_as - b_both), b_ht, b_both, b_as, bnjets)
    a = [100.0, 0.2, 1/0.5, 1/0.5]; t_b = 0.25
    cost = (a[0]*np.abs(r_b - t_b) +
            a[1]*np.abs(r_s - 100.0) +
            a[2]*np.maximum(b_E - 5.5, 0.0) +
            a[3]*np.maximum(b_T - 2.5, 0.0))
    return _log_cost(cost), r_b, r_s, HT, AS
