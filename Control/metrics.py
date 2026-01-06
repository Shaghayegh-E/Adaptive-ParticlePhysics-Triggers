# rates, cost helpers, accumulation utilities 
from __future__ import annotations
import numpy as np
from typing import Tuple

def update_accumulated(acc_list, new_val, new_sample_size, total_samples_list):
    if not acc_list:
        acc_list.append(new_val)
        total_samples_list.append(new_sample_size)
    else:
        prev_total = total_samples_list[-1]
        tot = prev_total + new_sample_size
        new_avg = (acc_list[-1] * prev_total + new_val * new_sample_size) / max(1, tot)
        acc_list.append(new_avg)
        total_samples_list.append(tot)
        
def comp_costs(
    b_accepted: np.ndarray,
    b_accepted_events: np.ndarray,
    b_ht_count: np.ndarray,
    b_both_count: np.ndarray,
    b_as_count: np.ndarray,
    bnjets: np.ndarray,
    ht_weight: float = 1.0,
    as_weight: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    # shape checks
    bnjets_reshaped = bnjets[:, None, None]  # (N,1,1)
    # Trigger path cost: average nJets over accepted events
    b_Ecomp_cost = ((b_accepted * bnjets_reshaped).sum(axis=0)) / (b_accepted_events + 1e-10)
    # Event-level cost: weighted counts per accepted event
    b_Tcomp_cost = (ht_weight * (b_ht_count - b_both_count) + as_weight * b_as_count) / (b_accepted_events + 1e-10)
    return b_Ecomp_cost, b_Tcomp_cost


def average_perf_bins(performance_list, n_bins=25):
    time_indices = np.arange(len(performance_list))
    bins = np.array_split(time_indices, n_bins)
    avg_performance = []
    for bin_indices in bins:
        avg_perf = np.mean([performance_list[i] for i in bin_indices])
        avg_performance.append(avg_perf)

    return np.array(avg_performance)


def comp_cost_test(bht, bas, bnjets, sht1, sas1, snjets1, sht2, sas2, snjets2,
use_path1=True, use_path2=True):

    # Build HT and AS thresholds
    MAX_ht = np.percentile(bht, 99.99)
    ht_vals = np.linspace(np.percentile(bht, 0.01), MAX_ht, 100)

    MAX_as = np.percentile(bas, 99.99)
    as_vals = np.linspace(np.percentile(bas, 0.01), MAX_as, 100)

    HT, AS = np.meshgrid(ht_vals, as_vals, indexing='ij')

    # ---------- PATH 1 ----------
    if use_path1:
        s1_accepted_ht = sht1[:, None, None] >= HT[None, :, :]
        s2_accepted_ht = sht2[:, None, None] >= HT[None, :, :]
        b_accepted_ht = bht[:, None, None] >= HT[None, :, :]
        
    else:
        s1_accepted_ht = np.zeros_like(sht1, dtype=bool)[:, None, None]
        s2_accepted_ht = np.zeros_like(sht2, dtype=bool)[:, None, None]       
        b_accepted_ht = np.zeros_like(bht, dtype=bool)[:, None, None]

        
        

    # ---------- PATH 2 ----------
    if use_path2:
        s1_accepted_as = sas1[:, None, None] >= AS[None, :, :]
        s2_accepted_as = sas2[:, None, None] >= AS[None, :, :]
        b_accepted_as = bas[:, None, None] >= AS[None, :, :]
        
    else:
        s1_accepted_as = np.zeros_like(sas1, dtype=bool)[:, None, None]
        s2_accepted_as = np.zeros_like(sas2, dtype=bool)[:, None, None]
        b_accepted_as = np.zeros_like(bas, dtype=bool)[:, None, None]



    s1_ht_count = s1_accepted_ht.sum(axis=0)
    s2_ht_count = s2_accepted_ht.sum(axis=0)
    
    s1_as_count = s1_accepted_as.sum(axis=0)
    s2_as_count = s2_accepted_as.sum(axis=0)


    s1_both_count = (s1_accepted_ht & s1_accepted_as).sum(axis=0)
    s1_accepted_events = s1_ht_count + s1_as_count - s1_both_count
    
    s2_both_count = (s2_accepted_ht & s2_accepted_as).sum(axis=0)
    s2_accepted_events = s2_ht_count + s2_as_count - s2_both_count

    # ---------- Total Signal Rate ----------
    total_s_accepted_events = s1_accepted_events + s2_accepted_events
    
    total_s_rate = 100 * total_s_accepted_events / (sht1.shape[0] + sht2.shape[0] + 1e-10)

    Ht_cost = 1
    AS_cost = 4
    
    b_ht_count = b_accepted_ht.sum(axis=0)
    b_as_count = b_accepted_as.sum(axis=0)
    
    b_accepted = b_accepted_ht | b_accepted_as
    
    
    b_both_count = (b_accepted_ht & b_accepted_as).sum(axis=0)
    b_accepted_events = b_ht_count + b_as_count - b_both_count
    r_b = 100 * b_accepted_events / bht.shape[0]

    # ---------- COMPUTATIONAL COST ----------
    bnjets_reshaped = bnjets[:, None, None]
    b_Ecomp_cost = ((b_accepted * bnjets_reshaped).sum(axis=0))/(b_accepted_events)


    b_Tcomp_cost = (Ht_cost*(b_ht_count - b_both_count) + AS_cost * b_as_count)/(b_accepted_events)
    
    # ---------- COST FUNCTION ----------
    a = [100, 0.2]
    t_b = 0.25
    cost = a[0] * np.abs(r_b - t_b) + a[1] * np.abs(total_s_rate - 100)

    return cost, b_Ecomp_cost, b_Tcomp_cost

