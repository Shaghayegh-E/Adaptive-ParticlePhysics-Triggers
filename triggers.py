"""Trigger utilities: simple functions to compute pass rates and cuts."""
from typing import Iterable
import numpy as np


def Sing_Trigger(values: Iterable[float], ht_cut: float) -> float:
    """Compute the trigger pass rate (in percent) given values and a cut.

    This mirrors the original behavior: returns percentage of events with
    value >= cut.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0
    num_ = arr.shape[0]
    accepted_ht_ = np.sum(arr >= ht_cut)
    r_ = 100.0 * accepted_ht_ / num_
    return r_


def find_cut_for_target_rate(data: Iterable[float], target_rate: float) -> float:
    """Return a cut value that yields approximately `target_rate` acceptance.

    `target_rate` is a fraction (0..1). The function sorts the data and
    picks the appropriate quantile.
    """
    arr = np.sort(np.asarray(data))
    n = len(arr)
    if n == 0:
        raise ValueError("data is empty")
    cut_index = int((1.0 - target_rate) * n)
    cut_index = max(0, min(n - 1, cut_index))
    return arr[cut_index]
