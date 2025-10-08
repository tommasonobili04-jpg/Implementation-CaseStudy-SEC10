from __future__ import annotations
import numpy as np
from typing import Tuple

# NOTE: Space/time grids for Section 10.
# - build_space_grid: dense around ATM (spot), thinner in tails, using log-moneyness.
# - build_time_grid: inclusive grid from t0 to T with K_time steps (K_time+1 points).
# - adaptive_refine: inserts midpoints on intervals with largest weights (e.g., curvature).

def build_space_grid(spot: float,
                     atm: int,
                     tail: int,
                     lm_range: Tuple[float, float] = (-0.60, 0.60)) -> np.ndarray:
    """
    Build a price grid via log-moneyness L := log(F/spot).
    Returns exactly n = atm + 2*tail points: left tail (tail), ATM band (atm), right tail (tail).
    Args:
        spot: reference forward/spot level (>0).
        atm: number of points concentrated around L in [-0.10, 0.10).
        tail: number of points per tail outside [-0.10, 0.10].
        lm_range: global log-moneyness range [Lmin, Lmax].
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    Lmin, Lmax = lm_range
    Lmid = (-0.10, 0.10)

    # Left tail, ATM, right tail in log-moneyness
    L_left  = np.linspace(Lmin, Lmid[0], tail, endpoint=False)   # tail pts in [Lmin, Lmid[0])
    L_atm   = np.linspace(Lmid[0], Lmid[1], atm, endpoint=False) # atm pts  in [Lmid[0], Lmid[1])
    L_right = np.linspace(Lmid[1], Lmax,    tail, endpoint=True) # tail pts including Lmax

    L = np.concatenate([L_left, L_atm, L_right])
    if L.size != atm + 2 * tail:
        raise RuntimeError("grid construction mismatch: expected atm + 2*tail points")

    x = spot * np.exp(L)
    # Ensure strict increase
    if not np.all(np.diff(x) > 0):
        x = np.unique(x)
        if x.size < 3 or not np.all(np.diff(x) > 0):
            raise ValueError("grid too small or non-monotone; increase atm/tail")
    return x



def build_time_grid(t0: float, T: float, K_time: int) -> np.ndarray:
    """
    Inclusive time grid from t0 to T with K_time steps (K_time+1 points).
    """
    if not (0.0 <= t0 < T):
        raise ValueError("require 0 <= t0 < T")
    if K_time < 1:
        raise ValueError("K_time must be >= 1")
    return np.linspace(t0, T, K_time + 1)


def adaptive_refine(x: np.ndarray, weight: np.ndarray, n_extra: int) -> np.ndarray:
    """
    Insert 'n_extra' midpoints into intervals with largest weights.
    Typical 'weight' choices: |second derivative|, |price curvature|, variance density, etc.
    Args:
        x: strictly increasing 1D grid (n,).
        weight: nonnegative weights for intervals (n-1,), aligned with (x[i], x[i+1]).
        n_extra: number of midpoints to insert (>=0).
    Returns:
        x_ref: refined, strictly increasing grid.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(weight, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be 1D with >=2 points")
    if w.shape != (x.size - 1,):
        raise ValueError("weight must have shape (len(x)-1,)")
    if n_extra <= 0:
        return x

    idx = np.argsort(w)[::-1][:n_extra]
    mids = 0.5 * (x[idx] + x[idx + 1])
    x_ref = np.sort(np.unique(np.concatenate([x, mids])))
    return x_ref
