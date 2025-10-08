from __future__ import annotations
import numpy as np
from .marginals import noarb_call_fit  # in-spread convex/monotone fit
import pandas as pd
from typing import Tuple, Iterable, Optional

# NOTE: Quote loading and hygiene filters for Section 10.
# - Expected CSV columns: ["maturity","K","bid","ask"] (mid is computed).
# - We enforce basic no-arbitrage diagnostics per maturity:
#   * vertical monotonicity in K (calls: non-increasing),
#   * butterfly convexity (discrete second differences >= -tol),
#   * optional calendar check across maturities when K grids match.

REQUIRED_COLS = {"maturity", "K", "bid", "ask"}

def load_quotes(csv_path: str) -> pd.DataFrame:
    """
    Load quotes CSV with columns: maturity (float), K (float), bid, ask.
    Adds 'mid' = 0.5*(bid+ask), sorts by [maturity, K].
    """
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)} in {csv_path}")
    df = df.copy()
    df["mid"] = 0.5 * (df["bid"].astype(float) + df["ask"].astype(float))
    df = df.dropna(subset=["maturity", "K", "mid"]).sort_values(["maturity", "K"]).reset_index(drop=True)
    return df


def _vertical_monotonic_mask(k: np.ndarray, c: np.ndarray, tol: float) -> np.ndarray:
    """
    Calls must be non-increasing in strike: C(K_{i+1}) - C(K_i) <= tol.
    Returns a boolean mask 'keep' for rows that pass the pairwise test.
    """
    keep = np.ones_like(c, dtype=bool)
    dC = np.diff(c)
    bad = np.where(dC > tol)[0]  # violation indices refer to left point of each pair
    # Drop the left point of violating pair (conservative choice)
    keep[bad] = False
    return keep


def _butterfly_convex_mask(k: np.ndarray, c: np.ndarray, tol: float) -> np.ndarray:
    """
    Discrete convexity (butterfly) for calls: second difference >= -tol.
    For non-uniform K, use scaled second difference.
    """
    keep = np.ones_like(c, dtype=bool)
    if c.size < 3:
        return keep
    dk = np.diff(k)
    # scaled second diff: 2/(dk_{i-1}+dk_i) * ((C_{i+1}-C_i)/dk_i - (C_i-C_{i-1})/dk_{i-1})
    curv = np.empty_like(c)
    curv[0] = curv[-1] = np.nan
    for i in range(1, c.size - 1):
        term = ( (c[i + 1] - c[i]) / dk[i] ) - ( (c[i] - c[i - 1]) / dk[i - 1] )
        curv[i] = 2.0 * term / (dk[i - 1] + dk[i])
    bad = np.where(curv[1:-1] < -tol)[0] + 1
    keep[bad] = False
    return keep


def filter_hygiene(df: pd.DataFrame,
                   tol_vertical: float = 1e-8,
                   tol_butterfly: float = 1e-8,
                   repair_within_spread: bool = False) -> pd.DataFrame:
    """
    Remove local violations of vertical monotonicity and butterfly convexity per maturity.
    Optionally, perform a convex & non-increasing fit inside bid/ask to repair mids instead of dropping.
    """
    out = []
    for T, g in df.groupby("maturity", sort=True):
        k = g["K"].to_numpy(float)
        c = g["mid"].to_numpy(float)
        keep_v = _vertical_monotonic_mask(k, c, tol_vertical)
        keep_b = _butterfly_convex_mask(k, c, tol_butterfly)
        keep = keep_v & keep_b
        gg = g.loc[keep].copy()
        if repair_within_spread and gg.shape[0] >= 3:
            k2  = gg["K"].to_numpy(float)
            bid = gg["bid"].to_numpy(float)
            ask = gg["ask"].to_numpy(float)
            mid = gg["mid"].to_numpy(float)
            gg["mid"] = noarb_call_fit(k2, bid, ask, mid)
        gg = gg.drop_duplicates(subset=["K"]).sort_values("K")
        out.append(gg)
    res = pd.concat(out, ignore_index=True).sort_values(["maturity", "K"]).reset_index(drop=True)
    return res


def calendar_filter(df_t0: pd.DataFrame, df_T: pd.DataFrame, tol: float = 1e-8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    K_common = np.intersect1d(df_t0["K"].to_numpy(float), df_T["K"].to_numpy(float))
    if K_common.size == 0:
        return df_t0.sort_values("K").reset_index(drop=True), df_T.sort_values("K").reset_index(drop=True)
    m0 = df_t0.set_index("K").loc[K_common]
    m1 = df_T.set_index("K").loc[K_common]
    viol = (m1["mid"].to_numpy(float) + tol) < m0["mid"].to_numpy(float)
    if viol.any():
        bad_K = K_common[viol]
        df_t0 = df_t0[~df_t0["K"].isin(bad_K)].copy()
    # never extrapolate; ensure sorted and deduped
    df_t0 = df_t0.sort_values("K").reset_index(drop=True)
    df_T  = df_T.sort_values("K").reset_index(drop=True)
    return df_t0, df_T


def normalize_to_grid(df: pd.DataFrame, K_grid: np.ndarray) -> pd.DataFrame:
    """
    Interpolate mid quotes onto a target K grid (within min/max support).
    Drops K outside data range to avoid extrapolation.
    Returns a new DataFrame with the same 'maturity' and 'mid' on K_grid subset.
    """
    g = df.sort_values("K")
    K = g["K"].to_numpy(float)
    C = g["mid"].to_numpy(float)
    Kmin, Kmax = K[0], K[-1]
    sel = (K_grid >= Kmin) & (K_grid <= Kmax)
    Kg = K_grid[sel]
    Cg = np.interp(Kg, K, C)
    out = pd.DataFrame({"maturity": g["maturity"].iloc[0], "K": Kg, "mid": Cg})
    return out.reset_index(drop=True)
