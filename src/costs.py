from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# NOTE:
# - static_cost_from_market: price the static buckets using market call quotes on the strike grid.
# - variance_leg_cost: cost for realized-variance leg (market swap if available, else proxy).
# - delta_cost_estimate: microstructure cost from delta turnover (spread + optional impact).

def _interp_on_grid(K_mkt: np.ndarray, px_mkt: np.ndarray, K_tgt: np.ndarray) -> np.ndarray:
    """
    Interpolate prices on target strikes; returns NaN outside the market support.
    Caller decides whether to ignore, clip, or raise.
    """
    K_mkt = np.asarray(K_mkt, float)
    px_mkt = np.asarray(px_mkt, float)
    K_tgt = np.asarray(K_tgt, float)
    Kmin, Kmax = K_mkt[0], K_mkt[-1]
    out = np.full_like(K_tgt, np.nan, dtype=float)
    sel = (K_tgt >= Kmin) & (K_tgt <= Kmax)
    if sel.any():
        out[sel] = np.interp(K_tgt[sel], K_mkt, px_mkt)
    return out


def static_cost_from_market(static_legs: Dict[str, float | np.ndarray],
                            quotes: pd.DataFrame,
                            forward_level: Optional[float] = None,
                            side: str = "mid") -> float:
    """
    Cost of static replication stack:
      cost = alpha0 + alpha1 * F + sum_j w_call[j] * C_side(K_j)
    Strikes outside market support are ignored (conservative). If you prefer to raise, detect NaNs and raise.
    """
    K = np.asarray(static_legs["K"], float)
    w = np.asarray(static_legs["w_call"], float)
    a0 = float(static_legs["alpha0"])
    a1 = float(static_legs["alpha1"])

    side = side.lower()
    if side not in {"mid", "bid", "ask"}:
        side = "mid"
    col = side

    g = quotes.sort_values("K")
    Kmkt = g["K"].to_numpy(float)
    Cmkt = g[col].to_numpy(float)

    C_on_K = _interp_on_grid(Kmkt, Cmkt, K)
    mask = ~np.isnan(C_on_K)
    if not mask.any():
        # No overlap: ultra-conservative fallback is just linear part.
        F = float(forward_level) if forward_level is not None else float(np.median(K))
        return a0 + a1 * F

    F = float(forward_level) if forward_level is not None else float(np.median(K))
    cost = a0 + a1 * F + float(np.dot(w[mask], C_on_K[mask]))
    return cost


def variance_leg_cost(market_var: Dict[str, Any] | None,
                      proxy_cfg: Dict[str, Any] | None = None) -> float:
    """
    Market variance swap if available:
      cost = notional * swap_rate
    Else proxy:
      cost = proxy_cfg["notional"] * proxy_cfg["proxy_rate"]  (defaults to 0).
    """
    if market_var and ("swap_rate" in market_var) and ("notional" in market_var):
        return float(market_var["notional"]) * float(market_var["swap_rate"])
    if proxy_cfg and ("notional" in proxy_cfg) and ("proxy_rate" in proxy_cfg):
        return float(proxy_cfg["notional"]) * float(proxy_cfg["proxy_rate"])
    return 0.0

def delta_cost_estimate(turnover_series: np.ndarray,
                        spread: float,
                        impact: Dict[str, float] | None = None) -> float:
    """
    Microstructure cost for delta rebalancing:
      cost = spread * sum |Δθ| + c * (sum |Δθ|)^beta   (square-root law if beta=0.5)
    Args:
      turnover_series: array of per-step |Δθ| (or raw Δθ; abs is applied here).
      spread: per-unit half-spread in underlying delta terms.
      impact: {"c": float, "beta": float}, optional.
    """
    dtheta_abs = np.abs(np.asarray(turnover_series, dtype=float))
    base = spread * float(np.sum(dtheta_abs))
    if impact is None:
        return base
    c = float(impact.get("c", 0.0))
    beta = float(impact.get("beta", 0.5))
    return base + c * (float(np.sum(dtheta_abs)) ** beta)
