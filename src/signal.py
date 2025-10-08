from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional

from .hedge import realized_variance, delta_series

# NOTE:
# - robust_gap = P_mkt - Pi
# - after_cost_edge subtracts all costs.
# - pnl_vol_estimate estimates ex-ante PnL volatility per unit notional from λ*RV + Δ-hedge term.
# - size_kelly / size_sharpe sizing rules.
# - execution_sheet bundles key numbers for deployment.
# - market_forward_start_price: placeholder interface; returns 'fallback' if provided.

def market_forward_start_price(quotes_t0, quotes_T, kappa: float, strike: float | None,
                               fallback: Optional[float] = None) -> float:
    """
    Forward-start price is not directly inferable from vanilla smiles alone.
    Provide it either as:
      - cfg["market"]["forward_start_price"], or
      - fallback=... when calling this function.
    """
    if fallback is None:
        raise ValueError(
            "Forward-start market price not available. "
            "Set configs.market.forward_start_price in your YAML, "
            "or pass fallback=<price> to market_forward_start_price(...)."
        )
    return float(fallback)


def robust_gap(mkt_px: float, Pi: float) -> float:
    return float(mkt_px) - float(Pi)

def after_cost_edge(gap: float, C_stat: float, C_var: float, C_delta: float) -> float:
    return float(gap) - float(C_stat) - float(C_var) - float(C_delta)

def pnl_vol_estimate(paths: np.ndarray,
                     times: np.ndarray,
                     hedge: Dict[str, Any]) -> float:
    """
    Ex-ante PnL volatility per unit notional:
      PnL_path ≈ sum θ_k ΔF_k + λ * RV_path
    Return standard deviation across paths.
    """
    P = np.asarray(paths, float)
    if P.ndim == 1:  # promote single path to batch
        P = P.reshape(1, -1)
    t = np.asarray(times, float)
    lam = float(hedge["lambda"])
    theta_fn = hedge["delta"]["theta"]
    vals = []
    for p in P:
        dF = np.diff(p)
        theta = delta_series(p, t, theta_fn)
        pnl_delta = float(np.sum(theta * dF))
        rv = float(realized_variance(p, t))
        vals.append(pnl_delta + lam * rv)
    arr = np.asarray(vals, dtype=float)
    if arr.size <= 1:
        return float(abs(arr[0])) if arr.size == 1 else 0.0
    return float(np.std(arr, ddof=1))

def size_kelly(edge_ac: float, sigma_unit: float, kelly_fraction: float = 0.25) -> float:
    """
    Fractional Kelly: s = f * edge / sigma^2 (clip at 0 if edge<=0).
    """
    edge = float(edge_ac)
    if edge <= 0.0 or sigma_unit <= 0.0:
        return 0.0
    return float(kelly_fraction) * edge / (sigma_unit * sigma_unit)

def size_sharpe(edge_ac: float, sigma_unit: float, inv_sr: float) -> float:
    """
    Target Sharpe 1/inv_sr: s = edge / (inv_sr * sigma).
    """
    edge = float(edge_ac)
    if edge <= 0.0 or sigma_unit <= 0.0:
        return 0.0
    return edge / (float(inv_sr) * float(sigma_unit))

def execution_sheet(edge_ac: float,
                    size: float,
                    lambda_star: float,
                    turnover_est: float,
                    limits: Dict[str, float] | None = None) -> Dict[str, float]:
    """
    Minimal deployment sheet for risk/governance dashboards.
    """
    lims = limits or {}
    return {
        "edge_after_cost": float(edge_ac),
        "size": float(size),
        "lambda": float(lambda_star),
        "turnover": float(turnover_est),
        "cap_size": float(lims.get("max_size", np.inf)),
        "cap_turnover": float(lims.get("max_turnover", np.inf)),
    }
