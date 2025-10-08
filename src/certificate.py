from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Callable

from .hedge import realized_variance, delta_series

# NOTE:
# - pathwise_bound: computes RHS of the discrete certificate for a single path and compares to payoff.
# - check_certificate: batch check on multiple paths; returns violation stats.

def pathwise_bound(path: np.ndarray,
                   times: np.ndarray,
                   hedge: Dict[str, object],
                   payoff: Callable[[float, float], float],
                   corridor: Tuple[float, float] | None = None) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute LHS=Φ(F_{t0},F_1) and RHS=u(F_0)+v(F_1)+φ(t0,F_{t0})-φ(1,F_1)+λ*RV + PnL_Δ.
    Uses y-grid evaluation for φ(1,·), consistent with the dual.
    """
    f = np.asarray(path, float); t = np.asarray(times, float)
    if f.ndim != 1 or t.ndim != 1 or f.size != t.size:
        raise ValueError("path and times must be 1D with same length")
    if f.size < 2:
        raise ValueError("need at least two time points")

    F0, F1 = float(f[0]), float(f[-1])
    Ft0 = F0  # by convention times[0] = t0

    # Static terms
    u_val = float(hedge["u_fn"](F0))
    v_val = float(hedge["v_fn"](F1))

    # φ-terms: t0 on x-grid, terminal on y-grid (consistent with P_yx)
    phi_t0_val = float(hedge["phi_t0_fn"](Ft0))
    phi_T_val  = float(hedge["phi_Ty_fn"](F1))

    # Delta PnL (use theta at left endpoints)
    theta_fn = hedge["delta"]["theta"]
    theta = delta_series(f, t, theta_fn)
    pnl_delta = float(np.sum(theta * np.diff(f)))

    # Realized variance (corridor optional)
    if corridor is None:
        rv = realized_variance(f, t)
    else:
        L, U = corridor
        mask = (f[:-1] >= L) & (f[:-1] <= U)
        df = np.diff(f)
        dt = np.diff(t)
        rv = float(np.sum(((df * df) / dt)[mask]))

    lam = float(hedge["lambda"])
    rhs = u_val + v_val + (phi_t0_val - phi_T_val) + lam * rv + pnl_delta
    lhs = float(payoff(Ft0, F1))
    details = {"u": u_val, "v": v_val, "phi_term": phi_t0_val - phi_T_val,
               "rv": rv, "lambda": lam, "pnl_delta": pnl_delta}
    return lhs, rhs, details



def check_certificate(paths: np.ndarray,
                      times: np.ndarray,
                      hedge: Dict[str, object],
                      payoff: Callable[[float, float], float],
                      tol: float = 1e-3) -> Dict[str, float]:
    """
    Check certificate on a batch of paths. Counts violations of:
        Φ(F_{t0},F_1) <= RHS + tol
    Returns {"violations": int, "max_gap": float}.
    """
    P = np.asarray(paths, float)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    viol = 0
    max_gap = -np.inf
    for p in P:
        lhs, rhs, _ = pathwise_bound(p, times, hedge, payoff)
        gap = lhs - rhs
        max_gap = max(max_gap, gap)
        if gap > tol:
            viol += 1
    return {"violations": int(viol), "max_gap": float(max_gap)}
