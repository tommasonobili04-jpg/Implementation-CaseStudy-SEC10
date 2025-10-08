from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Union

ArrayLike = Union[float, np.ndarray]

# NOTE:
# - forward_start_factory returns Φ(x,y) = max(y - κ x - K, 0).
# - psi_star_factory provides a minimal Ψ* indicator used for semantics of the dual
#   (the curvature feasibility is enforced numerically via constraints/penalties in solver.py).


def forward_start_factory(kappa: float, strike: Optional[float] = None) -> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    """
    Create forward-start call payoff Φ(F_{t0}=x, F_1=y) = (y - κ x - K)_+.

    Args:
        kappa: κ in (0, +∞). Typical case: κ in (0,1].
        strike: K >= 0; if None, K := 0.

    Returns:
        payoff(x, y) supporting scalar or numpy arrays with broadcasting.
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    K = 0.0 if strike is None else float(strike)

    def payoff(x: ArrayLike, y: ArrayLike) -> ArrayLike:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        z = y_arr - kappa * x_arr - K
        return np.maximum(z, 0.0)

    return payoff


def psi_star_factory(cfg: dict) -> Callable[[ArrayLike, float], ArrayLike]:
    """
    Minimal convex-conjugate Ψ* proxy used only for semantics/diagnostics.
    In the constrained dual, curvature feasibility is: φ_xx >= -2 λ.
    The corresponding Ψ* is an indicator I_{[ -2λ, +∞ )}(φ_xx).

    Returns:
        psi_star(h, lam): 0 if h >= -2*lam (within a small tolerance), +inf otherwise.
                          Works on scalars or numpy arrays.
    """
    tol = float(cfg.get("tol_indicator", 1e-12))

    def psi_star(h: ArrayLike, lam: float = 0.0) -> ArrayLike:
        h_arr = np.asarray(h, dtype=float)
        thr = -2.0 * float(lam) - tol
        out = np.where(h_arr >= thr, 0.0, np.inf)
        # If user passes a scalar, return a scalar
        if np.isscalar(h) or (isinstance(h, np.ndarray) and h_arr.shape == ()):
            return float(out)  # type: ignore[arg-type]
        return out

    return psi_star
