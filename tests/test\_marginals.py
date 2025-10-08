from __future__ import annotations
import numpy as np

# Minimal self-test for marginals: BL positivity/normalization + convex order.
# Uses synthetic Black-style forward calls with higher variance at T.

def _black_call_forward(F: float, K: np.ndarray, sigma: float) -> np.ndarray:
    """ Forward Black (no discount): C = F*N(d1) - K*N(d2) with T=1. """
    from math import log, sqrt
    from scipy.stats import norm
    K = np.asarray(K, float)
    if sigma <= 0:
        return np.maximum(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma) / (sigma)
    d2 = d1 - sigma
    return F * norm.cdf(d1) - K * norm.cdf(d2)

def test_bl_and_cx():
    from src.marginals import breeden_litzenberger_forward, build_marginals, convex_order_check
    # Synthetic quotes at t0 and T
    F0 = 100.0
    K_grid = np.linspace(40.0, 160.0, 241)  # fine, uniform
    C0 = _black_call_forward(F0, K_grid, sigma=0.15)
    C1 = _black_call_forward(F0, K_grid, sigma=0.30)  # more dispersion

    import pandas as pd
    q0 = pd.DataFrame({"maturity": 0.5, "K": K_grid, "bid": C0*0.999, "ask": C0*1.001, "mid": C0})
    q1 = pd.DataFrame({"maturity": 1.0, "K": K_grid, "bid": C1*0.999, "ask": C1*1.001, "mid": C1})

    # Target (possibly different) grids
    x = np.linspace(50.0, 150.0, 221)
    y = np.linspace(45.0, 170.0, 251)

    mu, nu = build_marginals(q0, q1, x, y, clip_neg=0.0, tol_cx=1e-7)
    assert np.all(mu >= -1e-12) and np.all(nu >= -1e-12)
    assert abs(mu.sum() - 1.0) < 1e-8
    assert abs(nu.sum() - 1.0) < 1e-8

    ok = convex_order_check(mu, x, nu, y, tol=1e-7)
    assert ok, "convex order must hold (variance increased at T)"

if __name__ == "__main__":
    test_bl_and_cx()
    print("OK: marginals BL + convex order")
