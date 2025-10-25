from __future__ import annotations
import numpy as np

def _black_call_forward(F: float, K: np.ndarray, sigma: float) -> np.ndarray:
    """ Forward Black (no discount, T=1). """
    from scipy.stats import norm
    K = np.asarray(K, float)
    if sigma <= 0:
        return np.maximum(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma) / sigma
    d2 = d1 - sigma
    return F * norm.cdf(d1) - K * norm.cdf(d2)

def _quotes_from_call_curve(K: np.ndarray, C: np.ndarray, T: float):
    import pandas as pd
    return pd.DataFrame({"maturity": T, "K": K, "bid": C*0.999, "ask": C*1.001, "mid": C})

def test_dual_solver_small():
    # Grids
    x = np.linspace(80.0, 120.0, 61)
    y = np.linspace(75.0, 125.0, 71)
    t = np.linspace(0.5, 1.0, 11)

    # Synthetic smiles -> marginals (use our builder)
    from src.marginals import build_marginals
    F0 = 100.0
    Kmk = np.linspace(50.0, 150.0, 301)
    C0 = _black_call_forward(F0, Kmk, sigma=0.15)
    C1 = _black_call_forward(F0, Kmk, sigma=0.30)
    q0 = _quotes_from_call_curve(Kmk, C0, 0.5)
    q1 = _quotes_from_call_curve(Kmk, C1, 1.0)
    mu, nu = build_marginals(q0, q1, x, y, clip_neg=0.0, tol_cx=1e-7)

    # Payoff & solver
    from src.payoff import forward_start_factory, psi_star_factory
    payoff = forward_start_factory(kappa=1.0, strike=None)
    psi_star = psi_star_factory({"tol_indicator": 1e-12})

    from src.solver import solve_dual_penalized
    V = 0.10
    cfg = {
        "solver": {
            "eta0": 1e2, "eta_growth": 5.0,
            "eps_obs": 1e-6, "eps_curv": 1e-6, "eps_val": 1e-8,
            "max_outer": 10, "backend": "OSQP"
        }
    }
    out = solve_dual_penalized(mu, nu, (x, y, t), payoff, psi_star, V, cfg)
    # Basic assertions
    assert np.isfinite(out["Pi"])
    assert out["Pi"] >= 0.0
    res = out["residuals"]
    assert res["res_obs"] <= 1e-4
    assert res["res_curv"] <= 1e-4
