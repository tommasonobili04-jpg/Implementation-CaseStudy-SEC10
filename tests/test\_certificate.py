from __future__ import annotations
import numpy as np

# End-to-end small test:
# - build marginals from synthetic quotes,
# - solve penalized dual for u,v,φ,λ,
# - assemble hedge,
# - simulate a few paths and verify the discrete certificate holds within tolerance.

def _black_call_forward(F: float, K: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.stats import norm
    K = np.asarray(K, float)
    if sigma <= 0.0:
        return np.maximum(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma*sigma) / sigma
    d2 = d1 - sigma
    return F * norm.cdf(d1) - K * norm.cdf(d2)

def _quotes_from_call_curve(K: np.ndarray, C: np.ndarray, T: float):
    import pandas as pd
    return pd.DataFrame({"maturity": T, "K": K, "bid": C*0.999, "ask": C*1.001, "mid": C})

def _simulate_paths(F0: float, times: np.ndarray, sigma: float, n_paths: int = 5, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.asarray(times, float)
    P = np.zeros((n_paths, t.size), dtype=float)
    for p in range(n_paths):
        f = np.empty_like(t)
        f[0] = F0
        for k in range(t.size - 1):
            dt = t[k+1] - t[k]
            z = rng.standard_normal()
            # simple arithmetic Brownian martingale
            f[k+1] = f[k] + sigma * np.sqrt(dt) * z
        P[p] = f
    return P

def test_certificate_end_to_end_small():
    # grids
    x = np.linspace(90.0, 110.0, 41)
    y = np.linspace(85.0, 115.0, 51)
    t = np.linspace(0.5, 1.0, 21)

    # synthetic smiles → marginals
    F0 = 100.0
    Kmk = np.linspace(60.0, 140.0, 161)
    C0 = _black_call_forward(F0, Kmk, sigma=0.12)
    C1 = _black_call_forward(F0, Kmk, sigma=0.25)
    q0 = _quotes_from_call_curve(Kmk, C0, 0.5)
    q1 = _quotes_from_call_curve(Kmk, C1, 1.0)

    from src.marginals import build_marginals
    mu, nu = build_marginals(q0, q1, x, y, clip_neg=0.0, tol_cx=1e-7)

    # payoff & solve dual
    from src.payoff import forward_start_factory, psi_star_factory
    payoff = forward_start_factory(kappa=1.0, strike=None)
    psi_star = psi_star_factory({"tol_indicator": 1e-12})

    from src.solver import solve_dual_penalized
    V = 0.10
    cfg = {"solver": {"eta0": 1e2, "eta_growth": 5.0, "eps_obs": 1e-6, "eps_curv": 1e-6,
                      "eps_val": 1e-8, "max_outer": 8, "backend": "OSQP"}}
    out = solve_dual_penalized(mu, nu, (x, y, t), payoff, psi_star, V, cfg)

    # build hedge
    from src.hedge import build_hedge
    hedge = build_hedge(out["u"], out["v"], out["phi"], out["lambda"], (x, y, t), market=None)

    # paths: start at a typical x (center), simulate a few
    P = _simulate_paths(F0=100.0, times=t, sigma=0.6, n_paths=4, seed=11)

    from src.certificate import check_certificate
    stats = check_certificate(P, t, hedge, payoff, tol=5e-3)
    assert stats["violations"] == 0, f"certificate violated on {stats['violations']} paths, max gap={stats['max_gap']:.2e}"

if __name__ == "__main__":
    test_certificate_end_to_end_small()
    print("OK: certificate end-to-end (small)")
