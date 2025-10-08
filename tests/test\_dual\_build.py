from __future__ import annotations
import numpy as np

def _toy_payoff(kappa=1.0, K=0.0):
    def f(x, y):
        return np.maximum(y - kappa * x - K, 0.0)
    return f

def test_dual_build_shapes_and_ops():
    from scipy import sparse
    from src.dual_spde import build_dual_problem

    # minimal grids
    x = np.linspace(80.0, 120.0, 51)
    y = np.linspace(75.0, 125.0, 61)
    t = np.linspace(0.5, 1.0, 6)   # 5 steps (K=5)

    # simple marginals (dummy normalized)
    mu = np.ones_like(x); mu /= mu.sum()
    nu = np.ones_like(y); nu /= nu.sum()

    out = build_dual_problem(mu, nu, x, y, t, payoff=_toy_payoff(), psi_star=lambda h,lam:0.0, cfg={})

    # shapes
    Mx, My, K = out["meta"]["Mx"], out["meta"]["My"], out["meta"]["K"]
    assert out["Dxx_single"].shape == (Mx, Mx)
    assert out["Dxx_block"].shape == ((K+1)*Mx, (K+1)*Mx)
    assert out["P_yx"].shape == (My, Mx)
    assert out["psi_xy"].shape == (Mx, My)
    assert out["idx_phi_t0"].size == Mx and out["idx_phi_T"].size == Mx

    # operator sanity
    assert sparse.isspmatrix_csr(out["Dxx_single"])
    assert sparse.isspmatrix_csr(out["Dxx_block"])
    assert sparse.isspmatrix_csr(out["P_yx"])

    # interpolation conservativity: row sums ~ 1
    row_sums = np.asarray(out["P_yx"].sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-12)

    # finite difference: interior diagonals negative
    D = out["Dxx_single"].toarray()
    diag = np.diag(D)
    assert (diag[1:-1] < 0).all()

if __name__ == "__main__":
    test_dual_build_shapes_and_ops()
    print("OK: dual build (operators, shapes)")
