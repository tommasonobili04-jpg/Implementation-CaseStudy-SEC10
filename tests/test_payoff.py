from __future__ import annotations
import numpy as np

def test_forward_start_factory_scalar_and_vector():
    from src.payoff import forward_start_factory
    payoff0 = forward_start_factory(kappa=1.0, strike=None)
    payoffK = forward_start_factory(kappa=1.0, strike=0.0)

    # Scalars
    x, y = 100.0, 105.0
    assert payoff0(x, y) == max(y - x, 0.0)
    assert payoffK(x, y) == payoff0(x, y)

    # Arrays with broadcasting
    x_arr = np.array([100.0, 100.0, 100.0])
    y_arr = np.array([95.0, 105.0, 120.0])
    out = payoff0(x_arr, y_arr)
    assert np.allclose(out, np.maximum(y_arr - x_arr, 0.0))

def test_psi_star_indicator():
    from src.payoff import psi_star_factory
    psi_star = psi_star_factory({"tol_indicator": 1e-12})

    lam = 0.3
    # h >= -2*lam -> 0
    for h in [0.0, -0.5, -0.6]:
        val = psi_star(h, lam)
        assert np.isfinite(val) and abs(val) < 1e-14

    # h < -2*lam -> +inf
    bad = psi_star(-1.0, lam)  # qui -1.0 < -0.6
    assert np.isinf(bad)
