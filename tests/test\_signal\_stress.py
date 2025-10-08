from __future__ import annotations
import numpy as np
import pandas as pd

def _dummy_quotes(Kmin=60.0, Kmax=140.0, n=81, T=1.0, seed=0):
    rng = np.random.default_rng(seed)
    K = np.linspace(Kmin, Kmax, n)
    base = np.maximum(100.0 - K, 0.0) + 5.0  # monotone decreasing proxy
    noise = 0.02 * base * (rng.random(size=n) - 0.5)
    mid = np.maximum(base + noise, 0.001)
    return pd.DataFrame({"maturity": T, "K": K, "bid": mid*0.999, "ask": mid*1.001, "mid": mid})

def test_signal_and_stress_scaffold():
    from src.signal import robust_gap, after_cost_edge, pnl_vol_estimate, size_kelly, size_sharpe, execution_sheet
    from src.stress import run_stress

    # Synthetic numbers for signal
    Pi = 1.00
    Pmkt = 1.08
    gap = robust_gap(Pmkt, Pi)
    C_stat, C_var, C_delta = 0.02, 0.01, 0.005
    edge_ac = after_cost_edge(gap, C_stat, C_var, C_delta)
    assert edge_ac > 0.0

    # PnL vol from toy paths and hedge (lambda=0, zero theta) -> should be ~0
    times = np.linspace(0.5, 1.0, 11)
    paths = np.tile(np.linspace(100, 100, 11), (3,1))  # flat paths
    hedge = {"lambda": 0.0, "delta": {"theta": lambda t,f: 0.0}}
    sigma_unit = pnl_vol_estimate(paths, times, hedge)
    assert abs(sigma_unit) < 1e-12

    # Sizing
    s_k = size_kelly(edge_ac, sigma_unit + 0.1, kelly_fraction=0.25)  # avoid zero denom
    s_s = size_sharpe(edge_ac, sigma_unit + 0.1, inv_sr=0.5)
    assert s_k >= 0.0 and s_s >= 0.0

    # Stress runner with stubbed pipeline_fn
    q0 = _dummy_quotes(T=0.5); qT = _dummy_quotes(T=1.0)
    def pipeline_fn(quotes_t0, quotes_T, Vh, cfg):
        # Simple proxy: Π increases slightly with +dv via higher mids
        Pi0 = float(quotes_t0["mid"].mean())
        Pi1 = float(quotes_T["mid"].mean())
        Pi_est = 0.5 * (Pi0 + Pi1)
        return {"Pi": Pi_est, "lambda": 0.3}
    df = run_stress((q0, qT), V=0.12, cfg={}, pipeline_fn=pipeline_fn,
                    vol_bumps=[-1.0, 0.0, +1.0], haircuts=[0.0, 0.05])
    assert {"dv","haircut","Pi","lambda"}.issubset(df.columns)
    assert df.shape[0] == 6  # 3 bumps x 2 haircuts

if __name__ == "__main__":
    test_signal_and_stress_scaffold()
    print("OK: signal + stress scaffolding")
