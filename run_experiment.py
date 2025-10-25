from __future__ import annotations
import os, sys, json
import numpy as np
import yaml
from typing import Dict, Any, Tuple

# Make 'src' importable when run from project root
_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from src.utils import set_seed, save_json, ensure_dir
from src.logging_kpi import init_logger, kpi_dump
from src.grids import build_space_grid, build_time_grid
from src.data_io import load_quotes, filter_hygiene, calendar_filter, normalize_to_grid
from src.marginals import build_marginals
from src.payoff import forward_start_factory, psi_star_factory
from src.solver import solve_dual_penalized
from src.hedge import build_hedge, estimate_turnover
from src.certificate import check_certificate
from src.costs import static_cost_from_market, variance_leg_cost, delta_cost_estimate
from src.signal import (market_forward_start_price, robust_gap, after_cost_edge,
                        pnl_vol_estimate, size_kelly, size_sharpe, execution_sheet)
from src.plots import plot_Pi_vs_V, plot_phi_slices, plot_market_vs_robust
import matplotlib.pyplot as plt


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _spot_proxy_from_quotes(df) -> float:
    # Safe, deterministic proxy if spot/forward not provided: median strike
    return float(np.median(df["K"].to_numpy(float)))


def _simulate_paths(F0: float, times: np.ndarray, sigma: float, n_paths: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.asarray(times, float)
    P = np.zeros((n_paths, t.size), dtype=float)
    for p in range(n_paths):
        f = np.empty_like(t)
        f[0] = F0
        for k in range(t.size - 1):
            dt = t[k+1] - t[k]
            z = rng.standard_normal()
            f[k+1] = f[k] + sigma * np.sqrt(dt) * z  # ABM proxy
        P[p] = f
    return P


def _build_all(quotes_t0, quotes_T, cfg: Dict[str, Any]) -> Dict[str, Any]:
    logger = init_logger("run")
    set_seed(int(cfg.get("seed", 42)))

    # Grids
    t0_val = float(quotes_t0["maturity"].iloc[0])
    T_val  = float(quotes_T["maturity"].iloc[0])
    K_time = int(cfg["grids"]["K_time"])
    t = build_time_grid(t0_val, T_val, K_time)

    # Spot/forward proxy for grid centering and linear legs
    F0 = float(cfg.get("spot", _spot_proxy_from_quotes(quotes_t0)))

    x = build_space_grid(F0, cfg["grids"]["x_atm"], cfg["grids"]["x_tail"])
    y = build_space_grid(F0, cfg["grids"]["y_atm"], cfg["grids"]["y_tail"])

    # Marginals
    mu, nu = build_marginals(quotes_t0, quotes_T, x, y, clip_neg=0.0, tol_cx=1e-7)

    # Payoff and dual solver
    payoff = forward_start_factory(cfg["payoff"]["kappa"], cfg["payoff"]["strike"])
    psi_star = psi_star_factory(cfg.get("psi", {}))

    out = solve_dual_penalized(mu, nu, (x, y, t), payoff, psi_star, V=float(cfg["budget"]["V"]), cfg=cfg)

    # Hedge stack
    hedge = build_hedge(out["u"], out["v"], out["phi"], out["lambda"], (x, y, t), market=cfg.get("market"))

    bundle = {"mu": mu, "nu": nu, "x": x, "y": y, "t": t, "F0": F0, "out": out, "hedge": hedge, "payoff": payoff}
    return bundle


def main(cfg_path: str) -> Dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    logger = init_logger("main")
    set_seed(int(cfg.get("seed", 42)))

    # Load and clean quotes
    q0_raw = load_quotes(cfg["paths"]["t0"])
    qT_raw = load_quotes(cfg["paths"]["T"])
    q0 = filter_hygiene(q0_raw)
    qT = filter_hygiene(qT_raw)
    q0, qT = calendar_filter(q0, qT)

    # Build/solve
    pack = _build_all(q0, qT, cfg)
    out = pack["out"]; hedge = pack["hedge"]
    x, y, t = pack["x"], pack["y"], pack["t"]

    # Certificate check on simulated paths (optional but recommended)
    if "sim" in cfg:
        P = _simulate_paths(F0=pack["F0"], times=t,
                            sigma=float(cfg["sim"].get("sigma", 0.6)),
                            n_paths=int(cfg["sim"].get("n_paths", 8)),
                            seed=int(cfg["sim"].get("seed", 11)))
        cert = check_certificate(P, t, hedge, pack["payoff"], tol=float(cfg.get("tol", {}).get("path", 5e-3)))
    else:
        cert = {"violations": 0, "max_gap": 0.0}

    # Market vs robust, costs, signal, sizing
    fs_px = cfg.get("market", {}).get("forward_start_price", None)
    P_mkt = market_forward_start_price(q0, qT, cfg["payoff"]["kappa"], cfg["payoff"]["strike"], fallback=fs_px)
    Pi = float(out["Pi"])
    gap = robust_gap(P_mkt, Pi)

    # Static costs from quotes at t0 and T
    C_stat_u = static_cost_from_market(hedge["static_u"], q0, forward_level=pack["F0"], side="ask")
    C_stat_v = static_cost_from_market(hedge["static_v"], qT, forward_level=pack["F0"], side="ask")
    C_stat = float(C_stat_u + C_stat_v)

    # Variance leg cost
    C_var = variance_leg_cost(cfg.get("market", {}).get("var"), cfg.get("proxy_var", {}))

    # Delta cost estimate via turnover on simulated paths (if present)
    if "sim" in cfg:
        theta_fn = hedge["delta"]["theta"]
        turnovers = [estimate_turnover(p, t, theta_fn) for p in P]
        C_delta = delta_cost_estimate(np.array(turnovers), cfg["costs"]["delta_spread"], cfg["costs"].get("impact", {}))
        sigma_unit = pnl_vol_estimate(P, t, hedge)
        turnover_est = float(np.mean(turnovers))
    else:
        C_delta = 0.0
        sigma_unit = 0.0
        turnover_est = 0.0

    edge_ac = after_cost_edge(gap, C_stat, C_var, C_delta)
    # Sizing (Sharpe-target preferred for deployment; Kelly as reference)
    size_sr = size_sharpe(edge_ac, max(sigma_unit, 1e-8), cfg["sizing"]["inv_sr"])
    size_k = size_kelly(edge_ac, max(sigma_unit, 1e-8), cfg["sizing"]["kelly_fraction"])
    sheet = execution_sheet(edge_ac, size_sr, out["lambda"], turnover_est, limits=cfg.get("limits", {}))

    # Save artifacts
    results_dir = cfg.get("results_dir", "results")
    figs_dir = os.path.join(results_dir, "figs")
    ensure_dir(results_dir); ensure_dir(figs_dir)

    # KPIs
    kpis = {
        "Pi": Pi, "lambda": float(out["lambda"]),
        "edge_after_cost": float(edge_ac),
        "sigma_unit": float(sigma_unit),
        "size_sharpe": float(size_sr),
        "size_kelly": float(size_k),
        "C_stat": float(C_stat), "C_var": float(C_var), "C_delta": float(C_delta),
        "cert_violations": int(cert["violations"]), "cert_max_gap": float(cert["max_gap"]),
    }
    save_json(os.path.join(results_dir, "results.json"), {**kpis, "residuals": out["residuals"]})
    kpi_dump(os.path.join(results_dir, "kpi.json"), kpis)

    # Plots (optional)
    try:
        fig = plot_market_vs_robust(P_mkt, Pi); fig.savefig(os.path.join(figs_dir, "market_vs_robust.pdf")); plt.close(fig)
        # phi slices: t0 and T
        M = x.size; K = t.size - 1; phi = out["phi"].reshape(K+1, M)
        fig = plot_phi_slices(x, [("t0", phi[0]), ("T", phi[-1])]); fig.savefig(os.path.join(figs_dir, "phi_slices.pdf")); plt.close(fig)
    except Exception as e:
        logger.warning(f"Plotting skipped: {e}")

    return {"out": out, "hedge": hedge, "kpis": kpis, "cert": cert, "paths_used": cfg.get("sim") is not None}


# Pipeline function compatible with stress.run_stress (signature required)
def main_inner(quotes_t0, quotes_T, V: float, cfg: Dict[str, Any]) -> Dict[str, float]:
    # Clone cfg with overridden budget V
    cfg2 = dict(cfg); cfg2 = {**cfg2, "budget": {**cfg.get("budget", {}), "V": float(V)}}
    pack = _build_all(quotes_t0, quotes_T, cfg2)
    return {"Pi": float(pack["out"]["Pi"]), "lambda": float(pack["out"]["lambda"])}


if __name__ == "__main__":
    # Example: python run_experiment.py configs/example.yaml
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", type=str, help="Path to YAML config")
    args = ap.parse_args()
    res = main(args.cfg)
    print(json.dumps(res["kpis"], indent=2))
