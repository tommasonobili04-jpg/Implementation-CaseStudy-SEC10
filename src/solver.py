from __future__ import annotations
import numpy as np
import cvxpy as cp
from typing import Dict, Tuple
from scipy import sparse

from .logging_kpi import init_logger
try:
    from .utils import timer
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def timer(*args, **kwargs):
        yield

def _try_solve(prob: cp.Problem, order: Tuple) -> str:
    """Prova solver in ordine; ritorna lo status finale."""
    for name, opts in order:
        try:
            prob.solve(solver=name, **opts)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return prob.status
        except Exception:
            pass
    return prob.status

def solve_dual_penalized(mu: np.ndarray,
                         nu: np.ndarray,
                         grids: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         payoff, psi_star,
                         V: float,
                         cfg: dict) -> Dict[str, object]:
    """
    minimize   <mu,u> + <nu,v> + V*lambda + eps_reg*(||phi||^2 + ||u||^2 + ||v||^2)
    s.t.       u_i + v_j + phi(t0,x_i) - phi(1,y_j) >= psi_ij
               Dxx phi + 2*lambda >= 0,  lambda >= 0,
               u[0] == 0,  sum(nu * v) == 0
    """
    logger = init_logger("solver")
    x, y, t = grids
    solver_cfg = dict(cfg.get("solver", {}))
    backend = str(solver_cfg.get("backend", "ECOS")).upper()
    max_iters = int(solver_cfg.get("max_iters", 200000))
    eps_val = float(solver_cfg.get("eps_val", 1e-8))
    eps_reg = float(solver_cfg.get("eps_reg", 1e-4))  # più “forte” per stabilizzare OSQP

    from .dual_spde import build_dual_problem
    D = build_dual_problem(mu, nu, x, y, t, payoff, psi_star, cfg)

    Mx, My, K = D["meta"]["Mx"], D["meta"]["My"], D["meta"]["K"]
    Dxx_block: sparse.csr_matrix = D["Dxx_block"]
    P_yx: sparse.csr_matrix = D["P_yx"]
    psi_xy: np.ndarray = D["psi_xy"]
    idx_phi_t0: np.ndarray = D["idx_phi_t0"]
    idx_phi_T: np.ndarray = D["idx_phi_T"]

    u = cp.Variable(Mx)
    v = cp.Variable(My)
    phi = cp.Variable((K + 1) * Mx)
    lam = cp.Variable(nonneg=True)

    phi_t0 = phi[idx_phi_t0]
    phi_Ty = P_yx @ phi[idx_phi_T]

    U = cp.reshape(u, (Mx, 1))
    Vv = cp.reshape(v, (1, My))
    PHI0 = cp.reshape(phi_t0, (Mx, 1))
    PHI1 = cp.reshape(phi_Ty, (1, My))

    constr = []
    constr += [(U + Vv + PHI0 - PHI1) >= psi_xy]
    constr += [(Dxx_block @ phi + 2.0 * lam) >= 0]
    constr += [u[0] == 0]                                    # gauge #1
    constr += [cp.sum(cp.multiply(nu, v)) == 0]              # gauge #2

    obj = cp.Minimize(
        mu @ u + nu @ v + V * lam
        + eps_reg * (cp.sum_squares(phi) + cp.sum_squares(u) + cp.sum_squares(v))
    )

    prob = cp.Problem(obj, constr)

    # ordine: backend richiesto, poi ECOS, poi SCS, poi OSQP
    orders = []
    if backend == "OSQP":
        orders.append((cp.OSQP, {"verbose": False, "max_iter": max_iters, "eps_abs": eps_val, "eps_rel": eps_val, "polishing": True}))
    elif backend == "SCS":
        orders.append((cp.SCS, {"verbose": False, "max_iters": max_iters, "eps": eps_val}))
    else:
        orders.append((cp.ECOS, {"verbose": False, "max_iters": max_iters, "abstol": eps_val, "reltol": eps_val}))

    # fallback robusti
    orders.extend([
        (cp.ECOS, {"verbose": False, "max_iters": max_iters, "abstol": eps_val, "reltol": eps_val}),
        (cp.SCS,  {"verbose": False, "max_iters": max_iters, "eps": eps_val}),
        (cp.OSQP, {"verbose": False, "max_iter": max_iters, "eps_abs": eps_val, "eps_rel": eps_val, "polishing": True}),
    ])

    status = _try_solve(prob, tuple(orders))
    if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Dual solve failed: status={status}")

    u_val = np.asarray(u.value, float)
    v_val = np.asarray(v.value, float)
    phi_val = np.asarray(phi.value, float)
    lam_val = float(lam.value)
    Pi_raw = float(mu @ u_val + nu @ v_val + V * lam_val)
    Pi = float(max(Pi_raw, 0.0))

    def _curv_residual(D, phi, lam):
        r = -(D["Dxx_block"] @ phi + 2.0 * lam)
        return float(np.maximum(r, 0.0).max())

    def _obs_residual(D, u, v, phi):
        Mx = D["meta"]["Mx"]; My = D["meta"]["My"]
        psi_xy: np.ndarray = D["psi_xy"]
        P_yx: sparse.csr_matrix = D["P_yx"]
        idx_phi_t0: np.ndarray = D["idx_phi_t0"]
        idx_phi_T: np.ndarray = D["idx_phi_T"]
        phi_t0 = phi[idx_phi_t0]
        phi_Ty = P_yx @ phi[idx_phi_T]
        lhs = (u.reshape(Mx,1) + v.reshape(1,My) + phi_t0.reshape(Mx,1) - phi_Ty.reshape(1,My))
        gap = psi_xy - lhs
        return float(np.maximum(gap, 0.0).max())

    metrics = {
        "lambda": lam_val,
        "Pi_raw": Pi_raw, "Pi": Pi,
        "eta_used": 0.0,
        "res_curv": _curv_residual(D, phi_val, lam_val),
        "res_obs": _obs_residual(D, u_val, v_val, phi_val),
        "outer_bracket": [0.0, 0.0],
    }
    return {"u": u_val, "v": v_val, "phi": phi_val, "lambda": lam_val, "Pi": Pi,
            "residuals": metrics}
