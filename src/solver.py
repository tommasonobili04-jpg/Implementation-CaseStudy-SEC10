from __future__ import annotations
import math
from typing import Dict, Tuple, Optional

import numpy as np
import cvxpy as cp
from scipy import sparse

from .logging_kpi import init_logger
from .utils import timer

# NOTE:
# - Penalized dual solver:
#   inner:   minimize  ∫u dμ + ∫v dν + λ V + η * ||pos( - (Dxx φ + 2λ) )||_2^2
#            s.t. obstacle:  u_i + v_j + φ(t0, x_i) - φ(1, y_j) >= ψ_xy[i,j]
#   outer:   1D convex (golden-section) search over λ >= 0.
# - Curvature feasibility is driven by penalty η; we escalate η until residual ≤ eps_curv.


def _build_inner_problem(D: Dict[str, object], V: float, eta: float):
    """
    Build a CVXPY problem template for fixed λ (as Parameter). Variables: u, v, phi.
    Returns (problem, vars, params) so we can update lambda and re-solve warm-started.
    """
    x = D["grid_x"]; y = D["grid_y"]; t = D["grid_t"]
    mu = D["mu"]; nu = D["nu"]
    Mx = x.size; My = y.size; K = t.size - 1

    Dxx_block: sparse.csr_matrix = D["Dxx_block"]  # ((K+1)Mx, (K+1)Mx)
    P_yx: sparse.csr_matrix = D["P_yx"]            # (My, Mx)
    psi_xy: np.ndarray = D["psi_xy"]               # (Mx, My)

    idx_phi_t0: np.ndarray = D["idx_phi_t0"]
    idx_phi_T:  np.ndarray = D["idx_phi_T"]

    # Variables
    u  = cp.Variable(Mx)
    v  = cp.Variable(My)
    phi = cp.Variable((K + 1) * Mx)

    # Parameter: lambda
    lam = cp.Parameter(nonneg=True, value=0.0)

    # Slices of phi
    phi_t0 = cp.vstack([phi[idx] for idx in idx_phi_t0]).flatten()  # shape (Mx,)
    phi_Tx = cp.vstack([phi[idx] for idx in idx_phi_T]).flatten()   # shape (Mx,)
    phi_Ty = cp.matmul(P_yx, phi_Tx)                                # shape (My,)

    # Obstacle constraints (broadcast as matrix inequality): u_i+v_j+phi_t0_i-phi_Ty_j >= psi_ij
    U = cp.reshape(u, (Mx, 1))
    Vv = cp.reshape(v, (1, My))
    PHI0 = cp.reshape(phi_t0, (Mx, 1))
    PHI1 = cp.reshape(phi_Ty, (1, My))
    constr_obs = U + Vv + PHI0 - PHI1 >= psi_xy

    # Curvature penalty: r = -(Dxx phi + 2λ), penalize ||pos(r)||_2^2
    Dxx_phi = Dxx_block @ phi
    resid = -(Dxx_phi + 2.0 * lam)
    pen = cp.sum_squares(cp.pos(resid))

    # Objective (penalized)
    obj = cp.sum(cp.multiply(mu, u)) + cp.sum(cp.multiply(nu, v)) + lam * V + eta * pen

    prob = cp.Problem(cp.Minimize(obj), [constr_obs])
    return prob, {"u": u, "v": v, "phi": phi}, {"lam": lam}


def _solve_inner(prob: cp.Problem,
                 vars: Dict[str, cp.Variable],
                 params: Dict[str, cp.Parameter],
                 lam_value: float,
                 solver: str = "OSQP",
                 max_iters: int = 200000,
                 eps_abs: float = 1e-8,
                 eps_rel: float = 1e-8) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Solve the inner problem for a fixed lambda value. Returns (objective_value, solution_dict).
    """
    params["lam"].value = float(lam_value)

    # Do not overwrite variable values here; let the solver warm-start from prior iterates.
    # (Previously we set v.value = None, which prevented warm-starting.)

    opts: Dict[str, object] = {}
    if solver.upper() == "OSQP":
        s = cp.OSQP
        opts = {
            "max_iter": max_iters,
            "eps_abs": eps_abs,
            "eps_rel": eps_rel,
            "polish": True,
            "verbose": False,
            "warm_start": True,
        }
    elif solver.upper() == "ECOS":
        s = cp.ECOS
        opts = {
            "max_iters": max_iters,
            "abstol": eps_abs,
            "reltol": eps_rel,
            "feastol": 1e-8,
            "verbose": False,
        }
    else:
        s = cp.SCS
        opts = {
            "max_iters": max_iters,
            "eps": max(eps_abs, eps_rel),
            "verbose": False,
            "warm_start": True,
        }

    prob.solve(solver=s, **opts)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Inner solve failed at lambda={lam_value}: status={prob.status}")

    sol = {k: np.asarray(v.value, dtype=float) for k, v in vars.items()}
    return float(prob.value), sol



def _curvature_residual(D: Dict[str, object], phi: np.ndarray, lam: float) -> float:
    """
    Max positive violation of curvature constraint: max(0, -(Dxx phi + 2 λ)).
    """
    Dxx_block: sparse.csr_matrix = D["Dxx_block"]
    r = -(Dxx_block @ phi + 2.0 * lam)
    return float(np.maximum(r, 0.0).max())


def _obstacle_residual(D: Dict[str, object], u: np.ndarray, v: np.ndarray, phi: np.ndarray) -> float:
    """
    Max positive violation of obstacle: max(0, ψ - (u+v+phi_t0-phi_Ty)).
    """
    Mx = D["meta"]["Mx"]; My = D["meta"]["My"]
    psi_xy: np.ndarray = D["psi_xy"]
    P_yx: sparse.csr_matrix = D["P_yx"]
    idx_phi_t0: np.ndarray = D["idx_phi_t0"]
    idx_phi_T:  np.ndarray = D["idx_phi_T"]

    phi_t0 = phi[idx_phi_t0]             # (Mx,)
    phi_Tx = phi[idx_phi_T]              # (Mx,)
    phi_Ty = P_yx @ phi_Tx               # (My,)

    lhs = (u.reshape(Mx, 1) + v.reshape(1, My) + phi_t0.reshape(Mx, 1) - phi_Ty.reshape(1, My))
    gap = psi_xy - lhs
    return float(np.maximum(gap, 0.0).max())


def _golden_section_minimize(eval_fn, a: float, b: float, iters: int = 18) -> Tuple[float, float]:
    """
    Golden-section search for unimodal convex objective on [a,b].
    Returns (x_star, f(x_star)).
    """
    invphi = (math.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1/phi^2
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)
    fc = eval_fn(c)
    fd = eval_fn(d)
    for _ in range(iters):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = a + invphi2 * (b - a)
            fc = eval_fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = eval_fn(d)
    if fc <= fd:
        return c, fc
    return d, fd


def solve_dual_penalized(mu: np.ndarray,
                         nu: np.ndarray,
                         grids: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         payoff, psi_star,
                         V: float,
                         cfg: dict) -> Dict[str, object]:
    """
    One-shot convex formulation:
        minimize   <mu,u> + <nu,v> + V*lambda
        subject to u_i + v_j + phi(t0,x_i) - phi(1,y_j) >= psi_ij
                  Dxx phi + 2*lambda >= 0,    lambda >= 0
    """
    logger = init_logger("solver")
    x, y, t = grids

    # Build discrete operators & RHS once
    from .dual_spde import build_dual_problem
    D = build_dual_problem(mu, nu, x, y, t, payoff, psi_star, cfg)

    Mx, My, K = D["meta"]["Mx"], D["meta"]["My"], D["meta"]["K"]
    Dxx_block: sparse.csr_matrix = D["Dxx_block"]
    P_yx: sparse.csr_matrix = D["P_yx"]
    psi_xy: np.ndarray = D["psi_xy"]
    idx_phi_t0: np.ndarray = D["idx_phi_t0"]
    idx_phi_T: np.ndarray = D["idx_phi_T"]

    # Variables
    u = cp.Variable(Mx)
    v = cp.Variable(My)
    phi = cp.Variable((K + 1) * Mx)
    lam = cp.Variable(nonneg=True)

    # Slices
    phi_t0 = phi[idx_phi_t0]
    phi_Tx = phi[idx_phi_T]
    phi_Ty = P_yx @ phi_Tx

    # Obstacle constraint
    U = cp.reshape(u, (Mx, 1))
    Vv = cp.reshape(v, (1, My))
    PHI0 = cp.reshape(phi_t0, (Mx, 1))
    PHI1 = cp.reshape(phi_Ty, (1, My))
    constr_obs = (U + Vv + PHI0 - PHI1) >= psi_xy

    # Curvature feasibility
    constr_curv = (Dxx_block @ phi + 2.0 * lam) >= 0

    # Objective
    obj = cp.Minimize(mu @ u + nu @ v + V * lam)

    # Solver options
    backend = str(cfg.get("solver", {}).get("backend", "ECOS")).upper()
    solver = getattr(cp, backend, cp.ECOS)
    opts = {"verbose": False}
    if solver is cp.OSQP:
        opts.update({
            "max_iter": int(cfg.get("solver", {}).get("max_iters", 200000)),
            "eps_abs": float(cfg.get("solver", {}).get("eps_val", 1e-8)),
            "eps_rel": float(cfg.get("solver", {}).get("eps_val", 1e-8)),
        })

    prob = cp.Problem(obj, [constr_obs, constr_curv])
    prob.solve(solver=solver, **opts)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Dual solve failed: status={prob.status}")

    u_val = np.asarray(u.value, float)
    v_val = np.asarray(v.value, float)
    phi_val = np.asarray(phi.value, float)
    lam_val = float(lam.value)
    Pi = float(mu @ u_val + nu @ v_val + V * lam_val)

    # Diagnostics (consistent with the rest of the file)
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
        "Pi": Pi,
        "eta_used": 0.0,  # legacy key
        "res_curv": _curv_residual(D, phi_val, lam_val),
        "res_obs": _obs_residual(D, u_val, v_val, phi_val),
        "outer_bracket": [0.0, 0.0],  # legacy key
    }
    logger.info(f"lambda*={lam_val:.6f}, Pi={Pi:.8f}, "
                f"curv={metrics['res_curv']:.2e}, obs={metrics['res_obs']:.2e}")
    return {"u": u_val, "v": v_val, "phi": phi_val, "lambda": lam_val, "Pi": Pi,
            "residuals": metrics}
