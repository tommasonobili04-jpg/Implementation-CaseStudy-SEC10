from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Callable, Optional

# NOTE:
# - Breeden–Litzenberger on a non-uniform strike grid (forward calls -> densities).
# - Convex-order check via discrete call-transform inequality.
# - Minimal projection-to-no-arb: smooth/expand nu until μ ≼_cx ν within tolerance.

# ---------- BL density on non-uniform grid ----------
def _second_diff_nonuniform(K: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Scaled second difference on a non-uniform grid:
      C''(K_i) ≈ 2/(dKm + dKp) * ( (C_{i+1}-C_i)/dKp - (C_i-C_{i-1})/dKm ).
    """
    K = np.asarray(K, float)
    C = np.asarray(C, float)
    n = K.size
    if n < 3:
        raise ValueError("need at least 3 strikes for BL")
    if not np.all(np.diff(K) > 0):
        raise ValueError("K must be strictly increasing")

    out = np.zeros_like(C)
    dK = np.diff(K)
    # interior
    for i in range(1, n-1):
        dKm, dKp = dK[i-1], dK[i]
        term = ( (C[i+1]-C[i]) / dKp ) - ( (C[i]-C[i-1]) / dKm )
        out[i] = 2.0 * term / (dKm + dKp)
    # boundaries: one-sided quadratic estimate
    out[0]  = (C[2] - 2*C[1] + C[0]) / ((K[2]-K[1]) * (K[1]-K[0])) * 2.0
    out[-1] = (C[-1] - 2*C[-2] + C[-3]) / ((K[-1]-K[-2]) * (K[-2]-K[-3])) * 2.0
    return out


def breeden_litzenberger_forward(C_fwd: np.ndarray, K: np.ndarray,
                                 clip_neg: float = 0.0) -> np.ndarray:
    """
    Compute risk-neutral density q(K) = d^2 C_fwd / dK^2 on grid K (forward measure).
    Args:
        C_fwd: forward call prices on K.
        K: strictly increasing strikes.
        clip_neg: negatives are clipped to max(0, q) + clip_neg (tiny floor).
    Returns:
        q: density samples (non-normalized) on K (same shape as K).
    """
    q = _second_diff_nonuniform(K, C_fwd)
    q = np.maximum(q, clip_neg)
    return q


def _interval_weights_from_samples(K: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Integrate q over intervals to get probability mass per interval using trapezoid rule.
    Returns node-centered masses p with same length as K (endpoints get half-mass).
    """
    K = np.asarray(K, float)
    q = np.asarray(q, float)
    if K.size != q.size:
        raise ValueError("K and q must have the same length")
    if K.size < 2:
        raise ValueError("need at least 2 nodes")

    dK = np.diff(K)
    w_interval = 0.5 * (q[:-1] + q[1:]) * dK  # (n-1,)
    p = np.zeros_like(q)
    p[0]      += 0.5 * w_interval[0]
    p[1:-1]   += 0.5 * (w_interval[:-1] + w_interval[1:])
    p[-1]     += 0.5 * w_interval[-1]
    total = p.sum()
    if total <= 0:
        raise ValueError("zero or negative mass from BL; check inputs")
    p /= total
    return p


# ---------- Optional tails (simple, safe) ----------
def wing_extrapolation(K: np.ndarray, C_fwd: np.ndarray,
                       left_slope: Optional[float] = None,
                       right_slope: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal tail handling: by default returns inputs unchanged.
    If slopes are provided, extend one node on each side with linear tails.
    """
    if left_slope is None and right_slope is None:
        return K, C_fwd
    K = np.asarray(K, float)
    C = np.asarray(C_fwd, float)
    outK, outC = [K.copy(), C.copy()]
    if left_slope is not None:
        dK = K[1] - K[0]
        outK[0] = K[0] - dK
        outC[0] = max(0.0, C[0] + left_slope * (-dK))
    if right_slope is not None:
        dK = K[-1] - K[-2]
        outK[-1] = K[-1] + dK
        outC[-1] = max(0.0, C[-1] + right_slope * dK)
    return outK, outC


# ---------- Convex-order check ----------
def convex_order_check(mu_p: np.ndarray, x: np.ndarray,
                       nu_p: np.ndarray, y: np.ndarray,
                       K: Optional[np.ndarray] = None,
                       tol: float = 1e-8) -> bool:
    """
    Check μ ≼_cx ν via call-transform inequality:
      E_μ[(X-K)+] <= E_ν[(Y-K)+] for all K in a dense grid.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    mu_p = np.asarray(mu_p, float); nu_p = np.asarray(nu_p, float)
    if K is None:
        K = np.unique(np.concatenate([x, y]))
    K = np.asarray(K, float)

    def call_expect(p: np.ndarray, grid: np.ndarray) -> np.ndarray:
        # E[(Z-K)+] = sum_i (grid_i - K)+ * p_i  (discrete)
        G = np.maximum(grid[:, None] - K[None, :], 0.0)  # shape (n, m)
        return (G.T @ p).astype(float)  # (m,)

    C_mu = call_expect(mu_p, x)
    C_nu = call_expect(nu_p, y)
    return np.all(C_mu <= C_nu + tol)


# ---------- Minimal projection to enforce μ ≼_cx ν ----------
def _smooth_kernel(p: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Symmetric tri-diagonal smoothing kernel (conserves mass).
    """
    n = p.size
    q = p.copy()
    if n >= 3 and strength > 0:
        alpha = min(0.5, 0.2 * strength)
        q[1:-1] = (1-2*alpha)*p[1:-1] + alpha*(p[:-2] + p[2:])
        q[0]    = (1-alpha)*p[0] + alpha*p[1]
        q[-1]   = (1-alpha)*p[-1] + alpha*p[-2]
    q /= q.sum()
    return q

def noarb_call_fit(K: np.ndarray, bid: np.ndarray, ask: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """
    Convex & non-increasing fit within bid/ask on a non-uniform K grid:
    minimize ||C - mid||^2  s.t. vertical monotonicity, convexity (scaled second diff), and bid<=C<=ask.
    """
    import cvxpy as cp
    K = np.asarray(K, float); m = K.size
    C = cp.Variable(m)

    # Monotonicity: C_{i+1} - C_i <= 0
    D1 = np.eye(m, k=1) - np.eye(m)
    mono = D1[:-1] @ C <= 0

    # Convexity on non-uniform grid: scaled second diff >= 0
    dK = np.diff(K)
    rows = []
    for i in range(1, m-1):
        dKm, dKi = dK[i-1], dK[i]
        row = np.zeros(m)
        row[i-1] = -2.0 / (dKm * (dKm + dKi))
        row[i]   =  2.0 / (dKm + dKi) * (1.0/dKm + 1.0/dKi)
        row[i+1] = -2.0 / (dKi * (dKm + dKi))
        rows.append(row)
    A = np.asarray(rows)
    convex = A @ C >= 0

    constr = [mono, convex, C >= bid, C <= ask]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(C - mid)), constr)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError("no-arb call fit failed")
    return np.asarray(C.value).ravel()


def project_to_noarb(mu_p: np.ndarray, x: np.ndarray,
                     nu_p: np.ndarray, y: np.ndarray,
                     tol: float = 1e-8, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project ν to the closest ν' (L2) s.t. μ ≼_cx ν' on K = union(x,y).
    Keeps μ normalized, adjusts ν. OSQP primary; ECOS fallback.
    """
    import cvxpy as cp
    x = np.asarray(x, float); y = np.asarray(y, float)
    mu = np.maximum(mu_p, 0.0); mu /= mu.sum()
    nu = np.maximum(nu_p, 0.0); nu /= nu.sum()

    K = np.unique(np.concatenate([x, y]))
    Gx = np.maximum(x[:, None] - K[None, :], 0.0)
    Gy = np.maximum(y[:, None] - K[None, :], 0.0)

    nu_var = cp.Variable(y.size, nonneg=True)
    constr = [cp.sum(nu_var) == 1.0, Gx.T @ mu <= Gy.T @ nu_var + tol]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(nu_var - nu)), constr)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Convex-order projection failed: {prob.status}")
    return mu, np.asarray(nu_var.value).ravel()

def convex_order_certificate(mu_p: np.ndarray, x: np.ndarray,
                             nu_p: np.ndarray, y: np.ndarray,
                             K: np.ndarray | None = None) -> float:
    """
    Max call-transform violation: max_K { E_mu[(X-K)+] - E_nu[(Y-K)+] }.
    Non-positive => convex order holds.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    mu = np.asarray(mu_p, float); nu = np.asarray(nu_p, float)
    if K is None:
        K = np.unique(np.concatenate([x, y]))
    K = np.asarray(K, float)
    Gx = np.maximum(x[:, None] - K[None, :], 0.0)
    Gy = np.maximum(y[:, None] - K[None, :], 0.0)
    Cmu = (Gx.T @ mu)
    Cnu = (Gy.T @ nu)
    return float(np.max(Cmu - Cnu))


# ---------- High-level builder ----------
def build_marginals(quotes_t0: pd.DataFrame,
                    quotes_T: pd.DataFrame,
                    grid_x: np.ndarray,
                    grid_y: np.ndarray,
                    left_right_slopes: Tuple[Optional[float], Optional[float]] = (None, None),
                    clip_neg: float = 0.0,
                    tol_cx: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build μ, ν on grids (x=grid_x, y=grid_y) from forward call quotes at t0 and T=1.
    Assumes quotes are forwardized (no discounting) and cleaned (see data_io.filter_hygiene).
    """
    # Interpolate quotes to target grids (within support) and run BL
    def _interp_mid(df: pd.DataFrame, K_grid: np.ndarray) -> np.ndarray:
        g = df.sort_values("K")
        K = g["K"].to_numpy(float); C = g["mid"].to_numpy(float)
        Kmin, Kmax = K[0], K[-1]
        sel = (K_grid >= Kmin) & (K_grid <= Kmax)
        if not sel.any():
            raise ValueError("target grid outside data support")
        Kg = K_grid[sel]
        Cg = np.interp(Kg, K, C)
        return Kg, Cg, sel

    # t0
    Kx, Cx, selx = _interp_mid(quotes_t0, grid_x)
    # T
    Ky, Cy, sely = _interp_mid(quotes_T, grid_y)

    # Optional wing adjustment (conservative, default: identity)
    if any(v is not None for v in left_right_slopes):
        Kx, Cx = wing_extrapolation(Kx, Cx, left_right_slopes[0], left_right_slopes[1])
        Ky, Cy = wing_extrapolation(Ky, Cy, left_right_slopes[0], left_right_slopes[1])

    # In-spread repair before BL to enforce monotonicity/convexity
    if isinstance(quotes_t0, pd.DataFrame):
        g0 = quotes_t0.sort_values("K")
        Cx = noarb_call_fit(
            Kx,
            np.interp(Kx, g0["K"], g0["bid"]),
            np.interp(Kx, g0["K"], g0["ask"]),
            Cx
        )
    if isinstance(quotes_T, pd.DataFrame):
        g1 = quotes_T.sort_values("K")
        Cy = noarb_call_fit(
            Ky,
            np.interp(Ky, g1["K"], g1["bid"]),
            np.interp(Ky, g1["K"], g1["ask"]),
            Cy
        )

    qx = breeden_litzenberger_forward(Cx, Kx, clip_neg=clip_neg)
    qy = breeden_litzenberger_forward(Cy, Ky, clip_neg=clip_neg)


    mu_full = np.zeros_like(grid_x, dtype=float)
    nu_full = np.zeros_like(grid_y, dtype=float)
    mu_comp = _interval_weights_from_samples(Kx, qx)
    nu_comp = _interval_weights_from_samples(Ky, qy)
    mu_full[selx] = mu_comp
    nu_full[sely] = nu_comp

    # Project to convex order if needed
    mu_adj, nu_adj = project_to_noarb(mu_full, grid_x, nu_full, grid_y, tol=tol_cx)
    return mu_adj, nu_adj
