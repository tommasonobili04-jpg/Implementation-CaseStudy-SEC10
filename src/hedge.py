from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Tuple

from .utils import interp1_strict

# NOTE:
# - static_buckets_from_potential: approximate a tradable static replication for u or v
#   via nonnegative curvature weights on the strike grid (call-spread representation).
# - delta_schedule_from_phi: computes -∂_x φ(t, x) on the (t, x) grid and returns an
#   interpolating function θ(t, f) for dynamic hedging.
# - realized_variance: simple discrete quadratic-variation proxy.
# - build_hedge: assembles the hedge stack and fast evaluators used by the certificate.
# - estimate_turnover: delta turnover proxy (for cost estimation).

def _dx_nonuniform(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    First derivative on a non-uniform grid using centered differences (one-sided at boundaries).
    """
    x = np.asarray(x, float); f = np.asarray(f, float)
    n = x.size
    if n < 2: raise ValueError("need n>=2")
    fx = np.empty_like(f)

    # interior (weighted two-slope average)
    for i in range(1, n-1):
        dl = x[i] - x[i-1]; dr = x[i+1] - x[i]
        sl = (f[i] - f[i-1]) / dl
        sr = (f[i+1] - f[i]) / dr
        fx[i] = (dr*sl + dl*sr) / (dl + dr)

    # boundaries: one-sided
    fx[0]  = (f[1] - f[0]) / (x[1] - x[0])
    fx[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    return fx


def _node_masses_from_curvature(x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Construct nonnegative call weights from (clipped) curvature.
    Returns (K, w_call, alpha0, alpha1) for representation:
      u(f) ~ alpha0 + alpha1 f + sum_j w_call[j] * (f - K[j])_+.
    """
    x = np.asarray(x, float); u = np.asarray(u, float)
    if x.size < 3: raise ValueError("need >=3 nodes for curvature-based buckets")

    # Nonnegative curvature (clip negatives to zero to get a convex envelope)
    # Discrete curvature via second difference on nonuniform grid:
    dx = np.diff(x)
    curv = np.zeros_like(u)
    for i in range(1, x.size-1):
        dl = dx[i-1]; dr = dx[i]
        term = ( (u[i+1]-u[i]) / dr ) - ( (u[i]-u[i-1]) / dl )
        curv[i] = 2.0 * term / (dl + dr)
    # one-sided estimates at boundaries:
    curv[0]  = (u[2] - 2*u[1] + u[0]) / ((x[2]-x[1])*(x[1]-x[0])) * 2.0
    curv[-1] = (u[-1]-2*u[-2]+u[-3]) / ((x[-1]-x[-2])*(x[-2]-x[-3])) * 2.0

    curv = np.maximum(curv, 0.0)

    # Node "masses": integrate curvature over small cells around nodes
    dnode = np.empty_like(u)
    dnode[0]  = 0.5*(x[1]-x[0])
    dnode[-1] = 0.5*(x[-1]-x[-2])
    dnode[1:-1] = 0.5*(x[2:]-x[:-2])
    w_call = curv * dnode

    # Linear part: slope at left and intercept to match value at left
    alpha1 = (u[1]-u[0]) / (x[1]-x[0])
    alpha0 = u[0] - alpha1 * x[0]
    return x.copy(), w_call, float(alpha0), float(alpha1)


def static_buckets_from_potential(x: np.ndarray, u: np.ndarray, K: np.ndarray | None = None) -> Dict[str, np.ndarray | float]:
    """
    Build a static replication stack from potential samples (x, u(x)).
    """
    if K is not None and (K.ndim != 1 or not np.all(np.diff(K) > 0)):
        raise ValueError("K must be strictly increasing if provided")
    if K is None:
        K = np.asarray(x, float)
    Kw, w_call, a0, a1 = _node_masses_from_curvature(x, u)
    # Map weights to target strike grid (here we assume K==x; otherwise simple interp)
    if not np.allclose(K, Kw):
        # conservative mapping by linear interpolation of cumulative weights
        c_src = np.cumsum(w_call)
        c_src = c_src / (c_src[-1] + 1e-16)
        c_tgt = np.interp(K, Kw, c_src)
        w_call = np.diff(np.concatenate([[0.0], c_tgt])) * (w_call.sum())
    return {"K": np.asarray(K, float), "w_call": np.asarray(w_call, float), "alpha0": a0, "alpha1": a1}


def delta_schedule_from_phi(grid_t: np.ndarray, grid_x: np.ndarray, phi_flat: np.ndarray) -> Dict[str, object]:
    """
    Compute θ(t, f) = -∂_x φ(t, f). Returns an interpolator:
      theta(t, f) with linear interpolation in both t and x.
    """
    t = np.asarray(grid_t, float)
    x = np.asarray(grid_x, float)
    K = t.size - 1
    M = x.size
    if phi_flat.size != (K+1)*M:
        raise ValueError("phi size mismatch")

    phi = phi_flat.reshape(K+1, M)
    phix = np.vstack([_dx_nonuniform(x, phi[k, :]) for k in range(K+1)])  # (K+1, M)

    def theta(time: float, f: float) -> float:
        # time interpolation
        if time <= t[0]:
            k0, k1, wt = 0, 0, 0.0
        elif time >= t[-1]:
            k0, k1, wt = K, K, 0.0
        else:
            k1 = int(np.searchsorted(t, time))
            k0 = k1 - 1
            wt = (time - t[k0]) / (t[k1] - t[k0])
        # x interpolation
        f = float(f)
        if f <= x[0]:
            v0 = phix[k0, 0]; v1 = phix[k1, 0]
        elif f >= x[-1]:
            v0 = phix[k0, -1]; v1 = phix[k1, -1]
        else:
            i1 = int(np.searchsorted(x, f))
            i0 = i1 - 1
            wx = (f - x[i0]) / (x[i1] - x[i0])
            v0 = (1-wx)*phix[k0, i0] + wx*phix[k0, i1]
            v1 = (1-wx)*phix[k1, i0] + wx*phix[k1, i1]
        phix_t = (1-wt)*v0 + wt*v1
        return -float(phix_t)  # θ = -φ_x

    return {"theta": theta, "phix": phix, "phi_grid_t": t, "phi_grid_x": x}


def realized_variance(path: np.ndarray, times: np.ndarray) -> float:
    """
    Discrete quadratic-variation proxy: sum (ΔF)^2 / Δt.
    """
    f = np.asarray(path, float)
    t = np.asarray(times, float)
    df = np.diff(f)
    dt = np.diff(t)
    if (dt <= 0).any():
        raise ValueError("times must be strictly increasing")
    return float(np.sum((df * df) / dt))


def delta_series(path: np.ndarray, times: np.ndarray, theta_fn: Callable[[float, float], float]) -> np.ndarray:
    """
    Evaluate θ_k = θ(t_k, F_{t_k}) along a path.
    """
    f = np.asarray(path, float); t = np.asarray(times, float)
    return np.array([theta_fn(tk, fk) for tk, fk in zip(t[:-1], f[:-1])], dtype=float)


def build_hedge(u: np.ndarray,
                v: np.ndarray,
                phi: np.ndarray,
                lambda_: float,
                grids: Tuple[np.ndarray, np.ndarray, np.ndarray],
                market: Dict[str, object] | None = None) -> Dict[str, object]:
    """
    Assemble hedge components and fast evaluators used by the certificate and execution.
    Returns a dict with:
      - 'static_u', 'static_v' (bucketized),
      - 'u_fn', 'v_fn' (piecewise-linear evaluators),
      - 'delta' (dict with 'theta' interpolator),
      - 'phi_t0', 'phi_Tx' (on x-grid), 'phi_Ty' (on y-grid) and their evaluators,
      - 'lambda' (variance leg notional),
      - 'grids'.
    """
    x, y, t = grids
    # Static buckets
    U = static_buckets_from_potential(x, u, K=x)
    Vb = static_buckets_from_potential(y, v, K=y)
    # Evaluators (piecewise-linear)
    u_fn = interp1_strict(x, u)
    v_fn = interp1_strict(y, v)
    # Delta schedule
    delta = delta_schedule_from_phi(t, x, phi)

    # φ at t0 and T
    M = x.size
    Kt = t.size - 1
    if phi.size != (Kt + 1) * M:
        raise ValueError("phi size mismatch with (t,x) grid")
    phi_t0 = phi[:M]
    phi_Tx = phi[Kt * M:(Kt + 1) * M]

    # Evaluate terminal slice on y-grid (consistent with dual interpolation P_yx)
    phi_Ty = np.interp(y, x, phi_Tx)

    # Correct evaluators
    phi_t0_fn = interp1_strict(x, phi_t0)
    phi_Tx_fn = interp1_strict(x, phi_Tx)   # <-- x-grid evaluator (fixed)
    phi_Ty_fn = interp1_strict(y, phi_Ty)   # <-- y-grid evaluator

    hedge = {
        "static_u": U,
        "static_v": Vb,
        "u_fn": u_fn,
        "v_fn": v_fn,
        "delta": delta,
        "phi_t0": phi_t0,
        "phi_Tx": phi_Tx,
        "phi_Ty": phi_Ty,
        "phi_t0_fn": phi_t0_fn,
        "phi_Tx_fn": phi_Tx_fn,
        "phi_Ty_fn": phi_Ty_fn,
        "lambda": float(lambda_),
        "grids": {"x": x, "y": y, "t": t},
        "market": market or {},
    }
    return hedge



def estimate_turnover(path: np.ndarray, times: np.ndarray, theta_fn: Callable[[float, float], float]) -> float:
    """
    Sum of absolute delta changes along the path: sum |θ_{k+1} - θ_k|.
    """
    theta = delta_series(path, times, theta_fn)
    dtheta = np.diff(theta)
    return float(np.sum(np.abs(dtheta)))
