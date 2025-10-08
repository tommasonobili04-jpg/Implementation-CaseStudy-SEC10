from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from scipy import sparse

# NOTE:
# - Builds discrete operators for the dual HJB–SPDE:
#   * Dxx: second-derivative matrix on non-uniform x-grid (for each time slice).
#   * P_yx: interpolation matrix mapping values on x-grid to y-grid at T (phi(1,y) ≈ P_yx @ phi_T(x)).
#   * psi_xy: obstacle right-hand side psi(y - kappa x) on (x,y) grid pairs.
# - Returns a dictionary consumed by solver.py to set up the penalized program.


def _dxx_matrix_nonuniform(x: np.ndarray) -> sparse.csr_matrix:
    """
    Tridiagonal second-derivative on a non-uniform grid (vectorized).
    Interior diagonals are negative (concavity of discrete Laplacian).
    """
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size < 3 or not np.all(np.diff(x) > 0):
        raise ValueError("x must be 1D, strictly increasing, length >= 3")

    dx = np.diff(x)
    M = x.size

    # interior coefficients
    im1 = 2.0 / ((dx[:-1] + dx[1:]) * dx[:-1])
    ip1 = 2.0 / ((dx[:-1] + dx[1:]) * dx[1:])
    ii  = -(im1 + ip1)

    main = np.zeros(M); sub = np.zeros(M-1); sup = np.zeros(M-1)
    main[1:-1] = ii
    sub[1:]    = im1
    sup[:-1]   = ip1

    # One-sided, second-order, non-uniform 3-point boundary rows
    dx1, dx2 = x[1] - x[0], x[2] - x[1]
    main[0] =  2.0 / (dx1 * (dx1 + dx2))
    sup[0]  = -2.0 / (dx1 * dx2)
    sup[1]  =  2.0 / (dx2 * (dx1 + dx2))

    dxm1, dxm2 = x[-1] - x[-2], x[-2] - x[-3]
    main[-1] =  2.0 / (dxm1 * (dxm1 + dxm2))
    sub[-1]  = -2.0 / (dxm1 * dxm2)
    sub[-2]  =  2.0 / (dxm2 * (dxm1 + dxm2))


    return sparse.diags([sub, main, sup], offsets=[-1, 0, 1], format="csr")



def _interp_matrix_linear(x_src: np.ndarray, y_tgt: np.ndarray) -> sparse.csr_matrix:
    """
    Linear interpolation matrix P such that (P @ f_src)(y_j) = f_src(y_j).
    Requires y within x support.
    """
    x = np.asarray(x_src, float); y = np.asarray(y_tgt, float)
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0)):
        raise ValueError("x_src must be strictly increasing; y_tgt non-decreasing")
    if y[0] < x[0] - 1e-12 or y[-1] > x[-1] + 1e-12:
        raise ValueError("y_tgt must lie within x_src support")

    idx = np.clip(np.searchsorted(x, y, side="right") - 1, 0, x.size - 2)
    x0 = x[idx]; x1 = x[idx + 1]
    w1 = (y - x0) / (x1 - x0)
    w0 = 1.0 - w1
    rows = np.arange(y.size)

    P = sparse.csr_matrix(
        (np.concatenate([w0, w1]),
         (np.concatenate([rows, rows]), np.concatenate([idx, idx + 1]))),
        shape=(y.size, x.size)
    )
    # Row-stochastic property up to machine epsilon (test downstream).
    return P



def build_dual_problem(mu: np.ndarray,
                       nu: np.ndarray,
                       grid_x: np.ndarray,
                       grid_y: np.ndarray,
                       grid_t: np.ndarray,
                       payoff,        # Callable [[x,y]] -> psi(y - kappa x)
                       psi_star,      # not used numerically here; kept for signature completeness
                       cfg: dict) -> Dict[str, object]:
    """
    Assemble discrete objects required by the penalized dual solver:
      - Dxx_block: block-diagonal second-derivative on phi over all time slices (size ((K+1)M) x ((K+1)M)).
      - P_yx: interpolation at terminal time (My x Mx) to evaluate phi(1, y).
      - psi_xy: obstacle RHS on Cartesian pairs (x_i, y_j) (shape (Mx, My)).
      - index helpers for phi at t0 and T in the flattened layout.

    Flattening convention for phi:
      phi_flat = [phi(t_0, x_1..x_M), phi(t_1, x_1..x_M), ..., phi(t_K, x_1..x_M)]
      length = (K+1) * Mx
    """
    x = np.asarray(grid_x, float)
    y = np.asarray(grid_y, float)
    t = np.asarray(grid_t, float)
    Mx = x.size; My = y.size; K = t.size - 1
    if K < 1:
        raise ValueError("grid_t must contain at least two points t0 < ... < T")

    # Operators
    Dxx_single = _dxx_matrix_nonuniform(x)        # (Mx, Mx)
    I_time = sparse.eye(K + 1, format="csr")
    Dxx_block = sparse.kron(I_time, Dxx_single, format="csr")  # ((K+1)Mx, (K+1)Mx)

    P_yx = _interp_matrix_linear(x, y)            # (My, Mx)

    # Obstacle RHS on Cartesian product (vectorized)
    X, Y = np.meshgrid(x, y, indexing="ij")       # X: (Mx, My), Y: (Mx, My)
    psi_xy = payoff(X, Y)                         # (Mx, My), already (y - k x - K)_+

    # Indices for phi at t0 and T
    # - t0 slice occupies entries [0 : Mx)
    # - T  slice occupies entries [K*Mx : (K+1)*Mx)
    idx_phi_t0 = np.arange(0, Mx, dtype=int)
    idx_phi_T  = np.arange(K * Mx, (K + 1) * Mx, dtype=int)

    # Sanity metadata
    meta = {
        "Mx": Mx, "My": My, "K": K,
        "shape_phi": ((K + 1) * Mx,),
        "x_min": float(x[0]), "x_max": float(x[-1]),
        "y_min": float(y[0]), "y_max": float(y[-1]),
    }

    return {
        "Dxx_single": Dxx_single.tocsr(),
        "Dxx_block": Dxx_block.tocsr(),
        "P_yx": P_yx.tocsr(),
        "psi_xy": np.asarray(psi_xy, float),
        "idx_phi_t0": idx_phi_t0,
        "idx_phi_T": idx_phi_T,
        "grid_x": x, "grid_y": y, "grid_t": t,
        "mu": np.asarray(mu, float), "nu": np.asarray(nu, float),
        "meta": meta,
    }
