from __future__ import annotations
import numpy as np
from typing import Dict
from scipy import sparse

def _dxx_matrix_nonuniform(x: np.ndarray) -> sparse.csr_matrix:
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size < 3 or not np.all(np.diff(x) > 0):
        raise ValueError("x must be 1D, strictly increasing, length >= 3")
    dx = np.diff(x); M = x.size
    im1 = 2.0 / ((dx[:-1] + dx[1:]) * dx[:-1])
    ip1 = 2.0 / ((dx[:-1] + dx[1:]) * dx[1:])
    ii  = -(im1 + ip1)

    main = np.zeros(M); sub = np.zeros(M-1); sup = np.zeros(M-1)
    main[1:-1] = ii
    sub[1:]    = im1
    sup[:-1]   = ip1

    # bordi one-sided 2° ordine non-uniformi
    dx1, dx2 = x[1]-x[0], x[2]-x[1]
    main[0] =  2.0 / (dx1 * (dx1 + dx2))
    sup[0]  = -2.0 / (dx1 * dx2)
    sup[1]  =  2.0 / (dx2 * (dx1 + dx2))
    dxm1, dxm2 = x[-1]-x[-2], x[-2]-x[-3]
    main[-1] =  2.0 / (dxm1 * (dxm1 + dxm2))
    sub[-1]  = -2.0 / (dxm1 * dxm2)
    sub[-2]  =  2.0 / (dxm2 * (dxm1 + dxm2))

    return sparse.diags([sub, main, sup], offsets=[-1,0,1], format="csr")

def _interp_matrix_linear(x_src: np.ndarray, y_tgt: np.ndarray) -> sparse.csr_matrix:
    x = np.asarray(x_src, float); y = np.asarray(y_tgt, float)
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0)):
        raise ValueError("x_src must be strictly increasing; y_tgt non-decreasing")

    Mx, My = x.size, y.size
    idx = np.searchsorted(x, y, side="right") - 1
    idx = np.clip(idx, 0, Mx-2)
    x0 = x[idx]; x1 = x[idx+1]
    denom = np.where((x1 - x0) <= 0, 1.0, (x1 - x0))
    w1 = np.clip((y - x0) / denom, 0.0, 1.0)
    w0 = 1.0 - w1

    rows = np.arange(My)
    P = sparse.csr_matrix(
        (np.concatenate([w0, w1]),
         (np.concatenate([rows, rows]), np.concatenate([idx, idx+1]))),
        shape=(My, Mx)
    )
    return P

def build_dual_problem(mu: np.ndarray,
                       nu: np.ndarray,
                       grid_x: np.ndarray,
                       grid_y: np.ndarray,
                       grid_t: np.ndarray,
                       payoff,
                       psi_star,
                       cfg: dict) -> Dict[str, object]:
    x = np.asarray(grid_x, float)
    y = np.asarray(grid_y, float)
    t = np.asarray(grid_t, float)
    Mx, My, K = x.size, y.size, t.size - 1
    if K < 1:
        raise ValueError("grid_t must contain at least two points t0 < ... < T")

    Dxx_single = _dxx_matrix_nonuniform(x)
    Dxx_block = sparse.kron(sparse.eye(K+1, format="csr"), Dxx_single, format="csr")
    P_yx = _interp_matrix_linear(x, y)

    X, Y = np.meshgrid(x, y, indexing="ij")
    psi_xy = payoff(X, Y)

    idx_phi_t0 = np.arange(0, Mx, dtype=int)
    idx_phi_T  = np.arange(K*Mx, (K+1)*Mx, dtype=int)

    meta = {"Mx": Mx, "My": My, "K": K, "shape_phi": ((K+1)*Mx,),
            "x_min": float(x[0]), "x_max": float(x[-1]),
            "y_min": float(y[0]), "y_max": float(y[-1])}

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
