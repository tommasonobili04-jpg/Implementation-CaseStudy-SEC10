from __future__ import annotations

import os
import json
import time
import math
import random
import logging
from contextlib import contextmanager
from typing import Callable, Iterable, Tuple

import numpy as np

# NOTE: Core utilities used across the pipeline (seed, timer, interpolation, finite differences).
# Keep dependencies minimal and deterministic.

def set_seed(seed: int) -> None:
    """
    Set global RNG seed for reproducibility across Python, NumPy, and hash.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(int(seed))


@contextmanager
def timer(name: str = "", logger: logging.Logger | None = None, level: int = logging.INFO):
    """
    Context manager to measure wall time of a code block.
    Usage:
        with timer("solve", logger):
            ...
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        msg = f"{name} took {dt:.3f}s" if name else f"elapsed {dt:.3f}s"
        if logger is not None:
            logger.log(level, msg)
        else:
            print(msg)


def finite_diff_second(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Monotone second derivative on a non-uniform grid (second order).
    Interior:
      2/(dx_{i-1}+dx_i) * ( (f_{i+1}-f_i)/dx_i - (f_i-f_{i-1})/dx_{i-1} )
    Boundaries: one-sided non-uniform 3-point stencils (second order).
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("finite_diff_second requires n >= 3")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")

    fxx = np.empty_like(f)
    dx = np.diff(x)

    # interior
    for i in range(1, n - 1):
        dxm = dx[i - 1]
        dxp = dx[i]
        term = ((f[i + 1] - f[i]) / dxp) - ((f[i] - f[i - 1]) / dxm)
        fxx[i] = 2.0 * term / (dxm + dxp)

    # left boundary: one-sided, non-uniform (3 punti)
    h0, h1 = x[1] - x[0], x[2] - x[1]
    fxx[0] = 2.0 * (f[0] / (h0 * (h0 + h1)) - f[1] / (h0 * h1) + f[2] / (h1 * (h0 + h1)))

    # right boundary: one-sided, non-uniform (3 punti)
    hm1, hm2 = x[-1] - x[-2], x[-2] - x[-3]
    fxx[-1] = 2.0 * (f[-3] / (hm2 * (hm2 + hm1)) - f[-2] / (hm2 * hm1) + f[-1] / (hm1 * (hm2 + hm1)))

    return fxx


def interp1_strict(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Strictly bounded linear interpolation: raises if query is outside [x_min, x_max].
    Useful to avoid silent extrapolation in PDE/optimization loops.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1D arrays of equal length")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")
    x_min, x_max = x[0], x[-1]

    def _f(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if (q < x_min).any() or (q > x_max).any():
            raise ValueError("query outside interpolation domain")
        return np.interp(q, x, y)

    return _f


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    """
    Save a JSON file with UTF-8 encoding and sorted keys for reproducibility.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str) -> dict:
    """
    Load a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
