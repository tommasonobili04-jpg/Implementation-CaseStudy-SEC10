from __future__ import annotations

import numpy as np
import json
import logging
import os
from typing import Dict, Any

from .utils import ensure_dir

# NOTE: Lightweight logger + KPI dump for reproducible runs.

def init_logger(name: str = "impl", level: int = logging.INFO) -> logging.Logger:
    """
    Create a console logger with a concise timestamped format.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = "[%(asctime)s] %(levelname)s - %(message)s"
        datefmt = "%H:%M:%S"
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(sh)
        logger.propagate = False
    return logger


def kpi_dump(path: str, metrics: Dict[str, Any]) -> None:
    """
    Persist run metrics to JSON (indent=2, sorted keys).
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

def rng(seed: int | None) -> np.random.Generator:
    """
    Deterministic NumPy Generator aligned with set_seed; used by simulations/tests.
    """
    return np.random.default_rng(None if seed is None else int(seed))
