from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Callable, Dict, Any, List

# NOTE:
# - perturb_smiles: multiplicative bump on call mids as a proxy for ±vol-point tests.
# - haircut_budget: reduce V by a fraction h.
# - cost_scale: multiply all costs by (1+chi).
# - run_stress: generic scenario runner; expects a 'pipeline_fn(quotes_t0, quotes_T, V, cfg) -> {"Pi":..., "lambda":...}'.

def perturb_smiles(quotes: pd.DataFrame, dv: float) -> pd.DataFrame:
    """
    Simple multiplicative bump: mid' = mid * (1 + 0.02 * dv), where dv is in 'vol points' proxy.
    """
    g = quotes.copy()
    g["mid"] = g["mid"].astype(float) * (1.0 + 0.02 * float(dv))
    g["bid"] = g["mid"] * 0.999
    g["ask"] = g["mid"] * 1.001
    return g

def haircut_budget(V: float, h: float) -> float:
    return float(V) * (1.0 - float(h))

def cost_scale(costs: Dict[str, float], chi: float) -> Dict[str, float]:
    return {k: float(v) * (1.0 + float(chi)) for k, v in costs.items()}

def run_stress(quotes: Tuple[pd.DataFrame, pd.DataFrame],
               V: float,
               cfg: Dict[str, Any],
               pipeline_fn: Callable[[pd.DataFrame, pd.DataFrame, float, Dict[str, Any]], Dict[str, float]],
               vol_bumps: List[float] | None = None,
               haircuts: List[float] | None = None,
               cost_bumps: List[float] | None = None) -> pd.DataFrame:
    """
    Execute a grid of stress scenarios and collect (Pi, lambda).

    The pipeline_fn should re-solve given (quotes_t0', quotes_T', V', cfg') and
    return {"Pi": ..., "lambda": ...}.

    - vol_bumps: list of 'dv' bumps (in vol points proxy) applied to both smiles.
    - haircuts: list of budget haircuts h -> V' = (1-h)*V.
    - cost_bumps: if provided, multiply cfg["costs"] by (1+chi) for each chi.
      (Backwards compatible: if None, behaves as a single run with chi=0.0.)
    """
    vol_bumps = [-2.0, -1.0, 0.0, +1.0, +2.0] if vol_bumps is None else list(vol_bumps)
    haircuts  = [0.0, 0.05, 0.10]              if haircuts  is None else list(haircuts)
    cb_list   = [0.0] if cost_bumps is None else list(cost_bumps)

    q0_base, qT_base = quotes
    rows: list[dict] = []

    for dv in vol_bumps:
        q0 = perturb_smiles(q0_base, dv)
        qT = perturb_smiles(qT_base, dv)
        for h in haircuts:
            Vh = haircut_budget(V, h)
            for chi in cb_list:
                # Clone cfg and scale costs, if present
                cfg2 = dict(cfg)
                if "costs" in cfg:
                    cfg2["costs"] = cost_scale(cfg["costs"], chi)
                else:
                    cfg2["costs"] = {}

                res = pipeline_fn(q0, qT, Vh, cfg2)  # must return {"Pi":..., "lambda":...}
                rows.append({
                    "dv": dv,
                    "haircut": h,
                    "cost_bump": chi,
                    "Pi": float(res.get("Pi", np.nan)),
                    "lambda": float(res.get("lambda", np.nan)),
                })

    return pd.DataFrame(rows, columns=["dv", "haircut", "cost_bump", "Pi", "lambda"])
