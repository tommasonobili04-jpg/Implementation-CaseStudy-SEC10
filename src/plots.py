from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Sequence

# NOTE: Minimal plotting utilities (no style assumptions).


def plot_Pi_vs_V(V_list: Sequence[float],
                 Pi_list: Sequence[float],
                 lambdas: Sequence[float] | None = None):
    fig = plt.figure()
    V = np.asarray(V_list, float)
    Pi = np.asarray(Pi_list, float)
    plt.plot(V, Pi, marker="o")
    plt.xlabel("Variance budget V")
    plt.ylabel("Robust value Π(Φ;V)")
    plt.title("Robust value vs budget")
    if lambdas is not None:
        # Show slope at last point (≈ λ*)
        lam = float(lambdas[-1])
        plt.annotate(f"lambda* ≈ {lam:.4f}", xy=(V[-1], Pi[-1]))
    plt.grid(True, alpha=0.3)
    return fig


def plot_phi_slices(grid_x, phi_slices: list[tuple[str, np.ndarray]]):
    """
    phi_slices: list of (label, phi_values_on_x) to compare multiple time slices.
    """
    fig = plt.figure()
    x = np.asarray(grid_x, float)
    for label, phi_vals in phi_slices:
        plt.plot(x, np.asarray(phi_vals, float), label=label)
    plt.xlabel("x")
    plt.ylabel("phi(t, x)")
    plt.title("Phi profiles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return fig


def plot_market_vs_robust(P_mkt: float, Pi: float):
    fig = plt.figure()
    plt.bar(["Market", "Robust Π"], [float(P_mkt), float(Pi)])
    plt.ylabel("Price")
    plt.title("Market vs Robust Price")
    return fig
