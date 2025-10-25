# Implementation Case Study – SEC10

📊 **Robust Pricing and Hedging under Convex Constraints (Section 10)**  
Python implementation of the full case study, including data preprocessing, convex marginal reconstruction, dual penalized optimization, and diagnostic certificates.

---

## 🧠 Overview

This repository implements a self-contained pipeline for **robust option pricing and hedging** under convex constraints.  
The workflow reproduces the structure of *Section 10* of the reference paper:

1. Load and sanitize market option quotes (`data_io.py`)
2. Build marginal distributions via Breeden–Litzenberger densities (`marginals.py`)
3. Assemble discrete operators for the dual HJB-SPDE (`dual_spde.py`)
4. Solve the penalized dual optimization (`solver.py`)
5. Build hedge instruments and evaluate certificates (`hedge.py`, `certificate.py`)
6. Produce diagnostics, plots, and KPIs (`run_experiment.py`)

All numerical components are verified by **unit tests** under `tests/`.

---

## Environment Setup

### Requirements
- Python ≥ 3.10
- `pip`, `virtualenv`, and a modern C/C++ compiler (for `cvxpy` backends)

### Quickstart

```bash
# Clone the repo
git clone https://github.com/tommasonobili04-jpg/Implementation-CaseStudy-SEC10.git
cd Implementation-CaseStudy-SEC10

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install -U pip
pip install -r requirements.txt

# Run all unit tests (should show 8 passed)
pytest -q tests
