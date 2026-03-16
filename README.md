# PINN-Based Structural Damage Identification for Kirchhoff-Love Plates

A Physics-Informed Neural Network (PINN) framework for identifying localised
structural damage in thin plates from vibration response data.

---
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?logo=nvidia)

## Method Overview

The effective areal-density field of a damaged plate is modelled as:

$$\sigma^*(x,y) = \rho h + \sum_{k=1}^{K} \alpha_k\,\Delta\sigma\;\text{sigmoid}\left[\beta\left(r_k^2 - d_k^2(x,y)\right)\right]$$

Each damage site $k$ is described by four learnable parameters: centre $(x_k, y_k)$, radius $r_k$, and intensity $\alpha_k$. Training proceeds in three stages:

**Stage 1** — Learn healthy-plate dynamics from undamaged measurements (Adam + L-BFGS).  
**Stage 2** — Joint optimisation of network weights and damage parameters on damaged measurements.  
**Stage 3** — Damage parameter refinement with damage-focused adaptive collocation sampling.


---

## Repository Structure

```
├── src/
│   ├── config.py      # All hyper-parameters and physical constants
│   ├── network.py     # Fully-connected DNN
│   ├── sampling.py    # Collocation-point generation and data loading
│   ├── pinn.py        # PINN class: forward pass, PDE residual, training loops
│   ├── plot.py        # Visualisation functions
│   └── main.py        # Training orchestration (Stages 1 → 2 → 3)
├── docs/
│   ├── method.md      # Mathematical background
│   └── quickstart.md  # Usage and configuration guide
├── data/
│   └── example/
│       └── README.md  # Expected CSV format
├── results/           # Commit representative output figures here
├── requirements.txt
└── LICENSE
```

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0 (CUDA recommended)
- NumPy, SciPy, Matplotlib, Pandas
- imageio (optional, for GIF output)

```bash
pip install -r requirements.txt
```
