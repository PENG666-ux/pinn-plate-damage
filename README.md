# PINN-Based Structural Damage Identification for Kirchhoff-Love Plates

A **Physics-Informed Neural Network (PINN)** framework for identifying
localised structural damage in thin plates from vibration response data.
The network simultaneously satisfies the Kirchhoff-Love plate equation and
fits sparse multi-snapshot displacement measurements, recovering damage
position, size, and intensity without any labelled damage examples.

---

## Method Overview

The effective areal-density field of a damaged plate is modelled as:

```
σ*(x,y) = ρh + Σ αₖ Δσ · sigmoid[ β(rₖ² − dₖ²(x,y)) ]
```

where each damage site k is described by four learnable parameters:
centre (xₖ, yₖ), radius rₖ, and intensity αₖ.

Training proceeds in **three stages**:

```
Healthy data  →  Stage 1  →  Network weights (healthy baseline)
                                      ↓ loaded into
Damaged data  →  Stage 2  →  Network + damage params (joint optimisation)
                                      ↓ loaded into
Damaged data  →  Stage 3  →  Refined damage params (adaptive sampling)
```

See [`docs/method.md`](docs/method.md) for the full mathematical formulation.

---

## Repository Structure

```
├── src/
│   ├── config.py      # All hyper-parameters and physical constants
│   ├── network.py     # Fully-connected DNN (tanh, Xavier init)
│   ├── sampling.py    # Collocation-point generation + CSV data loading
│   ├── pinn.py        # PINN class (forward pass, PDE residual, training loops)
│   ├── plot.py        # Visualisation functions
│   └── main.py        # Training orchestration (Stages 1 → 2 → 3)
│
├── docs/
│   └── method.md      # Mathematical background
│
├── data/
│   └── example/
│       └── README.md  # Expected data format and column specification
│
├── results/           # Commit representative output figures here
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/pinn-plate-damage.git
cd pinn-plate-damage

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

PyTorch with CUDA is strongly recommended. CPU training is supported but slow.

---

## Data Preparation

The code expects two folders of COMSOL-exported CSV files:

- `data/healthy/` — snapshots from the undamaged plate simulation  
- `data/damaged/` — snapshots from the damaged plate simulation  

Each CSV must contain the columns `t`, `x`, `y`, `u`
(normalised time, coordinates, and out-of-plane deflection).

See [`data/example/README.md`](data/example/README.md) for the full
specification.

If your files live elsewhere, update these two lines in `src/config.py`:

```python
DATA_HEALTHY = "/path/to/healthy/folder"
DATA_DAMAGE  = "/path/to/damaged/folder"
```

---

## Usage

```bash
cd src
python main.py
```

Training produces:

| Output file | Contents |
|-------------|----------|
| `stage1_model.pth` | Stage-1 network checkpoint |
| `stage2_model.pth` | Stage-2 joint-optimisation checkpoint |
| `stage3_final_model.pth` | Final model |
| `damage_params_stage2.npz` | Damage parameters after Stage 2 |
| `damage_map_final.png` | Inferred σ*(x,y) heatmap |
| `main_damage_parameter_evolution.png` | Parameter evolution curves |
| `total_loss_curve.png` | Three-stage loss history |
| `damage_evolution/evolution.gif` | Animated damage map (if imageio is installed) |

---

## Configuration

All settings live in `src/config.py`. Key parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `K_MAX` | 3 | Number of candidate damage sites |
| `BETA` | 60.0 | Sigmoid sharpness (fixed) |
| `DELTA_SIGMA` | 300.0 kg/m² | Areal-density change of damage |
| `N_PDE` | 20 000 | Total collocation points |
| `GRID_SIZE` | 15 | Spatial measurement grid (15×15) |
| `S1_ADAM_EPOCHS` | 9 000 | Stage-1 Adam iterations |
| `S1_LBFGS_EPOCHS` | 1 000 | Stage-1 L-BFGS iterations |
| `S2_EPOCHS` | 1 000 | Stage-2 Adam iterations |
| `S3_EPOCHS` | 1 000 | Stage-3 Adam iterations |

---

## Physical Parameters (default: aluminium plate)

| Parameter | Value |
|-----------|-------|
| Young's modulus E | 69.0 GPa |
| Poisson's ratio ν | 0.33 |
| Density ρ | 2700 kg/m³ |
| Thickness h | 1 mm |
| Plate size | 300 × 300 mm |

---

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 2.0 (CUDA 11.8+ recommended)  
- NumPy, SciPy, Matplotlib, Pandas  
- imageio (optional, for GIF output)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pinn_plate_damage,
  author  = {Your Name},
  title   = {PINN-Based Structural Damage Identification for Kirchhoff-Love Plates},
  year    = {2025},
  url     = {https://github.com/<your-username>/pinn-plate-damage}
}
```

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.
