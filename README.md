<div align="center">

<sub>PHYSICS-INFORMED LEARNING · INVERSE PROBLEMS · STRUCTURAL DYNAMICS</sub>

# PINN Plate Damage Identification

### Localized mass-perturbation recovery in Kirchhoff–Love plates

An auditable, three-stage physics-informed neural network for estimating the
location, radius, and intensity of localized areal-density perturbations from
sparse transverse-vibration measurements.

<p>
  <a href="https://github.com/PENG666-ux/pinn-plate-damage/actions/workflows/tests.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/PENG666-ux/pinn-plate-damage/tests.yml?branch=main&style=flat-square&label=tests"></a>
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-2F855A?style=flat-square"></a>
  <a href="CITATION.cff"><img alt="Cite this repository" src="https://img.shields.io/badge/Citation-CFF-6B46C1?style=flat-square"></a>
</p>

[Method](#method) · [Workflow](#three-stage-workflow) · [Quick start](#quick-start) · [Validation](#validation-by-design) · [Limitations](#scientific-scope)

</div>

---

## At a glance

| Physics | Inference | Reproducibility |
|:--|:--|:--|
| Rectangular Kirchhoff–Love operator with explicit aspect-ratio scaling | Smooth, bounded estimates of centre, radius, and amplitude | Seeded sampling and sensor-grouped train/validation/test partitions |
| Simply-supported displacement and bending-moment constraints | Adaptive collocation around active perturbation sites | Versioned checkpoints with configuration hashes and RNG states |
| Manufactured-solution verification of the PDE and edge operators | Three-stage Adam / L-BFGS optimization | Automated tests, smoke mode, and machine-readable metrics |

> [!IMPORTANT]
> This implementation identifies **localized areal-density perturbations**.
> It changes inertia, not bending stiffness, and must not be interpreted as a
> general crack, delamination, or stiffness-loss model without extending and
> independently validating the governing physics.

## Method

For normalized coordinates $x=X/L_x$, $y=Y/L_y$, and $t=\tau/T$, the network
$\hat{w}_\theta(t,x,y)$ is constrained by the rectangular-plate residual

$$
\mathcal{R}_\theta =
\hat{w}_{,xxxx}
+2a^2\hat{w}_{,xxyy}
+a^4\hat{w}_{,yyyy}
+\frac{\sigma(x,y)L_x^4}{DT^2}\hat{w}_{,tt},
\qquad
a=\frac{L_x}{L_y},
\quad
D=\frac{Eh^3}{12(1-\nu^2)}.
$$

The inverse field is represented by a sparse collection of differentiable
signed-distance gates:

$$
\sigma(x,y)=\rho h+
\sum_{k=1}^{K}\alpha_k\,\Delta\sigma\,
\operatorname{sigmoid}\!\left(\beta\,[r_k-d_k(x,y)]\right),
\qquad
d_k=\sqrt{(x-x_k)^2+(y-y_k)^2}.
$$

The mappings enforce $\alpha_k\in(0,1)$, bounded centres, and physically
configured radius limits throughout optimization. See the
[full derivation and assumptions](docs/method.md).

## Three-stage workflow

| Stage | Objective | Optimization | Collocation strategy |
|:--:|:--|:--|:--|
| **01 · Baseline** | Learn healthy-plate dynamics | Adam → deterministic L-BFGS | Global Latin hypercube + boundary points |
| **02 · Inversion** | Jointly update field and perturbation parameters | Two-rate Adam | Global coverage + emerging-site refinement |
| **03 · Refinement** | Resolve the dominant perturbation field | Damage-focused Adam | High-density sampling near active estimates |

Every stage preserves the same governing equation and explicit
simply-supported edge constraints. The test partition remains untouched until
the final evaluation.

## Quick start

### 1. Install

```bash
git clone https://github.com/PENG666-ux/pinn-plate-damage.git
cd pinn-plate-damage
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -e ".[test]"
pytest
```

### 2. Prepare measurements

Place healthy and perturbed measurements in separate folders. Each CSV must
contain finite numeric columns:

| Column | Quantity | Accepted representation |
|:--|:--|:--|
| `t` | Time | normalized or seconds |
| `x`, `y` | Plate coordinates | normalized or metres |
| `u` | Transverse displacement | one consistent physical unit |

Choose the coordinate convention explicitly with `--coordinate-mode`; the
loader rejects inconsistent ranges instead of silently mixing units.

### 3. Verify, then train

```bash
# Fast end-to-end pipeline check
python -m src.main \
  --smoke-test \
  --healthy-data data/healthy \
  --damaged-data data/damaged \
  --coordinate-mode physical

# Recorded experiment
python -m src.main \
  --healthy-data data/healthy \
  --damaged-data data/damaged \
  --coordinate-mode physical \
  --output-dir results/run-1234 \
  --seed 1234
```

The run directory contains the split manifest, validation/test metrics,
stage-wise checkpoints, inferred parameters, and publication-ready figures.

## Validation by design

The repository distinguishes software verification from scientific validation.

| Layer | Evidence produced |
|:--|:--|
| **Operator verification** | A closed-form healthy mode satisfies the PDE and simply-supported boundary operators |
| **Implementation verification** | Automated tests cover sampling counts, bounds, leakage prevention, and checkpoint recovery |
| **Predictive evaluation** | RMSE, MAE, and relative $L_2$ error are reported on held-out sensor locations |
| **Experiment traceability** | Seeds, data-split counts, runtime versions, normalization, and configuration hashes are recorded |

For publishable claims, add multi-seed recovery experiments, a healthy-plate
negative control, noise sweeps, sensor-density ablations, known-parameter
synthetic cases, and a conventional inverse-method baseline. The complete
checklist is in the [validation protocol](docs/validation.md).

## Outputs

```text
results/run-1234/
├── data_split.json          # immutable train / validation / test counts
├── metrics.json             # held-out metrics and inferred parameters
├── stage1.pt                # healthy baseline checkpoint
├── stage2.pt                # joint inversion checkpoint
├── stage3.pt                # refined checkpoint
├── damage_parameters.npz    # machine-readable parameter estimates
├── damage_map.png           # inferred areal-density field
└── training_history.png     # objective components across training
```

<details>
<summary><strong>Repository layout</strong></summary>

```text
.
├── src/
│   ├── config.py            # physical constants and experiment controls
│   ├── network.py           # fully connected field network
│   ├── sampling.py          # deterministic and adaptive sampling
│   ├── pinn.py              # physics, losses, training, and checkpoints
│   ├── plot.py              # auditable result visualization
│   └── main.py              # command-line experiment pipeline
├── tests/                   # manufactured-solution and unit tests
├── docs/
│   ├── method.md            # derivation and assumptions
│   ├── quickstart.md        # detailed usage
│   └── validation.md        # minimum academic validation protocol
├── CITATION.cff
└── pyproject.toml
```

</details>

## Scientific scope

- The forward model assumes a thin, isotropic, undamped Kirchhoff–Love plate.
- The implemented edges are simply supported; clamped or free plates require
  different residuals.
- The perturbation is additive mass per area. Cracks, voids, and delamination
  generally require stiffness or constitutive-field modeling.
- A low PINN loss does not prove parameter identifiability. Multiple fields may
  explain sparse measurements similarly.
- Hardware and library versions can affect floating-point trajectories even
  when deterministic algorithms are requested.

## Citation

If this repository supports your research, cite the version used through
GitHub's **Cite this repository** panel or [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built for transparent inverse modeling, reproducible experiments, and defensible scientific claims.</sub>
</div>
