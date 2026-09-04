# PINN identification of localized mass perturbations in thin plates

[![tests](https://github.com/PENG666-ux/pinn-plate-damage/actions/workflows/tests.yml/badge.svg)](https://github.com/PENG666-ux/pinn-plate-damage/actions/workflows/tests.yml)

This repository implements a physics-informed neural network (PINN) for an
inverse Kirchhoff-Love plate problem. It estimates the centres, radii, and
amplitudes of smooth, localized **areal-density perturbations** from transverse
vibration measurements.

> Scientific scope: the current model changes inertia, not bending stiffness.
> It is therefore a mass-perturbation surrogate and must not be interpreted as
> a general crack, delamination, or material-stiffness damage model without an
> independently justified constitutive extension.

## What is verified

- The dimensionless biharmonic operator retains the correct aspect-ratio terms
  for rectangular plates.
- Simply-supported displacement and bending-moment conditions are explicit
  loss terms.
- A closed-form healthy-plate mode is used as a manufactured-solution test.
- Complete sensor locations, rather than individual time rows, are assigned to
  train/validation/test partitions to prevent temporal leakage.
- Sampling, data partitions, and L-BFGS objectives are deterministic under a
  recorded seed.
- Checkpoints record the code schema, configuration hash, normalization,
  optimizer state, histories, runtime versions, and random-number states.

These checks establish implementation consistency; they do **not** establish
identifiability or experimental validity for a particular dataset.

## Installation and tests

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## Data contract

Place healthy and perturbed measurements in separate folders. Every CSV must
contain numeric, finite columns `t,x,y,u`. All files in a folder are combined
in sorted filename order. Choose the coordinate convention explicitly:

- `--coordinate-mode normalized`: `t,x,y` are already in `[0,1]`.
- `--coordinate-mode physical`: time is in seconds and coordinates are in
  metres; values are divided by `T_PHYSICAL`, `X_PHYSICAL`, and `Y_PHYSICAL`.

## Run

```bash
python -m src.main \
  --healthy-data data/healthy \
  --damaged-data data/damaged \
  --coordinate-mode physical \
  --output-dir results/run-1234 \
  --seed 1234
```

Use `--smoke-test` to exercise all three stages with one Adam epoch each. A
full run writes immutable split counts, held-out metrics, checkpoints, inferred
parameters, and figures to the selected output directory.

For paper-quality evidence, repeat the complete pipeline with multiple
pre-registered seeds and report all runs (median and dispersion), a no-damage
negative control, noise sweeps, sensor-density ablations, and synthetic cases
with known parameters. See [method details](docs/method.md) and the
[validation protocol](docs/validation.md).

## Reproducibility cautions

GPU kernels can still differ across hardware or PyTorch/CUDA releases. The
checkpoint records the runtime version, and deterministic algorithms are
requested with warnings enabled. Archive the raw data, exact commit, config,
and environment alongside published results. Do not select the best random
seed or tune hyperparameters against the test partition.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The code is MIT licensed.
