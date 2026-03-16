# Quick-Start Guide

## 1. Minimum working example

After installing dependencies and placing your CSV data in `data/healthy/`
and `data/damaged/`, run:

```bash
cd src
python main.py
```

Training on a single RTX 4060 (8 GB) takes approximately:

| Stage | Optimiser | Epochs | Typical time |
|-------|-----------|--------|-------------|
| 1 | Adam | 9 000 | ~40 min |
| 1 | L-BFGS | 1 000 | ~15 min |
| 2 | Adam | 1 000 | ~10 min |
| 3 | Adam | 1 000 | ~10 min |

## 2. Resuming from a checkpoint

To skip Stage 1 and start from a saved checkpoint:

```python
# In main.py, replace the Stage-1 block with:
pinn1 = build_pinn(cfg, t_h, x_h, y_h, u_h, t_pde1, x_pde1, y_pde1, device)
pinn1.load_model("stage1_model.pth")
```

## 3. Changing the number of damage sites

Edit `K_MAX` in `src/config.py` and update `INIT_POSITIONS`, `INIT_RADII_MM`,
and `INIT_ALPHA` to match the new count.

## 4. Using a different plate material

Update the physical constants block in `src/config.py`:

```python
E          = 200e9    # steel: 200 GPa
NU         = 0.30
RHO0       = 7850.0   # kg/m3
H_PLATE    = 0.002    # 2 mm
```

## 5. Output files

All plots and model checkpoints are written to the working directory
(wherever `main.py` is run from). Redirect them by editing the `save_path`
arguments in the `main()` function.
