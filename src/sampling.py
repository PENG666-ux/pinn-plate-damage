# sampling.py

import numpy as np
import pandas as pd
import torch

try:
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Collocation point generators
# ---------------------------------------------------------------------------

def lhs_points(n, t_range=(0., 1.), x_range=(0., 1.), y_range=(0., 1.)):
    """Latin hypercube samples over (t, x, y)."""
    if SCIPY_AVAILABLE:
        sampler = qmc.LatinHypercube(d=3)
        s = sampler.random(n=n)
        s = qmc.scale(s, [t_range[0], x_range[0], y_range[0]],
                         [t_range[1], x_range[1], y_range[1]])
    else:
        s = np.random.uniform(
            [t_range[0], x_range[0], y_range[0]],
            [t_range[1], x_range[1], y_range[1]],
            size=(n, 3)
        )
    return s[:, 0:1], s[:, 1:2], s[:, 2:3]


def boundary_points(n):
    """Uniform samples on the four spatial boundaries, t uniform in [0,1]."""
    n_per = n // 4
    segs = []
    for _ in range(n_per):       # x=0
        segs.append([np.random.uniform(), 0., np.random.uniform()])
    for _ in range(n_per):       # x=1
        segs.append([np.random.uniform(), 1., np.random.uniform()])
    for _ in range(n_per):       # y=0
        segs.append([np.random.uniform(), np.random.uniform(), 0.])
    for _ in range(n - 3 * n_per):  # y=1
        segs.append([np.random.uniform(), np.random.uniform(), 1.])
    arr = np.array(segs, dtype=np.float64)
    return arr[:, 0:1], arr[:, 1:2], arr[:, 2:3]


def damage_focused_points(n, centres, radii, weights, sigma_factor=2.0):
    """Gaussian samples centred on active damage sites, weighted by alpha."""
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    t_list, x_list, y_list = [], [], []
    for (cx, cy), r, w in zip(centres, radii, weights):
        k = max(int(n * w), 10)
        sigma = max(r * sigma_factor, 0.05)
        xs = np.clip(np.random.normal(cx, sigma, k), 0., 1.)
        ys = np.clip(np.random.normal(cy, sigma, k), 0., 1.)
        ts = np.random.uniform(0., 1., k)
        t_list.append(ts.reshape(-1, 1))
        x_list.append(xs.reshape(-1, 1))
        y_list.append(ys.reshape(-1, 1))
    t = np.vstack(t_list)
    x = np.vstack(x_list)
    y = np.vstack(y_list)
    if len(t) > n:
        idx = np.random.choice(len(t), n, replace=False)
        t, x, y = t[idx], x[idx], y[idx]
    return t, x, y


def damage_focused_points_hires(n, centres, radii, weights):
    """High-resolution sampling: 70% inside the damage disc, 30% near edge."""
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    t_list, x_list, y_list = [], [], []
    for (cx, cy), r, w in zip(centres, radii, weights):
        k = max(int(n * w), 20)
        n_in = int(k * 0.7)
        n_out = k - n_in
        # inner disc
        r_samp = r * np.sqrt(np.random.uniform(0, 1, n_in))
        theta = np.random.uniform(0, 2 * np.pi, n_in)
        xs_in = np.clip(cx + r_samp * np.cos(theta), 0., 1.)
        ys_in = np.clip(cy + r_samp * np.sin(theta), 0., 1.)
        # outer annulus
        sigma = max(r * 1.5, 0.03)
        xs_out = np.clip(np.random.normal(cx, sigma, n_out), 0., 1.)
        ys_out = np.clip(np.random.normal(cy, sigma, n_out), 0., 1.)
        xs = np.concatenate([xs_in, xs_out])
        ys = np.concatenate([ys_in, ys_out])
        ts = np.random.uniform(0., 1., k)
        t_list.append(ts.reshape(-1, 1))
        x_list.append(xs.reshape(-1, 1))
        y_list.append(ys.reshape(-1, 1))
    t = np.vstack(t_list)
    x = np.vstack(x_list)
    y = np.vstack(y_list)
    if len(t) > n:
        idx = np.random.choice(len(t), n, replace=False)
        t, x, y = t[idx], x[idx], y[idx]
    return t, x, y


def generate_pde_points(N, stage=1, epoch=0, alpha_vals=None,
                        x_vals=None, y_vals=None, r_vals=None,
                        stage2_switch=2000):
    """
    Generate collocation points according to current stage and epoch.

    Stage 1 / Stage 2 early  : 80 % LHS + 20 % boundary
    Stage 2 late             : 60 % damage-focused + 30 % uniform + 10 % boundary
    Stage 3                  : 80 % damage-focused (hi-res) + 15 % uniform + 5 % boundary
    """
    if stage == 1 or (stage == 2 and epoch < stage2_switch):
        n_lhs = int(N * 0.8)
        n_bnd = N - n_lhs
        t1, x1, y1 = lhs_points(n_lhs)
        t2, x2, y2 = boundary_points(n_bnd)
        t = np.vstack([t1, t2])
        x = np.vstack([x1, x2])
        y = np.vstack([y1, y2])
        return t.astype(np.float64), x.astype(np.float64), y.astype(np.float64)

    active = np.where(alpha_vals > 1e-3)[0] if alpha_vals is not None else np.array([])

    if stage == 2:
        n_dmg = int(N * 0.6)
        n_glo = int(N * 0.3)
        n_bnd = N - n_dmg - n_glo
    else:  # stage 3
        n_dmg = int(N * 0.8)
        n_glo = int(N * 0.15)
        n_bnd = N - n_dmg - n_glo

    t_glo = np.random.uniform(0., 1., (n_glo, 1)).astype(np.float64)
    x_glo = np.random.uniform(0., 1., (n_glo, 1)).astype(np.float64)
    y_glo = np.random.uniform(0., 1., (n_glo, 1)).astype(np.float64)
    t_bnd, x_bnd, y_bnd = boundary_points(n_bnd)

    if len(active) == 0:
        t_dmg, x_dmg, y_dmg = lhs_points(n_dmg)
    else:
        centres = [(x_vals[i], y_vals[i]) for i in active]
        radii   = [r_vals[i] for i in active]
        weights = alpha_vals[active]
        if stage == 3:
            t_dmg, x_dmg, y_dmg = damage_focused_points_hires(n_dmg, centres, radii, weights)
        else:
            t_dmg, x_dmg, y_dmg = damage_focused_points(n_dmg, centres, radii, weights)

    t = np.vstack([t_glo, t_bnd, t_dmg])
    x = np.vstack([x_glo, x_bnd, x_dmg])
    y = np.vstack([y_glo, y_bnd, y_dmg])
    return t.astype(np.float64), x.astype(np.float64), y.astype(np.float64)


def to_tensor(arr, device, dtype=torch.float64):
    return torch.tensor(arr, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Data loading and uniform-grid sub-sampling
# ---------------------------------------------------------------------------

def load_csv_folder(folder):
    import os
    frames = []
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            try:
                frames.append(pd.read_csv(os.path.join(folder, fname)))
            except Exception as e:
                print(f"Warning: skipped {fname}: {e}")
    if not frames:
        raise RuntimeError(f"No CSV files found in {folder}")
    return pd.concat(frames, ignore_index=True)


def uniform_grid_sample(df, grid_size=15, num_time_points=80, tol=0.01):
    """
    Sub-sample df onto a regular grid_size × grid_size spatial grid,
    keeping num_time_points per location.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    x_pts = np.linspace(df['x'].min(), df['x'].max(), grid_size)
    y_pts = np.linspace(df['y'].min(), df['y'].max(), grid_size)

    pieces = []
    found = 0
    for xv in x_pts:
        for yv in y_pts:
            mask = (np.abs(df['x'] - xv) < tol) & (np.abs(df['y'] - yv) < tol)
            sub = df[mask]
            if sub.empty:
                continue
            # use the first matched coordinate to avoid floating-point spread
            mx, my = sub.iloc[0]['x'], sub.iloc[0]['y']
            sub = df[(np.abs(df['x'] - mx) < tol) & (np.abs(df['y'] - my) < tol)]
            if len(sub) >= num_time_points:
                idx = np.linspace(0, len(sub) - 1, num_time_points, dtype=int)
                sub = sub.iloc[idx]
            pieces.append(sub)
            found += 1

    if found == 0:
        raise RuntimeError("No spatial grid points matched within tolerance.")

    result = pd.concat(pieces, ignore_index=True)
    n_spatial = result[['x', 'y']].drop_duplicates().__len__()
    print(f"Grid sampling: {found}/{grid_size**2} locations matched, "
          f"{n_spatial} unique, {len(result)} total points.")
    return result
