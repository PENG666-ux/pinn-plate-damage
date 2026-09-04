"""Deterministic sampling and leakage-resistant dataset utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import qmc
except ImportError:  # pragma: no cover - scipy is a declared dependency
    qmc = None


REQUIRED_COLUMNS = ("t", "x", "y", "u")


def _rng(seed=None) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def lhs_points(n: int, t_range=(0.0, 1.0), x_range=(0.0, 1.0),
               y_range=(0.0, 1.0), seed=None):
    """Latin-hypercube samples over ``(t, x, y)`` with an explicit seed."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        empty = np.empty((0, 1), dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()
    bounds = (t_range, x_range, y_range)
    if any(lo >= hi for lo, hi in bounds):
        raise ValueError("every sampling range must satisfy lower < upper")
    rng = _rng(seed)
    if qmc is not None:
        sample = qmc.LatinHypercube(d=3, seed=rng).random(n=n)
        sample = qmc.scale(sample, [r[0] for r in bounds], [r[1] for r in bounds])
    else:
        sample = rng.uniform([r[0] for r in bounds], [r[1] for r in bounds], (n, 3))
    return tuple(sample[:, i:i + 1].astype(np.float64) for i in range(3))


def boundary_points(n: int, seed=None):
    """Sample exactly ``n`` points, balanced over the four spatial edges."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = _rng(seed)
    counts = np.full(4, n // 4, dtype=int)
    counts[: n % 4] += 1
    blocks = []
    for edge, count in enumerate(counts):
        block = rng.uniform(0.0, 1.0, (count, 3))
        if edge < 2:
            block[:, 1] = float(edge)
        else:
            block[:, 2] = float(edge - 2)
        blocks.append(block)
    sample = np.vstack(blocks) if blocks else np.empty((0, 3))
    rng.shuffle(sample)
    return tuple(sample[:, i:i + 1].astype(np.float64) for i in range(3))


def _focused_points(n, centres, radii, weights, rng, high_resolution=False):
    if n == 0:
        return lhs_points(0)
    centres = np.asarray(centres, dtype=float)
    radii = np.asarray(radii, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if centres.ndim != 2 or centres.shape[1] != 2 or len(centres) == 0:
        raise ValueError("centres must have shape (K, 2)")
    if len(radii) != len(centres) or len(weights) != len(centres):
        raise ValueError("centres, radii, and weights must have equal lengths")
    probabilities = np.clip(weights, 0.0, None)
    probabilities = (probabilities / probabilities.sum() if probabilities.sum()
                     else np.full(len(weights), 1 / len(weights)))
    sites = rng.choice(len(centres), size=n, p=probabilities)
    x = np.empty(n); y = np.empty(n)
    for site in range(len(centres)):
        idx = np.flatnonzero(sites == site)
        if not len(idx):
            continue
        cx, cy = centres[site]
        radius = max(float(radii[site]), np.finfo(float).eps)
        if high_resolution:
            inside = rng.random(len(idx)) < 0.7
            radial = radius * np.sqrt(rng.random(len(idx)))
            radial[~inside] = np.abs(rng.normal(radius, max(0.25 * radius, 0.005), (~inside).sum()))
            angle = rng.uniform(0.0, 2.0 * np.pi, len(idx))
            x[idx] = cx + radial * np.cos(angle)
            y[idx] = cy + radial * np.sin(angle)
        else:
            sigma = max(2.0 * radius, 0.025)
            x[idx] = rng.normal(cx, sigma, len(idx))
            y[idx] = rng.normal(cy, sigma, len(idx))
    sample = np.column_stack([rng.uniform(0.0, 1.0, n),
                              np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)])
    return tuple(sample[:, i:i + 1].astype(np.float64) for i in range(3))


def generate_pde_points(n: int, stage=1, epoch=0, alpha_vals=None,
                        x_vals=None, y_vals=None, r_vals=None,
                        stage2_switch=2000, seed=None):
    """Generate exactly ``n`` collocation points for the requested stage."""
    rng = _rng(seed)
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2, or 3")
    if stage == 1 or (stage == 2 and epoch < stage2_switch):
        n_global = int(round(0.8 * n)); n_boundary = n - n_global; n_damage = 0
    elif stage == 2:
        n_damage = int(round(0.6 * n)); n_global = int(round(0.3 * n)); n_boundary = n - n_damage - n_global
    else:
        n_damage = int(round(0.8 * n)); n_global = int(round(0.15 * n)); n_boundary = n - n_damage - n_global
    blocks = [lhs_points(n_global, seed=rng), boundary_points(n_boundary, seed=rng)]
    if n_damage:
        alpha = np.asarray(alpha_vals if alpha_vals is not None else [], dtype=float)
        active = np.flatnonzero(alpha > 1e-3)
        if active.size:
            centres = np.column_stack([np.asarray(x_vals)[active], np.asarray(y_vals)[active]])
            focused = _focused_points(n_damage, centres, np.asarray(r_vals)[active],
                                       alpha[active], rng, stage == 3)
        else:
            focused = lhs_points(n_damage, seed=rng)
        blocks.append(focused)
    return tuple(np.vstack([block[i] for block in blocks]) for i in range(3))


def adaptive_points(n, stage, epoch, x_vals, y_vals, r_vals, alpha_vals,
                    early_phase=2000, seed=None):
    """Backward-compatible wrapper used by :class:`PINN`."""
    return generate_pde_points(n, stage, epoch, alpha_vals, x_vals, y_vals,
                               r_vals, early_phase, seed)


def load_csv_folder(folder) -> pd.DataFrame:
    """Load sorted CSV files, rejecting malformed or non-finite observations."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {folder}")
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{path} contains NaN or infinite values")
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["x", "y", "t"], kind="stable").reset_index(drop=True)


def uniform_grid_sample(df, grid_size=15, num_time_points=80, tol=0.01):
    """Select nearest unique sensors on a regular target grid and sample time."""
    if df.empty:
        raise ValueError("Input dataframe is empty")
    if grid_size < 2 or num_time_points < 1 or tol <= 0:
        raise ValueError("grid_size >= 2, num_time_points >= 1, and tol > 0 are required")
    sensors = df[["x", "y"]].drop_duplicates().to_numpy()
    targets = np.array(np.meshgrid(np.linspace(df.x.min(), df.x.max(), grid_size),
                                   np.linspace(df.y.min(), df.y.max(), grid_size))).reshape(2, -1).T
    selected = set()
    for target in targets:
        distances = np.linalg.norm(sensors - target, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tol:
            selected.add(tuple(sensors[nearest]))
    if not selected:
        raise RuntimeError("No spatial sensor matched the requested grid within tolerance")
    pieces = []
    for x, y in sorted(selected):
        rows = df[(df.x == x) & (df.y == y)].sort_values("t", kind="stable")
        idx = np.linspace(0, len(rows) - 1, min(num_time_points, len(rows)), dtype=int)
        pieces.append(rows.iloc[np.unique(idx)])
    return pd.concat(pieces, ignore_index=True)


def split_by_sensor(df, val_fraction=0.15, test_fraction=0.15, seed=1234):
    """Split whole spatial sensor groups to prevent temporal leakage."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("fractions must be non-negative and sum to less than one")
    sensors = df[["x", "y"]].drop_duplicates().to_numpy()
    if len(sensors) < 3:
        raise ValueError("At least three unique sensor locations are required for splitting")
    order = _rng(seed).permutation(len(sensors))
    n_test = max(1, int(round(test_fraction * len(sensors))))
    n_val = max(1, int(round(val_fraction * len(sensors))))
    if n_test + n_val >= len(sensors):
        n_test = n_val = 1
    labels = {}
    for i, idx in enumerate(order):
        labels[tuple(sensors[idx])] = ("test" if i < n_test else
                                      "validation" if i < n_test + n_val else "train")
    groups = df.apply(lambda row: labels[(row.x, row.y)], axis=1)
    return tuple(df.loc[groups == name].reset_index(drop=True)
                 for name in ("train", "validation", "test"))
