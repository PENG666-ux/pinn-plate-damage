"""Reproducible collocation sampling and validated, normalised CSV data."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc


def _count(n: int) -> int:
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("Sample count must be a non-negative integer.")
    return int(n)


def _columns(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[:, 0:1], points[:, 1:2], points[:, 2:3]


def lhs_points(
    n: int,
    t_range: tuple[float, float] = (0.0, 1.0),
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latin hypercube points; omitted seed respects ``np.random.seed``."""
    n = _count(n)
    bounds = np.asarray([t_range, x_range, y_range], dtype=float)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Each sampling range must have finite lower < upper.")
    if n == 0:
        return _columns(np.empty((0, 3)))
    if seed is None:
        seed = int(np.random.randint(0, 2**32 - 1, dtype=np.int64))
    points = qmc.LatinHypercube(d=3, seed=seed).random(n=n)
    return _columns(qmc.scale(points, bounds[:, 0], bounds[:, 1]))


def boundary_points(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly n points on spatial edges, with t uniform on [0, 1]."""
    n = _count(n)
    points = np.random.uniform(size=(n, 3))
    edge = np.arange(n) % 4
    points[edge == 0, 1] = 0.0
    points[edge == 1, 1] = 1.0
    points[edge == 2, 2] = 0.0
    points[edge == 3, 2] = 1.0
    return _columns(points)


def _site_counts(n, centres, radii, weights):
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    radii = np.asarray(radii, dtype=float)
    weights = np.asarray(weights, dtype=float).copy()
    if len(centres) != len(radii) or len(radii) != len(weights):
        raise ValueError("Centres, radii and weights must have equal lengths.")
    if not len(weights):
        return centres, radii, np.empty(0, dtype=int)
    if (
        not np.isfinite(centres).all()
        or not np.isfinite(radii).all()
        or not np.isfinite(weights).all()
        or np.any(radii < 0)
        or np.any(weights < 0)
    ):
        raise ValueError("Damage sites and weights must be finite and non-negative.")
    if np.any((centres < 0) | (centres > 1)):
        raise ValueError("Normalised damage centres must be in [0, 1].")
    if weights.sum() == 0:
        return centres, radii, np.empty(0, dtype=int)
    return centres, radii, np.random.multinomial(n, weights / weights.sum())


def _damage_points(n, centres, radii, weights, hires=False, sigma_factor=2.0):
    n = _count(n)
    if n == 0:
        return _columns(np.empty((0, 3)))
    centres, radii, counts = _site_counts(n, centres, radii, weights)
    if not len(counts):
        return lhs_points(n)
    blocks = []
    for (cx, cy), radius, count in zip(centres, radii, counts):
        if count == 0:
            continue
        sigma = max(radius * sigma_factor, 0.03 if hires else 0.05)
        xs = np.random.normal(cx, sigma, count)
        ys = np.random.normal(cy, sigma, count)
        if hires:
            n_inside = int(count * 0.7)
            rs = radius * np.sqrt(np.random.uniform(size=n_inside))
            angles = np.random.uniform(0.0, 2.0 * np.pi, n_inside)
            xs[:n_inside] = cx + rs * np.cos(angles)
            ys[:n_inside] = cy + rs * np.sin(angles)
        blocks.append(
            np.column_stack(
                [np.random.uniform(size=count), np.clip(xs, 0, 1), np.clip(ys, 0, 1)]
            )
        )
    return _columns(np.vstack(blocks))


def damage_focused_points(n, centres, radii, weights, sigma_factor=2.0):
    """Exactly n Gaussian samples around sites, distributed by gate weights."""
    return _damage_points(n, centres, radii, weights, sigma_factor=sigma_factor)


def damage_focused_points_hires(n, centres, radii, weights):
    """Approximately 70% within discs and 30% nearby; exactly n points total."""
    return _damage_points(n, centres, radii, weights, hires=True, sigma_factor=1.5)


def generate_pde_points(
    N,
    stage=1,
    epoch=0,
    alpha_vals=None,
    x_vals=None,
    y_vals=None,
    r_vals=None,
    stage2_switch=2000,
):
    """Collocation mixture. Edge samples enforce the PDE, not a boundary loss."""
    N = _count(N)
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2 or 3.")
    if stage == 1 or (stage == 2 and epoch < stage2_switch):
        n_global = int(N * 0.8)
        groups = [lhs_points(n_global), boundary_points(N - n_global)]
    else:
        n_damage = int(N * (0.6 if stage == 2 else 0.8))
        n_global = int(N * (0.3 if stage == 2 else 0.15))
        n_boundary = N - n_damage - n_global
        alpha = np.asarray(alpha_vals if alpha_vals is not None else [], dtype=float)
        active = np.flatnonzero(alpha > 1e-3)
        if len(active):
            if any(v is None for v in (x_vals, y_vals, r_vals)):
                raise ValueError("Active gates require x_vals, y_vals and r_vals.")
            centres = list(zip(np.asarray(x_vals)[active], np.asarray(y_vals)[active]))
            sampler = damage_focused_points_hires if stage == 3 else damage_focused_points
            damage = sampler(
                n_damage, centres, np.asarray(r_vals)[active], alpha[active]
            )
        else:
            damage = lhs_points(n_damage)
        groups = [lhs_points(n_global), boundary_points(n_boundary), damage]
    return tuple(np.vstack([group[i] for group in groups]) for i in range(3))


def to_tensor(arr, device, dtype=torch.float64):
    """Convert an array to the training dtype and device."""
    return torch.as_tensor(arr, dtype=dtype, device=device)


def validate_dataframe(df: pd.DataFrame, source: str = "data") -> pd.DataFrame:
    """Validate already normalised t,x,y,u without guessing physical units.

    All datasets must share the same physical normalisation scales. This function
    does not normalise each dataset independently.
    """
    required = ["t", "x", "y", "u"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{source}: missing CSV columns {sorted(missing)}.")
    if df.empty:
        raise ValueError(f"{source}: no data rows.")
    try:
        df = df.loc[:, required].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: t,x,y,u must all be numeric.") from exc
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"{source}: data contains NaN or infinity.")
    if ((df[["t", "x", "y"]] < 0) | (df[["t", "x", "y"]] > 1)).any().any():
        raise ValueError(f"{source}: normalised t,x,y must lie in [0,1].")
    df = df.drop_duplicates().sort_values(["x", "y", "t"], kind="stable")
    if df.duplicated(["t", "x", "y"]).any():
        raise ValueError(f"{source}: conflicting u values at the same t,x,y.")
    return df.reset_index(drop=True)


def load_csv_folder(folder: str | Path) -> pd.DataFrame:
    """Read every CSV deterministically and fail on malformed input."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"CSV folder does not exist: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".csv")
    if not files:
        raise ValueError(f"No CSV files found in {folder}.")
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"Cannot read CSV {path}: {exc}") from exc
        frames.append(validate_dataframe(df, str(path)))
    return validate_dataframe(pd.concat(frames, ignore_index=True), str(folder))


def uniform_grid_sample(df, grid_size=15, num_time_points=80, tol=0.01):
    """Select distinct nearest sensor locations, then ordered time snapshots.

    Tolerance is the maximum absolute coordinate difference. Nearby sensors
    remain distinct; a location selected twice is used only once.
    """
    if grid_size < 1 or num_time_points < 1 or tol <= 0:
        raise ValueError("grid_size, num_time_points and tol must be positive.")
    df = validate_dataframe(df)
    locations = df[["x", "y"]].drop_duplicates().to_numpy()
    xs = np.linspace(locations[:, 0].min(), locations[:, 0].max(), grid_size)
    ys = np.linspace(locations[:, 1].min(), locations[:, 1].max(), grid_size)
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    distances, indices = cKDTree(locations).query(grid, p=np.inf)
    selected = sorted(set(indices[distances <= tol].tolist()))
    if not selected:
        raise ValueError("No spatial grid locations matched within tolerance.")
    groups = df.groupby(["x", "y"], sort=False)
    pieces = []
    for index in selected:
        location = tuple(locations[index])
        rows = groups.get_group(location).sort_values("t", kind="stable")
        count = min(num_time_points, len(rows))
        positions = np.linspace(0, len(rows) - 1, count, dtype=int)
        pieces.append(rows.iloc[positions])
    result = pd.concat(pieces, ignore_index=True)
    print(f"Grid sampling: {len(selected)} distinct locations, {len(result)} rows")
    return result
