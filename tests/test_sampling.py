import numpy as np
import pandas as pd
import pytest

from src.sampling import (boundary_points, generate_pde_points, lhs_points,
                          split_by_sensor)


def test_lhs_is_seeded_and_bounded():
    first = lhs_points(17, seed=42)
    second = lhs_points(17, seed=42)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert all(a.shape == (17, 1) for a in first)
    assert all(np.all((a >= 0) & (a <= 1)) for a in first)


@pytest.mark.parametrize("n", [0, 1, 7, 20])
def test_boundary_sampler_returns_exact_count(n):
    t, x, y = boundary_points(n, seed=3)
    assert t.shape == x.shape == y.shape == (n, 1)
    assert np.all((x == 0) | (x == 1) | (y == 0) | (y == 1))


@pytest.mark.parametrize("stage,epoch", [(1, 0), (2, 0), (2, 3000), (3, 0)])
def test_adaptive_sampler_returns_exact_count(stage, epoch):
    arrays = generate_pde_points(101, stage, epoch,
                                 alpha_vals=np.array([0.8]),
                                 x_vals=np.array([0.5]), y_vals=np.array([0.4]),
                                 r_vals=np.array([0.03]), seed=9)
    assert all(a.shape == (101, 1) for a in arrays)
    assert all(np.isfinite(a).all() and np.all((a >= 0) & (a <= 1)) for a in arrays)


def test_sensor_split_has_no_location_leakage():
    rows = [{"t": t, "x": x, "y": y, "u": x + y + t}
            for x in range(5) for y in range(4) for t in (0.0, 0.5, 1.0)]
    train, validation, test = split_by_sensor(pd.DataFrame(rows), seed=4)
    locations = [set(map(tuple, frame[["x", "y"]].to_numpy()))
                 for frame in (train, validation, test)]
    assert locations[0].isdisjoint(locations[1])
    assert locations[0].isdisjoint(locations[2])
    assert locations[1].isdisjoint(locations[2])
