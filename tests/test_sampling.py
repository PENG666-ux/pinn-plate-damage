"""Regression checks for reproducibility, exact counts, and CSV semantics."""

import numpy as np
import pandas as pd
import pytest

from src.sampling import (
    boundary_points,
    damage_focused_points,
    damage_focused_points_hires,
    generate_pde_points,
    lhs_points,
    load_csv_folder,
    uniform_grid_sample,
    validate_dataframe,
)


@pytest.mark.parametrize("n", [0, 1, 2, 7, 101])
@pytest.mark.parametrize("sampler", [damage_focused_points, damage_focused_points_hires])
def test_focused_sampling_always_returns_requested_count(n, sampler):
    weights = np.array([0.8, 0.2])
    before = weights.copy()
    result = sampler(n, [(0.3, 0.4), (0.7, 0.6)], [0.02, 0.04], weights)
    for column in result:
        assert column.shape == (n, 1)
        assert column.dtype == np.float64
        assert np.isfinite(column).all()
        assert ((column >= 0) & (column <= 1)).all()
    np.testing.assert_array_equal(weights, before)


@pytest.mark.parametrize("sampler", [damage_focused_points, damage_focused_points_hires])
def test_zero_weights_fall_back_to_global_points(sampler):
    result = sampler(10, [(0.5, 0.5)], [0.03], [0.0])
    assert np.hstack(result).shape == (10, 3)
    assert np.isfinite(np.hstack(result)).all()


@pytest.mark.parametrize("n", [0, 1, 11])
def test_boundary_points_are_on_an_edge(n):
    t, x, y = boundary_points(n)
    assert t.shape == x.shape == y.shape == (n, 1)
    assert ((x == 0) | (x == 1) | (y == 0) | (y == 1)).all()


def test_lhs_respects_explicit_and_global_seeds():
    first = np.hstack(lhs_points(30, seed=12))
    np.testing.assert_array_equal(first, np.hstack(lhs_points(30, seed=12)))
    np.random.seed(18)
    first = np.hstack(lhs_points(30))
    np.random.seed(18)
    np.testing.assert_array_equal(first, np.hstack(lhs_points(30)))
    for coordinate in first.T:
        assert len(np.unique(np.floor(coordinate * 30))) == 30


@pytest.mark.parametrize("n", [0, 1, 7, 101])
@pytest.mark.parametrize("stage,epoch", [(1, 0), (2, 0), (2, 2000), (3, 0)])
def test_pde_mixtures_have_exact_counts(n, stage, epoch):
    points = generate_pde_points(
        n, stage=stage, epoch=epoch,
        alpha_vals=[0.8, 0.2], x_vals=[0.3, 0.7], y_vals=[0.4, 0.6],
        r_vals=[0.02, 0.04],
    )
    assert np.hstack(points).shape == (n, 3)


@pytest.fixture
def data():
    return pd.DataFrame({
        "t": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5],
        "x": [0.0, 0.0, 0.0, 0.005, 0.005, 0.005],
        "y": [0.0] * 6, "u": [1.0, 0.0, 0.5, 0.2, 0.0, 0.1],
    })


def test_sampling_keeps_nearby_sensors_distinct_and_orders_time(data):
    sampled = uniform_grid_sample(data, grid_size=3, num_time_points=2, tol=0.01)
    assert len(sampled) == 4
    assert not sampled.duplicated(["t", "x", "y"]).any()
    for _, sensor in sampled.groupby(["x", "y"]):
        assert sensor["t"].tolist() == [0.0, 1.0]


def test_csv_loading_deduplicates_identical_rows(data, tmp_path):
    data.to_csv(tmp_path / "b.csv", index=False)
    data.iloc[:1].to_csv(tmp_path / "a.CSV", index=False)
    loaded = load_csv_folder(tmp_path)
    assert len(loaded) == len(data)
    assert loaded["t"].tolist() == [0.0, 0.5, 1.0] * 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "bad"])
def test_invalid_numeric_measurements_are_rejected(data, value):
    data["u"] = data["u"].astype(object)
    data.loc[0, "u"] = value
    with pytest.raises(ValueError, match="numeric|NaN|infinity"):
        validate_dataframe(data)


def test_conflicting_measurements_are_rejected(data):
    conflict = data.iloc[:1].copy()
    conflict["u"] = 123.0
    with pytest.raises(ValueError, match="conflicting"):
        validate_dataframe(pd.concat([data, conflict], ignore_index=True))


def test_missing_columns_empty_folder_and_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv_folder(tmp_path / "missing")
    with pytest.raises(ValueError, match="No CSV"):
        load_csv_folder(tmp_path)
    pd.DataFrame({"t": [0]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing CSV columns"):
        load_csv_folder(tmp_path)


def test_physical_units_are_not_silently_normalised(data):
    data.loc[0, "x"] = 300.0
    with pytest.raises(ValueError, match="normalised"):
        validate_dataframe(data)
