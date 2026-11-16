"""Analytic and state-transition checks for the plate inverse model.

These tests use small CPU models; they do not assert reproduction of the paper's
experimental errors or substitute synthetic observations for experimental data.
"""

import math

import numpy as np
import torch

import config as C
import pinn as pinn_module


def test_rectangular_plate_mode_satisfies_healthy_pde(model_factory, monkeypatch):
    """An exact KL eigenmode detects missing fourth-order coordinate factors."""
    monkeypatch.setattr(C, "X_PHYSICAL", 0.30)
    monkeypatch.setattr(C, "Y_PHYSICAL", 0.19)
    model = model_factory()
    model.set_stage(1)
    wave_number_squared = (math.pi / model.Lx) ** 2 + (
        2 * math.pi / model.Ly
    ) ** 2
    omega_normalized = (
        math.sqrt(model.D / (model.rho * model.h))
        * wave_number_squared
        * model.T_phys
    )

    def mode(t, x, y):
        return (
            torch.sin(math.pi * x)
            * torch.sin(2 * math.pi * y)
            * torch.cos(omega_normalized * t)
        )

    monkeypatch.setattr(model, "forward", mode)
    t = torch.tensor([[0.12], [0.37], [0.61]], dtype=torch.float64)
    x = torch.tensor([[0.21], [0.43], [0.67]], dtype=torch.float64)
    y = torch.tensor([[0.18], [0.32], [0.73]], dtype=torch.float64)
    residual = model.pde_residual(t, x, y)
    torch.testing.assert_close(
        residual, torch.zeros_like(residual), atol=1e-8, rtol=0
    )


def test_input_mapping_uses_fixed_normalized_domain(model_factory):
    """Interior-only observations cannot redefine the physical derivative scale."""
    model = model_factory()
    endpoints = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64
    )
    torch.testing.assert_close(model._scale(endpoints), 2 * endpoints - 1)


def test_adaptive_refresh_preserves_predictions_and_scaling(
    model_factory, monkeypatch
):
    model = model_factory()
    model.set_stage(3)
    coordinates = torch.tensor(
        [[0.17, 0.31, 0.23], [0.73, 0.57, 0.89]], dtype=torch.float64
    )
    t, x, y = coordinates.split(1, dim=1)
    before = model.forward(t, x, y).detach().clone()
    scale_before = model._scale(coordinates).detach().clone()

    def shifted_points(*args, **kwargs):
        values = np.linspace(0.43, 0.57, C.N_PDE).reshape(-1, 1)
        return values, values.copy(), values.copy()

    monkeypatch.setattr(pinn_module, "adaptive_points", shifted_points)
    model._refresh_pde_points(C.ADAPTIVE_UPDATE_INTERVAL + 1)
    assert model.last_adapt == C.ADAPTIVE_UPDATE_INTERVAL + 1
    torch.testing.assert_close(model._scale(coordinates), scale_before)
    torch.testing.assert_close(model.forward(t, x, y), before)


def test_density_uses_radial_sigmoid_and_physical_circles(
    model_factory, monkeypatch
):
    """Equal metre offsets on a rectangle must yield equal mask values."""
    monkeypatch.setattr(C, "X_PHYSICAL", 0.30)
    monkeypatch.setattr(C, "Y_PHYSICAL", 0.18)
    model = model_factory()
    model.set_stage(2)
    model.init_damage_params(
        positions=[(0.4, 0.5)], radii_mm=[12.0], alphas=[0.7]
    )
    x_center = model.x_i[0].detach()
    y_center = model.y_i[0].detach()
    radius = model.r_i[0].detach()
    offsets = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
    x = (x_center + offsets * radius / model.Lx).reshape(-1, 1)
    y = torch.full_like(x, y_center)
    density = model.effective_density(x, y)
    distance_normalized = offsets.reshape(-1, 1) * radius / model.Lx
    expected = model.rho * model.h + (
        model.alpha[0]
        * model.delta_sig
        * torch.sigmoid(
            model.beta * (radius / model.Lx - distance_normalized)
        )
    )
    torch.testing.assert_close(density, expected, atol=1e-6, rtol=1e-7)
    assert bool(torch.all(density[:-1] > density[1:]))
    center_density = model.effective_density(
        x_center.reshape(1, 1), y_center.reshape(1, 1)
    )
    assert center_density.item() > density[0].item()

    x_radial = (x_center + radius / model.Lx).reshape(1, 1)
    y_radial = (y_center + radius / model.Ly).reshape(1, 1)
    torch.testing.assert_close(
        model.effective_density(x_radial, y_center.reshape(1, 1)),
        model.effective_density(x_center.reshape(1, 1), y_radial),
    )

    model.set_stage(1)
    healthy = model.effective_density(x, y)
    torch.testing.assert_close(
        healthy, torch.full_like(healthy, model.rho * model.h)
    )


def test_stage_three_freezes_network_but_preserves_damage_gradients(model_factory):
    model = model_factory()
    model.init_damage_params(
        positions=[(0.4, 0.5)], radii_mm=[12.0], alphas=[0.7]
    )
    model.set_stage(3)
    assert all(not parameter.requires_grad for parameter in model.net.parameters())
    damage = [model.raw_alpha, model.raw_x_i, model.raw_y_i, model.raw_r_i]
    assert all(parameter.requires_grad for parameter in damage)
    optimizer_parameters = {
        id(parameter)
        for group in model.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameters == {id(parameter) for parameter in damage}

    t = torch.tensor([[0.20], [0.36], [0.57]], dtype=torch.float64)
    x = torch.tensor([[0.42], [0.44], [0.47]], dtype=torch.float64)
    y = torch.tensor([[0.51], [0.54], [0.56]], dtype=torch.float64)
    model.optimizer.zero_grad(set_to_none=True)
    model.pde_residual(t, x, y).square().mean().backward()
    assert all(parameter.grad is None for parameter in model.net.parameters())
    for parameter in damage:
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert parameter.grad.abs().sum().item() > 0


def test_checkpoint_restores_predictions_and_raw_damage(model_factory, tmp_path):
    model = model_factory()
    model.init_damage_params(
        positions=[(0.36, 0.67)], radii_mm=[7.5], alphas=[0.82]
    )
    model.set_stage(3)
    coordinates = torch.tensor(
        [[0.12, 0.25, 0.62], [0.76, 0.83, 0.41]], dtype=torch.float64
    )
    t, x, y = coordinates.split(1, dim=1)
    prediction = model.forward(t, x, y).detach().clone()
    checkpoint = tmp_path / "model.pt"
    model.save_model(checkpoint)
    restored = model_factory()
    with torch.no_grad():
        next(restored.net.parameters()).add_(0.4)
    restored.load_model(checkpoint)
    assert restored.stage == model.stage
    torch.testing.assert_close(restored.forward(t, x, y), prediction)
    for name in ("raw_alpha", "raw_x_i", "raw_y_i", "raw_r_i"):
        torch.testing.assert_close(getattr(restored, name), getattr(model, name))
    assert all(not parameter.requires_grad for parameter in restored.net.parameters())


def test_checkpoint_restores_physics_independently_of_current_config(
    model_factory, monkeypatch, tmp_path
):
    settings = {
        "E": 6.1e10, "NU": 0.29, "RHO0": 2650.0,
        "H_PLATE": 0.0012, "X_PHYSICAL": 0.33, "Y_PHYSICAL": 0.18,
        "T_PHYSICAL": 1.1e-4, "DELTA_SIGMA": 175.0, "BETA": 83.0,
    }
    original = {name: getattr(C, name) for name in settings}
    for name, value in settings.items():
        monkeypatch.setattr(C, name, value)
    model = model_factory()
    model.set_stage(3)
    checkpoint = tmp_path / "physical_model.pt"
    model.save_model(checkpoint)
    t = torch.tensor([[0.23], [0.41]], dtype=torch.float64)
    x = torch.tensor([[0.47], [0.55]], dtype=torch.float64)
    y = torch.tensor([[0.43], [0.58]], dtype=torch.float64)
    residual = model.pde_residual(t, x, y).detach()

    for name, value in original.items():
        monkeypatch.setattr(C, name, value)
    restored = model_factory()
    assert restored.Ly != model.Ly
    restored.load_model(checkpoint)
    for name in ("E", "nu", "rho", "h", "Lx", "Ly", "T_phys",
                 "D", "delta_sig", "beta"):
        assert getattr(restored, name) == getattr(model, name)
    torch.testing.assert_close(restored.pde_residual(t, x, y), residual)
