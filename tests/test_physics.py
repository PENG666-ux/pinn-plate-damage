import numpy as np
import torch

from src import config as C
from src.pinn import PINN
from src.sampling import lhs_points


class SimplySupportedMode(torch.nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.omega = omega

    def forward(self, scaled):
        t, x, y = ((scaled[:, i:i + 1] + 1.0) / 2.0 for i in range(3))
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.cos(self.omega * t)


def make_model(monkeypatch):
    monkeypatch.setattr(C, "N_BC", 16)
    t, x, y = lhs_points(24, seed=8)
    u = np.sin(np.pi * x) * np.sin(np.pi * y)
    return PINN(t, x, y, u, t, x, y, device="cpu")


def test_manufactured_solution_satisfies_healthy_plate(monkeypatch):
    model = make_model(monkeypatch)
    model.set_stage(1)
    eigenvalue = np.pi**4 * (1.0 + model.aspect2) ** 2
    omega = np.sqrt(eigenvalue / model.pde_scale)
    model.net = SimplySupportedMode(omega)
    t, x, y = (torch.tensor(a, dtype=torch.float64)
               for a in lhs_points(31, seed=11))
    residual = model.pde_residual(t, x, y)
    assert residual.abs().max().item() < 1e-10


def test_radius_parameterization_is_bounded(monkeypatch):
    model = make_model(monkeypatch)
    model.set_stage(2)
    with torch.no_grad():
        model.raw_r_i.copy_(torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float64))
    radii_mm = model.r_i.detach().numpy() * 1000
    assert np.all(radii_mm >= C.R_MIN_MM)
    assert np.all(radii_mm <= C.R_MAX_MM)


def test_simply_supported_boundary_residual(monkeypatch):
    model = make_model(monkeypatch)
    model.set_stage(1)
    # Express the manufactured physical displacement directly.
    model.u_mean.zero_()
    model.u_scale.fill_(1.0)
    eigenvalue = np.pi**4 * (1.0 + model.aspect2) ** 2
    model.net = SimplySupportedMode(np.sqrt(eigenvalue / model.pde_scale))
    displacement, moment = model.boundary_residuals(model.t_bc, model.x_bc, model.y_bc)
    assert displacement.abs().max().item() < 1e-10
    assert moment.abs().max().item() < 1e-10


def test_checkpoint_round_trip(monkeypatch, tmp_path):
    original = make_model(monkeypatch)
    original.set_stage(2)
    path = tmp_path / "model.pt"
    original.save_model(path)
    restored = make_model(monkeypatch)
    restored.load_model(path)
    before = original.forward(original.t_data, original.x_data, original.y_data)
    after = restored.forward(restored.t_data, restored.x_data, restored.y_data)
    assert torch.equal(before, after)
    assert original.damage_parameters() == restored.damage_parameters()
