"""Make the script-style source tree importable during repository tests."""

from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import pytest
import torch

import config as C
from pinn import PINN


@pytest.fixture
def model_factory(monkeypatch):
    """Build cheap, deterministic models without visualization side effects."""
    monkeypatch.setattr(C, "WIDTH", 8)
    monkeypatch.setattr(C, "DEPTH", 2)
    monkeypatch.setattr(C, "K_MAX", 1)
    monkeypatch.setattr(C, "N_PDE", 12)
    monkeypatch.setattr(C, "F_MNTR", 100_000)
    monkeypatch.setattr(C, "ENABLE_PLOTS", False)

    def build():
        generator = torch.Generator().manual_seed(29)
        coordinates = 0.2 + 0.6 * torch.rand(
            (12, 3), generator=generator, dtype=torch.float64
        )
        t, x, y = coordinates.split(1, dim=1)
        return PINN(
            t, x, y, torch.sin(t) * torch.sin(x) * torch.sin(y),
            t.clone(), x.clone(), y.clone(), device="cpu",
        )

    return build
