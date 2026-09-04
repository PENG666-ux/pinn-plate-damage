"""Minimal, auditable plots that do not assume access to ground truth."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(model, save_path):
    """Plot logged objective components on a logarithmic scale."""
    path = Path(save_path); path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray(model.history["epoch"])
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for key in ("total", "pde", "boundary", "data"):
        values = np.asarray(model.history[key], dtype=float)
        if len(values):
            ax.semilogy(epochs, np.maximum(values, np.finfo(float).tiny), label=key)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training objective components")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.savefig(path, dpi=300); plt.close(fig)


def plot_damage_map(model, save_path, threshold=0.05):
    """Plot inferred mass perturbation; no unavailable truth is overlaid."""
    path = Path(save_path); path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, 301)
    x, y = np.meshgrid(grid, grid)
    density = np.zeros_like(x)
    params = model.damage_parameters()
    for alpha, cx, cy, radius_mm in zip(params["alpha"], params["x"],
                                         params["y"], params["radius_mm"]):
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radius = radius_mm / (1000.0 * model.Lx)
        density += alpha * model.delta_sig / (1.0 + np.exp(
            np.clip(-model.beta * (radius - distance), -700, 700)))
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    image = ax.pcolormesh(x * model.Lx * 1000, y * model.Ly * 1000, density,
                          shading="auto", cmap="magma", vmin=0)
    for index, (alpha, cx, cy, radius_mm) in enumerate(zip(
            params["alpha"], params["x"], params["y"], params["radius_mm"])):
        if alpha >= threshold:
            circle = plt.Circle((cx * model.Lx * 1000, cy * model.Ly * 1000),
                                radius_mm, fill=False, color="cyan", linewidth=1)
            ax.add_patch(circle); ax.text(cx * model.Lx * 1000, cy * model.Ly * 1000,
                                          f" {index}: alpha={alpha:.2f}", color="white")
    fig.colorbar(image, ax=ax, label=r"Inferred areal-density change (kg m$^{-2}$)")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="Inferred damage field")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=300); plt.close(fig)


plot_loss_curve = plot_stage_total_loss = plot_history
