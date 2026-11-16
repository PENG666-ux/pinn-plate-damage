"""Headless diagnostics using model estimates, without assumed reference damage.

The supported plotting API is ``save_diagnostics``. Legacy plotting helpers that
required unavailable attributes or embedded a particular damage truth were removed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

__all__ = ["save_diagnostics"]


def save_diagnostics(models, final_model, output: str | Path) -> list[Path]:
    """Save logged losses and all inferred candidates; return created paths.

    Geometry comes from the model's physical scales, including checkpoint-restored
    scales. Gate values indicate candidate activation, not classification confidence.
    No reference position, radius, or damage count is inferred from an input filename.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for label, model in models.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(model.ep_log, model.loss_log, marker=".", label="total")
        ax.set(xlabel="Epoch", ylabel="Weighted loss", title=label)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output / f"{label}_loss.png"
        fig.savefig(path, dpi=160)
        paths.append(path)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    parameters = (
        final_model.x_i, final_model.y_i, final_model.r_i, final_model.alpha
    )
    arrays = [parameter.detach().cpu().numpy() for parameter in parameters]
    for index, (x, y, radius, gate) in enumerate(zip(*arrays, strict=True)):
        x_mm, y_mm = x * final_model.Lx * 1000, y * final_model.Ly * 1000
        ax.add_patch(Circle(
            (x_mm, y_mm), radius * 1000,
            fill=False, color=f"C{index % 10}",
        ))
        ax.text(x_mm, y_mm, f"{index}: gate={gate:.3f}", fontsize=8)
    ax.set(
        xlim=(0, final_model.Lx * 1000), ylim=(0, final_model.Ly * 1000),
        xlabel="x (mm)", ylabel="y (mm)",
        title="Estimated candidates (no reference)",
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    path = output / "damage_estimate.png"
    fig.savefig(path, dpi=160)
    paths.append(path)
    plt.close(fig)
    return paths
