"""Generate a healthy analytic mode for execution checks, not damage validation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C


def generate_demo(output: Path, spatial: int = 5, snapshots: int = 8) -> dict:
    """Write identical healthy/damaged inputs to exercise the entire pipeline.

    The simply supported free-vibration mode satisfies the homogeneous plate PDE.
    No damage is present in either file. These files cannot validate damage
    localisation, sizing, transfer to experiments, or the paper's reported errors.
    """
    if spatial < 3 or snapshots < 3:
        raise ValueError("Use at least 3 spatial coordinates and 3 snapshots.")
    tt, xx, yy = np.meshgrid(
        np.linspace(0, 1, snapshots),
        np.linspace(0, 1, spatial),
        np.linspace(0, 1, spatial), indexing="ij",
    )
    rigidity = C.E * C.H_PLATE**3 / (12 * (1 - C.NU**2))
    omega_star = (
        np.sqrt(rigidity / (C.RHO0 * C.H_PLATE))
        * np.pi**2 * (C.X_PHYSICAL**-2 + C.Y_PHYSICAL**-2)
        * C.T_PHYSICAL
    )
    uu = np.sin(np.pi * xx) * np.sin(np.pi * yy) * np.cos(omega_star * tt)
    df = pd.DataFrame({
        "t": tt.ravel(), "x": xx.ravel(), "y": yy.ravel(), "u": uu.ravel(),
    })
    output = Path(output).resolve()
    for folder in ("healthy", "damaged"):
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination / "mode.csv", index=False)
    metadata = {
        "purpose": "Engineering execution smoke test only",
        "damage_present": False,
        "damaged_folder_is_identical_to_healthy": True,
        "formula": "u=sin(pi*x)*sin(pi*y)*cos(omega_star*t)",
        "boundary_condition": "Simply supported analytic mode; not a paper experiment",
        "omega_star": float(omega_star),
        "rows_per_dataset": len(df),
        "physical_scales": {
            "x_m": C.X_PHYSICAL, "y_m": C.Y_PHYSICAL, "t_s": C.T_PHYSICAL,
        },
        "shared_displacement_scale": "Arbitrary common unit amplitude for both files",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output / "README.txt").write_text(
        "SYNTHETIC EXECUTION DEMO ONLY\n"
        "healthy/mode.csv and damaged/mode.csv contain the SAME undamaged "
        "analytic free-vibration mode. No damage was simulated.\n"
        "This verifies CSV loading, derivatives, optimisation, and persistence. "
        "It provides no evidence of damage-identification accuracy.\n"
        "All t,x,y are normalised to [0,1]; u uses the same unit amplitude.\n",
        encoding="utf-8",
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--spatial", type=int, default=5)
    parser.add_argument("--snapshots", type=int, default=8)
    args = parser.parse_args(argv)
    metadata = generate_demo(args.output, args.spatial, args.snapshots)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
