"""Command-line training and held-out evaluation entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

try:
    from . import config as C
    from .pinn import PINN
    from .plot import plot_damage_map, plot_history
    from .sampling import (generate_pde_points, load_csv_folder,
                           split_by_sensor, uniform_grid_sample)
except ImportError:
    import config as C
    from pinn import PINN
    from plot import plot_damage_map, plot_history
    from sampling import (generate_pde_points, load_csv_folder,
                          split_by_sensor, uniform_grid_sample)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthy-data", type=Path, default=Path(C.DATA_HEALTHY))
    parser.add_argument("--damaged-data", type=Path, default=Path(C.DATA_DAMAGE))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--coordinate-mode", choices=("normalized", "physical"),
                        default="normalized",
                        help="Units of t,x,y in CSV files; physical means seconds/metres.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=C.R_SEED)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one Adam epoch per stage and skip L-BFGS.")
    return parser.parse_args(argv)


def normalize_coordinates(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    frame = frame.copy()
    if mode == "physical":
        frame["t"] /= C.T_PHYSICAL
        frame["x"] /= C.X_PHYSICAL
        frame["y"] /= C.Y_PHYSICAL
    for column in ("t", "x", "y"):
        minimum, maximum = frame[column].min(), frame[column].max()
        if minimum < -1e-10 or maximum > 1.0 + 1e-10:
            raise ValueError(
                f"{column} range [{minimum:.6g}, {maximum:.6g}] is outside [0,1]. "
                "Choose the correct --coordinate-mode and verify physical constants.")
    return frame


def prepare_data(path: Path, mode: str):
    frame = normalize_coordinates(load_csv_folder(path), mode)
    return uniform_grid_sample(frame, C.GRID_SIZE, C.N_TIME_POINTS, C.SAMPLING_TOL)


def tensors(frame, device):
    return tuple(torch.tensor(frame[col].to_numpy().reshape(-1, 1),
                              dtype=torch.float64, device=device)
                 for col in ("t", "x", "y", "u"))


def build_model(frame, device, stage_seed):
    data = tensors(frame, device)
    pde = generate_pde_points(C.N_PDE, seed=stage_seed)
    return PINN(*data, *pde, device=device)


def split_manifest(train, validation, test):
    def describe(frame):
        return {"observations": len(frame),
                "sensors": len(frame[["x", "y"]].drop_duplicates()),
                "time_min": float(frame.t.min()), "time_max": float(frame.t.max())}
    return {name: describe(frame) for name, frame in
            (("train", train), ("validation", validation), ("test", test))}


def main(argv=None):
    args = parse_args(argv)
    C.R_SEED = args.seed
    if args.smoke_test:
        C.N_PDE, C.N_BC, C.F_MNTR = 128, 32, 1
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    healthy = prepare_data(args.healthy_data, args.coordinate_mode)
    damaged = prepare_data(args.damaged_data, args.coordinate_mode)
    h_train, h_val, h_test = split_by_sensor(healthy, C.VAL_FRACTION,
                                             C.TEST_FRACTION, args.seed)
    d_train, d_val, d_test = split_by_sensor(damaged, C.VAL_FRACTION,
                                             C.TEST_FRACTION, args.seed)
    manifest = {"healthy": split_manifest(h_train, h_val, h_test),
                "damaged": split_manifest(d_train, d_val, d_test)}
    (args.output_dir / "data_split.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    adam1, lbfgs, adam2, adam3 = ((1, 0, 1, 1) if args.smoke_test else
                                  (C.S1_ADAM_EPOCHS, C.S1_LBFGS_EPOCHS,
                                   C.S2_EPOCHS, C.S3_EPOCHS))

    model1 = build_model(h_train, device, args.seed)
    model1.w_data = C.W_DATA_S1
    model1.train(adam1, min(512, len(h_train)), stage=1,
                 validation_data=tensors(h_val, device))
    if lbfgs:
        model1.train_lbfgs(lbfgs, validation_data=tensors(h_val, device))
    model1.save_model(args.output_dir / "stage1.pt")

    model2 = build_model(d_train, device, args.seed + 1)
    checkpoint = torch.load(args.output_dir / "stage1.pt", map_location=device,
                            weights_only=False)
    model2.net.load_state_dict(checkpoint["net"])
    model2.w_data, model2.w_reg_r, model2.w_reg_g = (
        C.W_DATA_S2, C.W_REG_R_S2, C.W_REG_G_S2)
    model2.train(adam2, min(256, len(d_train)), stage=2,
                 validation_data=tensors(d_val, device))
    model2.save_model(args.output_dir / "stage2.pt")

    model3 = build_model(d_train, device, args.seed + 2)
    model3.load_model(args.output_dir / "stage2.pt")
    model3.w_data, model3.w_reg_r, model3.w_reg_g = (
        C.W_DATA_S3, C.W_REG_R_S3, C.W_REG_G_S3)
    model3.train(adam3, min(256, len(d_train)), stage=3,
                 start_epoch=adam2, validation_data=tensors(d_val, device))
    model3.save_model(args.output_dir / "stage3.pt")
    model3.save_damage_params(args.output_dir / "damage_parameters.npz")

    metrics = {"schema_version": C.SCHEMA_VERSION,
               "device": str(device), "seed": args.seed,
               "coordinate_mode": args.coordinate_mode,
               "elapsed_seconds": time.time() - started,
               "healthy": {"validation": model1.evaluate(tensors(h_val, device)),
                           "test": model1.evaluate(tensors(h_test, device))},
               "damaged": {"validation": model3.evaluate(tensors(d_val, device)),
                           "test": model3.evaluate(tensors(d_test, device))}}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    plot_history(model3, args.output_dir / "training_history.png")
    plot_damage_map(model3, args.output_dir / "damage_map.png")
    print(json.dumps(metrics["damaged"]["test"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
