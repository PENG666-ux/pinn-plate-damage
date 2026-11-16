"""Run the healthy baseline, optional transfer, and staged damage inversion."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

try:
    from . import config as C
    from .sampling import generate_pde_points, load_csv_folder, uniform_grid_sample
except ImportError:  # Support both python -m src.main and python src/main.py.
    import config as C
    from sampling import generate_pde_points, load_csv_folder, uniform_grid_sample


def parse_args(argv=None):
    """Parse reproducible input and output locations without changing source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthy", type=Path, default=Path(C.DATA_HEALTHY))
    parser.add_argument("--damaged", type=Path, default=Path(C.DATA_DAMAGE))
    parser.add_argument(
        "--pretrain", type=Path,
        help="Optional healthy simulation CSV folder before healthy-data fine-tuning.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=C.R_SEED)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use a tiny CPU-friendly schedule; this does not assess inverse accuracy.",
    )
    parser.add_argument("--plots", action="store_true", help="Save loss and estimate plots.")
    return parser.parse_args(argv)


def apply_runtime_config(args):
    """Apply explicit overrides before constructing any PINN instances."""
    C.R_SEED = args.seed
    C.DATA_HEALTHY = str(args.healthy.resolve())
    C.DATA_DAMAGE = str(args.damaged.resolve())
    # main owns optional final plots; old per-epoch plots are intentionally disabled.
    C.ENABLE_PLOTS = False
    if args.smoke:
        overrides = dict(
            WIDTH=8, DEPTH=2, N_PDE=12, PDE_LBFGS_SUBSET=12,
            GRID_SIZE=3, N_TIME_POINTS=4, S1_ADAM_EPOCHS=1,
            S1_LBFGS_EPOCHS=1, FINETUNE_EPOCHS=1,
            S2_EPOCHS=2, S3_EPOCHS=1, S2_EARLY_PHASE=1,
            ADAPTIVE_UPDATE_INTERVAL=1, F_MNTR=1,
        )
        for name, value in overrides.items():
            setattr(C, name, value)
        torch.set_num_threads(1)
    np.random.seed(C.R_SEED)
    torch.manual_seed(C.R_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(C.R_SEED)


def df_to_tensors(df, device):
    """Use float64 for all observations and fourth-order derivatives."""
    return tuple(
        torch.as_tensor(df[col].to_numpy().reshape(-1, 1),
                        dtype=torch.float64, device=device)
        for col in ("t", "x", "y", "u")
    )


def build_model(df, device):
    """Build a PINN from data and the common collocation sampler."""
    try:
        from .pinn import PINN
    except ImportError:
        from pinn import PINN
    data = df_to_tensors(df, device)
    points = tuple(
        torch.as_tensor(array, dtype=torch.float64, device=device)
        for array in generate_pde_points(C.N_PDE, stage=1)
    )
    return PINN(*data, *points, device=device)


def _load_data(path):
    return uniform_grid_sample(
        load_csv_folder(path), grid_size=C.GRID_SIZE,
        num_time_points=C.N_TIME_POINTS, tol=C.SAMPLING_TOL,
    )


def _batch_size(args):
    return 16 if args.smoke else 512


def _fit_healthy(model, args, finetune=False):
    if finetune:
        model.set_finetune_mode()
    else:
        model.set_stage(1)
    model.w_pde = C.W_PDE
    model.w_data = C.W_DATA_S1
    epochs = C.FINETUNE_EPOCHS if finetune else C.S1_ADAM_EPOCHS
    model.train(epochs, _batch_size(args), stage=1)
    if not finetune and C.S1_LBFGS_EPOCHS > 0:
        model.train_lbfgs(
            C.S1_LBFGS_EPOCHS, subset=C.PDE_LBFGS_SUBSET,
            max_iter=2 if args.smoke else 50,
        )


def _save_plots(models, final_model, output):
    """Load plotting dependencies only when plots are requested."""
    try:
        from .plot import save_diagnostics
    except ImportError:
        from plot import save_diagnostics
    save_diagnostics(models, final_model, output)


def main(argv=None):
    """Validate all inputs, run the schedule, and save inspectable artifacts."""
    args = parse_args(argv)
    apply_runtime_config(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    healthy = _load_data(args.healthy)
    damaged = _load_data(args.damaged)
    simulated = _load_data(args.pretrain) if args.pretrain is not None else None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "settings": {key: value for key, value in vars(C).items() if key.isupper()},
        "run": {
            "healthy": str(args.healthy.resolve()),
            "damaged": str(args.damaged.resolve()),
            "pretrain": str(args.pretrain.resolve()) if args.pretrain else None,
            "output": str(output), "device": str(device),
            "smoke": args.smoke, "plots": args.plots,
            "torch_version": torch.__version__,
            "rows": {"healthy": len(healthy), "damaged": len(damaged),
                     "pretrain": len(simulated) if simulated is not None else 0},
        },
        "data_convention": (
            "t,x,y in [0,1]; u normalised; all datasets must share the same "
            "physical and amplitude scales. No units are inferred."
        ),
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Device: {device}; output: {output}")
    if args.smoke:
        print("SMOKE RUN: verifies execution only, not inverse identification accuracy.")
    start = time.perf_counter()
    models = {}
    if simulated is not None:
        pretrain = build_model(simulated, device)
        _fit_healthy(pretrain, args)
        pretrain.save_model(output / "pretrain_model.pth")
        models["pretrain"] = pretrain
        baseline = build_model(healthy, device)
        baseline.net.load_state_dict(pretrain.net.state_dict())
        _fit_healthy(baseline, args, finetune=True)
    else:
        print("No --pretrain: fitting a healthy baseline directly.")
        baseline = build_model(healthy, device)
        _fit_healthy(baseline, args)
    baseline.save_model(output / "stage1_model.pth")
    models["stage1"] = baseline

    joint = build_model(damaged, device)
    joint.net.load_state_dict(baseline.net.state_dict())
    joint.init_damage_params()
    joint.set_stage(2)
    joint.w_pde, joint.w_data = C.W_PDE, C.W_DATA_S2
    joint.w_reg_r, joint.w_reg_g = C.W_REG_R_S2, C.W_REG_G_S2
    joint.train(C.S2_EPOCHS, _batch_size(args), stage=2, joint_training=True)
    joint.save_model(output / "stage2_model.pth")
    models["stage2"] = joint

    final = build_model(damaged, device)
    final.load_model(output / "stage2_model.pth")
    final.set_stage(3)
    final.w_pde, final.w_data = C.W_PDE, C.W_DATA_S3
    final.w_reg_r, final.w_reg_g = C.W_REG_R_S3, C.W_REG_G_S3
    final.train(C.S3_EPOCHS, _batch_size(args), stage=3, start_epoch=C.S2_EPOCHS)
    final.save_model(output / "stage3_final_model.pth")
    final.save_damage_params(output / "damage_params_final.npz")
    models["stage3"] = final
    metrics = final.evaluate()
    metrics.update(
        smoke=args.smoke, elapsed_seconds=time.perf_counter() - start,
        evaluation_scope="training observations only; no held-out accuracy claim",
        transfer_pretraining_used=simulated is not None,
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    if args.plots:
        _save_plots(models, final, output)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


if __name__ == "__main__":
    main()
