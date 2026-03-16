"""
main.py  —  Kirchhoff-Love plate damage identification via PINN
Three-stage training:
  Stage 1 : fit healthy-plate network  (Adam + L-BFGS)
  Stage 2 : joint network + damage optimisation  (Adam, damped network lr)
  Stage 3 : damage-parameter refinement  (Adam)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import glob

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import config as C
from pinn import PINN
from sampler import lhs_points

try:
    from plot import (plot_loss_curve, plot_damage_map,
                      plot_damage_parameter_evolution, plot_damage_position_trajectory,
                      plot_total_loss_curve, plot_lbfgs_training,
                      plot_stage_total_loss, plot_pinn_gradient_flow,
                      plot_damage_spatial_gradient_field_flow)
except ImportError:
    def _noop(*a, **kw): pass
    (plot_loss_curve, plot_damage_map, plot_damage_parameter_evolution,
     plot_damage_position_trajectory, plot_total_loss_curve,
     plot_lbfgs_training, plot_stage_total_loss, plot_pinn_gradient_flow,
     plot_damage_spatial_gradient_field_flow) = (_noop,)*9


# ── Data utilities ────────────────────────────────────────────────────────────

def load_csv_folder(folder):
    frames = []
    for f in os.listdir(folder):
        if f.endswith('.csv'):
            try:
                frames.append(pd.read_csv(os.path.join(folder, f)))
            except Exception as e:
                print(f"  Warning: could not read {f}: {e}")
    if not frames:
        raise RuntimeError(f"No CSV files found in {folder}")
    return pd.concat(frames, ignore_index=True)


def uniform_grid_sample(df, grid_size=C.GRID_SIZE,
                         n_time=C.N_TIME_POINTS, tol=C.SAMPLING_TOL):
    """
    Down-sample df to a uniform (grid_size × grid_size) spatial grid,
    taking up to n_time evenly spaced temporal snapshots per location.
    """
    xs = np.linspace(df['x'].min(), df['x'].max(), grid_size)
    ys = np.linspace(df['y'].min(), df['y'].max(), grid_size)
    Xg, Yg = np.meshgrid(xs, ys)
    grid = np.column_stack([Xg.ravel(), Yg.ravel()])

    parts, found = [], 0
    for xv, yv in grid:
        match = df[(np.abs(df['x'] - xv) < tol) & (np.abs(df['y'] - yv) < tol)]
        if len(match) == 0: continue
        xm, ym = match.iloc[0]['x'], match.iloc[0]['y']
        rows = df[(np.abs(df['x'] - xm) < tol) & (np.abs(df['y'] - ym) < tol)]
        if len(rows) >= n_time:
            idx = np.linspace(0, len(rows)-1, n_time, dtype=int)
            rows = rows.iloc[idx]
        parts.append(rows);  found += 1

    if found == 0:
        print("Warning: no spatial grid matches — falling back to random sample.")
        return df.sample(min(1000, len(df)), random_state=C.R_SEED)

    result = pd.concat(parts, ignore_index=True)
    print(f"  Grid sample: {found}/{len(grid)} locations, {len(result)} total points")
    return result


def df_to_tensors(df, device):
    def _t(col): return torch.tensor(df[col].values.reshape(-1,1), dtype=torch.float64)
    return _t('t').to(device), _t('x').to(device), _t('y').to(device), _t('u').to(device)


def pde_tensors(N, device, batch=C.PDE_LHS_BATCH):
    t, x, y = lhs_points(N, batch)
    def _t(a): return torch.tensor(a, dtype=torch.float64).to(device)
    return _t(t), _t(x), _t(y)


def batch_size_for(n_data, stage):
    if   stage == 1: return 2048 if n_data >= 20000 else (1024 if n_data >= 10000 else 512)
    elif stage == 2: return 1024 if n_data >= 20000 else (512  if n_data >= 10000 else 256)
    else:            return 512  if n_data >= 20000 else (256  if n_data >= 10000 else 128)


def gpu_memory_report():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak  = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  GPU: allocated {alloc:.2f} GB, peak {peak:.2f} GB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Kirchhoff-Love Plate Damage Identification — PINN (PyTorch)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))

    # ── Stage 1: healthy network ──────────────────────────────────────────────
    print("\n── Stage 1: healthy plate ──")
    df_h = load_csv_folder(C.DATA_HEALTHY)
    df_h = uniform_grid_sample(df_h)
    t_h, x_h, y_h, u_h = df_to_tensors(df_h, device)
    t_p1, x_p1, y_p1   = pde_tensors(C.N_PDE, device)

    m1 = PINN(t_h, x_h, y_h, u_h, t_p1, x_p1, y_p1, device=device)
    m1.set_stage(1)
    m1.w_data = C.W_DATA_S1

    t0 = time.time()
    m1.train(C.S1_ADAM_EPOCHS, batch_size_for(len(t_h), 1), stage=1, start_epoch=0)
    lbfgs_loss = m1.train_lbfgs(C.S1_LBFGS_EPOCHS)
    print(f"Stage 1 done in {time.time()-t0:.1f}s | L-BFGS final {lbfgs_loss:.3e}")
    gpu_memory_report()

    plot_lbfgs_training(m1, 1)
    plot_stage_total_loss(m1, 1)
    m1.save_model("stage1_model.pth")

    # ── Stage 2: joint optimisation ───────────────────────────────────────────
    print("\n── Stage 2: joint optimisation ──")
    df_d = load_csv_folder(C.DATA_DAMAGE)
    df_d = uniform_grid_sample(df_d)
    t_d, x_d, y_d, u_d = df_to_tensors(df_d, device)
    t_p2, x_p2, y_p2   = pde_tensors(C.N_PDE, device)

    m2 = PINN(t_d, x_d, y_d, u_d, t_p2, x_p2, y_p2, device=device)
    ck1 = torch.load("stage1_model.pth", map_location=device)
    m2.net.load_state_dict(ck1['net'])

    m2.init_damage_params()
    m2.set_stage(2)
    m2.w_data   = C.W_DATA_S2
    m2.w_pde    = 100.0
    m2.w_reg_r  = C.W_REG_R_S2
    m2.w_reg_g  = C.W_REG_G_S2

    plot_damage_map(m2, "Stage 2 Initial", "(Before Training)", stage=2,
                    save_path="initial_damage.png")

    t0 = time.time()
    m2.train(C.S2_EPOCHS, batch_size_for(len(t_d), 2),
             stage=2, start_epoch=0, joint_training=True)
    print(f"Stage 2 done in {time.time()-t0:.1f}s")
    gpu_memory_report()

    plot_stage_total_loss(m2, 2)
    plot_pinn_gradient_flow(m2, stage=2, save_path="stage2_grad_flow.png")
    plot_damage_spatial_gradient_field_flow(m2, stage=2,
                                            save_path="stage2_dmg_grad_field.png")
    m2.save_model("stage2_model.pth")
    m2.save_damage_params("damage_params_stage2.npz")

    # ── Stage 3: refinement ───────────────────────────────────────────────────
    print("\n── Stage 3: refinement ──")
    m3 = PINN(t_d, x_d, y_d, u_d, t_p2, x_p2, y_p2, device=device)
    m3.load_model("stage2_model.pth")
    m3.set_stage(3)
    m3.w_data  = C.W_DATA_S3
    m3.w_pde   = 100.0
    m3.w_reg_r = C.W_REG_R_S3
    m3.w_reg_g = C.W_REG_G_S3

    start_s3 = (m3.damage_ep_hist[-1] if m3.damage_ep_hist
                else (m3.ep_log[-1] if m3.ep_log else C.S2_EPOCHS))

    t0 = time.time()
    m3.train(C.S3_EPOCHS, batch_size_for(len(t_d), 3),
             stage=3, start_epoch=start_s3)
    print(f"Stage 3 done in {time.time()-t0:.1f}s")
    gpu_memory_report()

    plot_stage_total_loss(m3, 3)
    plot_pinn_gradient_flow(m3, stage=3, save_path="stage3_grad_flow.png")
    plot_total_loss_curve(m1, m2, m3)
    plot_damage_parameter_evolution(m3, start_epoch_stage2=0)
    plot_damage_position_trajectory(m3, start_epoch_stage2=0,
                                    save_path="damage_trajectory.png")
    plot_damage_map(m3, "Final", f"(Stage 3)", stage=3,
                    save_path="damage_map_final.png")
    plot_damage_spatial_gradient_field_flow(m3, stage=3,
                                            save_path="stage3_dmg_grad_field.png")
    m3.save_model("stage3_final_model.pth")

    # ── Evaluation ────────────────────────────────────────────────────────────
    res = m3.evaluate()
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  RMSE (data)          = {res['rmse_data']:.3e}")
    print(f"  Main damage α        = {res['main_alpha']:.4f}")
    print(f"  Main damage position = ({res['main_x']:.4f}, {res['main_y']:.4f})  [normalised]")
    print(f"  Main damage radius   = {res['main_r_mm']:.2f} mm")
    if 'final_total_loss' in res:
        print(f"  L_total (final)      = {res['final_total_loss']:.3e}")
        print(f"  L_pde   (final)      = {res['final_pde_loss']:.3e}")
        print(f"  L_data  (final)      = {res['final_data_loss']:.3e}")
    print("=" * 70)

    # Optional: generate evolution GIF
    try:
        import imageio.v3 as iio
        pngs = sorted(glob.glob("./damage_evolution/damage_*.png"))
        if pngs:
            frames = [iio.imread(p) for p in pngs]
            iio.imwrite("./damage_evolution/evolution.gif", frames, fps=5, loop=0)
            print("GIF saved → ./damage_evolution/evolution.gif")
    except Exception:
        pass


if __name__ == "__main__":
    main()
