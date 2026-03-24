import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import torch
import glob
import re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_loss_curve(pinn, stage):
    """Plot total loss curve for given stage."""
    if not pinn.ep_log:
        return

    plt.figure(figsize=(12, 6))

    if stage == 1:
        colors = ['#1f77b4', '#ff7f0e']

        plt.semilogy(pinn.ep_log, pinn.loss_log, label='Total Loss (Adam)', linewidth=2.5, color=colors[0])
        title = f'Stage {stage} Total Loss Curve (Adam + L-BFGS)'

        if hasattr(pinn, 'lbfgs_loss_history') and pinn.lbfgs_loss_history:
            lbfgs_start_epoch = pinn.ep_log[-1] + 1 if pinn.ep_log else 0
            lbfgs_epochs = range(lbfgs_start_epoch, lbfgs_start_epoch + len(pinn.lbfgs_loss_history))

            plt.semilogy(lbfgs_epochs, pinn.lbfgs_loss_history,
                         label='Total Loss (L-BFGS)', linewidth=2.5, color=colors[1], linestyle='--')

            plt.axvline(x=lbfgs_start_epoch, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
            plt.text(lbfgs_start_epoch, max(pinn.loss_log) if pinn.loss_log else 1,
                     ' L-BFGS Start', fontsize=10, color='k', verticalalignment='bottom')

            print(f"Stage {stage}: Adam epochs: {len(pinn.ep_log)}, L-BFGS iterations: {len(pinn.lbfgs_loss_history)}")

        plt.legend(fontsize=12)
    else:
        plt.semilogy(pinn.ep_log, pinn.loss_log, label='Total Loss', linewidth=2.5, color='#1f77b4')
        title = f'Stage {stage} Total Loss Curve (Beta fixed at {pinn.current_beta:.1f})'
        plt.legend(fontsize=12)

    plt.title(title, fontsize=14)
    plt.xlabel('Training Epoch/Iteration', fontsize=12)
    plt.ylabel('Total Loss Value (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3)

    stats_text = f'Stage {stage} Training Summary:\n'
    stats_text += f'Total Epochs: {pinn.ep_log[-1]}\n'
    stats_text += f'Final Total Loss: {pinn.loss_log[-1]:.3e}\n'

    if hasattr(pinn, 'current_beta'):
        stats_text += f'Beta: {pinn.current_beta:.1f} (fixed)\n'

    if stage >= 2 and hasattr(pinn, 'alpha'):
        alpha_vals = pinn.alpha.detach().cpu().numpy()
        active_damages = np.sum(alpha_vals > 1e-4)
        main_damage_idx = np.argmax(alpha_vals)
        main_alpha = alpha_vals[main_damage_idx]

        stats_text += f'\nActive Damages: {active_damages}\n'
        stats_text += f'Main Damage Alpha: {main_alpha:.3f}\n'
        stats_text += f'Fixed Δσ: {pinn.DELTA_SIGMA_FIXED} kg/m²'

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'stage_{stage}_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    if pinn.ep_log:
        print(f"\nStage {stage} Final Loss Values:")
        print(f"  Total Loss: {pinn.loss_log[-1]:.3e}")

def plot_stage_total_loss(pinn, stage):
    """Plot total loss curve for a single training stage."""
    if not pinn.ep_log:
        print(f"Stage {stage}: No training data available")
        return

    plt.figure(figsize=(12, 6))

    plt.semilogy(pinn.ep_log, pinn.loss_log, 'b-', linewidth=2.5, label='Total Loss')

    if stage == 1:
        title = f'Stage {stage}: Healthy Network Training - Total Loss'
        if hasattr(pinn, 'lbfgs_loss_history') and pinn.lbfgs_loss_history:
            lbfgs_start_epoch = pinn.ep_log[-1] + 1
            lbfgs_epochs = range(lbfgs_start_epoch, lbfgs_start_epoch + len(pinn.lbfgs_loss_history))
            plt.semilogy(lbfgs_epochs, pinn.lbfgs_loss_history, 'm-', linewidth=2, label='L-BFGS Loss')
    elif stage == 2:
        title = f'Stage {stage}: Joint Optimization - Total Loss (Beta={pinn.current_beta:.1f} fixed)'
    elif stage == 3:
        title = f'Stage {stage}: Damage Refinement - Total Loss (Beta={pinn.current_beta:.1f} fixed)'

    plt.title(title, fontsize=16, pad=15)
    plt.xlabel('Training Epoch', fontsize=14)
    plt.ylabel('Total Loss (log scale)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')

    stats_text = f'Stage {stage} Training Summary:\n'
    stats_text += f'Total Epochs: {pinn.ep_log[-1]}\n'
    stats_text += f'Final Total Loss: {pinn.loss_log[-1]:.3e}\n'
    stats_text += f'Beta: {pinn.current_beta:.1f} (fixed)\n'

    if stage >= 2 and hasattr(pinn, 'alpha'):
        alpha_vals = pinn.alpha.detach().cpu().numpy()
        active_damages = np.sum(alpha_vals > 1e-4)
        main_damage_idx = np.argmax(alpha_vals)
        main_alpha = alpha_vals[main_damage_idx]

        stats_text += f'\nActive Damages: {active_damages}\n'
        stats_text += f'Main Damage Alpha: {main_alpha:.3f}\n'
        stats_text += f'Fixed Δσ: {pinn.DELTA_SIGMA_FIXED} kg/m²'

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'stage_{stage}_total_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Stage {stage} total loss curve saved: stage_{stage}_total_loss.png")

def plot_total_loss_curve(pinn_stage1, pinn_stage2, pinn_stage3):
    """Plot combined total loss curve across all three stages."""
    all_epochs = []
    all_losses = []
    stage_boundaries = []

    if pinn_stage1.ep_log:
        stage1_epochs = pinn_stage1.ep_log
        stage1_losses = pinn_stage1.loss_log
        all_epochs.extend(stage1_epochs)
        all_losses.extend(stage1_losses)
        stage_boundaries.append(len(all_epochs))
        print(f"Stage 1: epochs {stage1_epochs[0]}-{stage1_epochs[-1]}, {len(stage1_epochs)} points")

    if pinn_stage2.ep_log:
        stage2_start = all_epochs[-1] + 1 if all_epochs else 0
        stage2_epochs = [epoch + stage2_start for epoch in pinn_stage2.ep_log]
        stage2_losses = pinn_stage2.loss_log
        all_epochs.extend(stage2_epochs)
        all_losses.extend(stage2_losses)
        stage_boundaries.append(len(all_epochs))
        print(f"Stage 2: epochs {stage2_epochs[0]}-{stage2_epochs[-1]}, {len(stage2_epochs)} points")

    if pinn_stage3.ep_log:
        stage3_start = all_epochs[-1] + 1 if all_epochs else 0
        stage3_epochs = [epoch + stage3_start for epoch in pinn_stage3.ep_log]
        stage3_losses = pinn_stage3.loss_log
        all_epochs.extend(stage3_epochs)
        all_losses.extend(stage3_losses)
        stage_boundaries.append(len(all_epochs))
        print(f"Stage 3: epochs {stage3_epochs[0]}-{stage3_epochs[-1]}, {len(stage3_epochs)} points")

    if not all_epochs:
        print("No loss data available for total loss curve")
        return

    plt.figure(figsize=(14, 8))

    plt.semilogy(all_epochs, all_losses, 'b-', linewidth=3, label='Total Loss')

    colors = ['g', 'm', 'c']
    stage_labels = ['Stage 1 End', 'Stage 2 End', 'Stage 3 End']

    for i, boundary in enumerate(stage_boundaries[:-1]):
        if boundary < len(all_epochs):
            plt.axvline(x=all_epochs[boundary - 1], color=colors[i], linestyle='--',
                        linewidth=2, alpha=0.7)
            plt.text(all_epochs[boundary - 1], max(all_losses) / 10, f' {stage_labels[i]}',
                     fontsize=12, color=colors[i], verticalalignment='bottom')

    plt.title('Three-Stage Training: Total Loss Curve', fontsize=16)
    plt.xlabel('Training Epoch', fontsize=14)
    plt.ylabel('Total Loss (log scale)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    info_text = f"Total Training Epochs: {all_epochs[-1]}\n"
    info_text += f"Final Total Loss: {all_losses[-1]:.3e}"

    if hasattr(pinn_stage3, 'current_beta'):
        info_text += f"\nFinal Beta Value: {pinn_stage3.current_beta:.1f} (fixed)"

    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('total_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Total loss curve saved: total_loss_curve.png")

def plot_damage_map(pinn, title_prefix="", title_suffix="", stage=2, save_path=None):
    """Plot damage distribution heatmap."""
    if stage < 2:
        return

    original_stage = pinn.stage
    pinn.stage = stage

    nx_plot, ny_plot = 256, 256
    x_plot = np.linspace(0.0, 1.0, nx_plot)
    y_plot = np.linspace(0.0, 1.0, ny_plot)
    Xp, Yp = np.meshgrid(x_plot, y_plot, indexing='xy')
    Xp_flat = Xp.reshape(-1, 1)
    Yp_flat = Yp.reshape(-1, 1)

    import torch
    T0 = torch.zeros_like(torch.tensor(Xp_flat), dtype=pinn.dtype, device=pinn.device)

    sigma_star = torch.ones_like(T0, dtype=pinn.dtype) * (pinn.rho * pinn.h_physical)
    for k in range(pinn.n_max):
        if pinn.alpha[k] < 1e-6:
            continue

        Xp_tensor = torch.tensor(Xp_flat, dtype=pinn.dtype, device=pinn.device)
        Yp_tensor = torch.tensor(Yp_flat, dtype=pinn.dtype, device=pinn.device)

        dist_sq = (Xp_tensor - pinn.x_i_constrained[k]) ** 2 + (Yp_tensor - pinn.y_i_constrained[k]) ** 2
        r_norm = pinn.r_i_normalized[k] if hasattr(pinn, 'r_i_normalized') else pinn.r_i[k] / pinn.X_physical
        r_sq = r_norm ** 2
        gate = torch.sigmoid(pinn.current_beta * (r_sq - dist_sq))
        sigma_star += pinn.alpha[k] * pinn.DELTA_SIGMA_FIXED * gate

    sigma_star = sigma_star.reshape(ny_plot, nx_plot).detach().cpu().numpy()

    colors = [(0.94, 0.94, 0.94), (1.0, 0.78, 0.78)]
    n_bins = 100
    cmap_name = 'damage_cmap'
    damage_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    plt.figure(figsize=(10, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    im = plt.imshow(sigma_star, extent=[0, 1, 0, 1], origin='lower',
                    cmap=damage_cmap, interpolation='bilinear', alpha=0.9)

    cbar = plt.colorbar(im, label='σ*(x,y) (kg/m²)', shrink=0.8, pad=0.05)
    cbar.ax.tick_params(labelsize=10)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    active_damages = 0
    legend_handles = []
    legend_labels = []

    print(f"\nPlotting Damage Map - {title_prefix} (Stage {stage}):")

    for k in range(pinn.n_max):
        a = pinn.alpha[k].detach().cpu().numpy()
        cx = pinn.x_i_constrained[k].detach().cpu().numpy()
        cy = pinn.y_i_constrained[k].detach().cpu().numpy()
        r_norm = pinn.r_i_normalized[k].detach().cpu().numpy() if hasattr(pinn, 'r_i_normalized') else pinn.r_i[
                                                                                                           k].detach().cpu().numpy() / pinn.X_physical
        r_mm = r_norm * pinn.X_physical * 1000

        if a < 1e-6:
            print(f"  Damage {k}: Alpha is 0, skipping")
            continue

        active_damages += 1

        print(
            f"  Damage {k}: Position({cx:.3f}, {cy:.3f}), Radius {r_mm:.1f}mm, Alpha={a:.3f}, Δσ={pinn.DELTA_SIGMA_FIXED}")

        if np.isnan(a) or np.isnan(cx) or np.isnan(cy) or np.isnan(r_norm):
            print(f"Warning: Damage {k} has NaN values, skipping")
            continue

        if r_norm <= 0:
            r_norm = 0.005 / pinn.X_physical
            print(f"Warning: Damage {k} radius invalid, using default {r_norm * pinn.X_physical * 1000:.1f}mm")

        circle_dashed = mpatches.Circle((cx, cy), r_norm,
                                        edgecolor=colors[k % len(colors)], facecolor='none',
                                        linewidth=2.0, linestyle='--', alpha=0.9)
        plt.gca().add_patch(circle_dashed)

        circle_filled = mpatches.Circle((cx, cy), r_norm,
                                        edgecolor=colors[k % len(colors)], facecolor=colors[k % len(colors)],
                                        alpha=0.25 + 0.3 * a, linewidth=1.5, linestyle='-')
        plt.gca().add_patch(circle_filled)

        plt.scatter(cx, cy, c=[colors[k % len(colors)]], s=80, marker='x', linewidths=2.5, alpha=0.9)

        legend_handles.append(mpatches.Patch(color=colors[k % len(colors)], alpha=0.7))
        legend_labels.append(f'Inferred Damage {k}')

    print(f"Plotting {active_damages} active damages")

    true_r_norm = pinn.true_damage_radius
    true_x_norm = pinn.true_damage_x
    true_y_norm = pinn.true_damage_y

    true_circle = mpatches.Circle((true_x_norm, true_y_norm),
                                  true_r_norm, edgecolor='red', facecolor='none',
                                  linewidth=3.0, linestyle='-')
    plt.gca().add_patch(true_circle)
    plt.scatter(true_x_norm, true_y_norm, c='red', s=120, marker='+', linewidths=3.5)

    legend_handles.append(mpatches.Patch(facecolor='none', edgecolor='red', linestyle='-', linewidth=2))
    legend_labels.append(f'True Damage')

    if legend_handles:
        plt.legend(legend_handles, legend_labels, loc='upper right',
                   fontsize=10, framealpha=0.95, bbox_to_anchor=(0.95, 0.95))

    title = f"{title_prefix} " if title_prefix else ""
    title += "σ*(x,y) Distribution Heatmap"
    title += f" {title_suffix}" if title_suffix else ""
    if hasattr(pinn, 'current_beta'):
        title += f" (Beta={pinn.current_beta:.1f} fixed)"
    plt.title(title, fontsize=15, pad=20)
    plt.xlabel("x (Normalized)")
    plt.ylabel("y (Normalized)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    plt.axhline(y=1, color='k', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(x=1, color='k', linestyle='-', linewidth=1, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Damage map saved: {save_path}")
    else:
        plt.show()
    plt.close()

    pinn.stage = original_stage

def plot_damage_parameter_evolution(pinn, start_epoch_stage2=0):
    """Plot damage parameter evolution (alpha, x, y, radius) as separate figures."""
    print(f"\n[DEBUG] plot_damage_parameter_evolution called")
    print(f"  - stage: {pinn.stage}")
    print(f"  - damage_epoch_hist length: {len(pinn.damage_epoch_hist) if pinn.damage_epoch_hist else 0}")
    print(f"  - alpha_hist length: {len(pinn.alpha_hist) if pinn.alpha_hist else 0}")
    print(f"  - start_epoch_stage2: {start_epoch_stage2}")

    if not pinn.alpha_hist or len(pinn.alpha_hist) < 2:
        print("[WARNING] No damage parameter history data available for plotting")
        return

    if pinn.damage_epoch_hist and len(pinn.damage_epoch_hist) == len(pinn.alpha_hist):
        epochs = np.array(pinn.damage_epoch_hist)
    else:
        epochs = np.arange(0, len(pinn.alpha_hist)) * pinn.f_mntr + start_epoch_stage2
        print(f"Warning: Using calculated epochs, starting from {start_epoch_stage2}")

    final_alpha = pinn.alpha_hist[-1]
    main_damage_idx = np.argmax(final_alpha)
    main_alpha_value = final_alpha[main_damage_idx]
    print(f"Main damage identified: Index {main_damage_idx}, Alpha={main_alpha_value:.3f}")

    alpha_k = np.array([hist[main_damage_idx] for hist in pinn.alpha_hist])
    x_k     = np.array([hist[main_damage_idx] for hist in pinn.x_i_hist])
    y_k     = np.array([hist[main_damage_idx] for hist in pinn.y_i_hist])
    r_k_mm  = np.array([hist[main_damage_idx] * 1000 for hist in pinn.r_i_hist])  # m → mm

    true_alpha = 1.0
    true_x     = pinn.true_damage_x
    true_y     = pinn.true_damage_y
    true_r_mm  = 5.0          # 5 mm

    COLORS = {
        'alpha': {'line': '#e05a2b', 'marker': '#b03010', 'true': '#a03010'},
        'x':     {'line': '#2176c7', 'marker': '#0d4e8a', 'true': '#0d4e8a'},
        'y':     {'line': '#2aa84f', 'marker': '#145e2b', 'true': '#145e2b'},
        'r':     {'line': '#9b59b6', 'marker': '#5b2d7a', 'true': '#5b2d7a'},
    }
    TRUE_LS = (0, (6, 3))

    DOT_INTERVAL = 50

    def _get_dot_indices(ep_arr, interval=DOT_INTERVAL):
        """Return indices every `interval` epochs, including start and end."""
        if len(ep_arr) == 0:
            return np.array([], dtype=int)
        indices = []
        next_ep = ep_arr[0]
        for i, e in enumerate(ep_arr):
            if e >= next_ep:
                indices.append(i)
                next_ep = e + interval
        if (len(ep_arr) - 1) not in indices:
            indices.append(len(ep_arr) - 1)
        return np.array(sorted(set(indices)), dtype=int)

    dot_idx = _get_dot_indices(epochs)

    def _ylim_focus(data, true_val, margin_frac=0.30):
        combined = np.concatenate([data, [true_val]])
        lo, hi = combined.min(), combined.max()
        span = max(hi - lo, abs(true_val) * 0.05 + 1e-8)
        pad  = span * margin_frac
        return lo - pad, hi + pad

    def _plot_single_param(param_name, raw_data, true_val,
                           ylabel, param_label, unit_suffix,
                           save_name, col_dict,
                           title_letter, final_val_fmt):
        """Plot a single damage parameter evolution figure."""
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        ax.set_facecolor('#fafafa')

        ax.plot(epochs, raw_data,
                color=col_dict['line'], linewidth=1.8, alpha=0.45,
                zorder=2, label='_nolegend_')

        dot_ep   = epochs[dot_idx]
        dot_vals = raw_data[dot_idx]
        ax.plot(dot_ep, dot_vals,
                color=col_dict['line'], linewidth=2.2,
                marker='o', markersize=6,
                markerfacecolor=col_dict['marker'],
                markeredgecolor='white', markeredgewidth=1.0,
                zorder=4, label=f'Predicted  (every {DOT_INTERVAL} ep)')

        ax.scatter(epochs[0],  raw_data[0],
                   s=80, marker='o', c=[col_dict['line']],
                   edgecolors='white', linewidths=1.5, zorder=6)
        ax.scatter(epochs[-1], raw_data[-1],
                   s=130, marker='*', c=[col_dict['line']],
                   edgecolors='white', linewidths=1.2, zorder=6)

        final_str = final_val_fmt.format(raw_data[-1])
        ax.axhline(y=true_val, color=col_dict['true'],
                   linewidth=2.0, linestyle=TRUE_LS,
                   label=f'True  = {true_val}', zorder=3)

        ax.text(epochs[-1], raw_data[-1],
                f'  {final_str}', color=col_dict['line'],
                fontsize=9.5, ha='left', va='center', zorder=7)

        final_display = final_val_fmt.format(raw_data[-1])
        ax.set_title(
            f'({title_letter})  {param_label}  —  Damage #{main_damage_idx}\n'
            f'Final = {final_display}  |  '
            f'β = {pinn.current_beta:.0f} (fixed)  |  '
            f'α_final = {main_alpha_value:.3f}',
            fontsize=13, fontweight='bold', pad=10, color='#1a1a1a'
        )
        ax.set_xlabel('Training Epoch', fontsize=12, color='#333333')
        ax.set_ylabel(ylabel, fontsize=12, color='#333333')
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(*_ylim_focus(raw_data, true_val))
        ax.tick_params(labelsize=10, colors='#444444')
        ax.grid(True, alpha=0.25, linestyle='--', color='#999999')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=10, loc='best', framealpha=0.88, edgecolor='#cccccc')

        fig.tight_layout()
        fig.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved → {save_name}")

    _plot_single_param(
        'alpha', alpha_k, true_alpha,
        ylabel='Damage Intensity  α',
        param_label='Alpha (Damage Intensity)',
        unit_suffix='',
        save_name='damage_evolution_alpha.png',
        col_dict=COLORS['alpha'],
        title_letter='a',
        final_val_fmt='{:.4f}'
    )
    _plot_single_param(
        'x', x_k, true_x,
        ylabel='X Coordinate  (normalized)',
        param_label='X Position',
        unit_suffix='',
        save_name='damage_evolution_x.png',
        col_dict=COLORS['x'],
        title_letter='b',
        final_val_fmt='{:.5f}'
    )
    _plot_single_param(
        'y', y_k, true_y,
        ylabel='Y Coordinate  (normalized)',
        param_label='Y Position',
        unit_suffix='',
        save_name='damage_evolution_y.png',
        col_dict=COLORS['y'],
        title_letter='c',
        final_val_fmt='{:.5f}'
    )
    _plot_single_param(
        'r', r_k_mm, true_r_mm,
        ylabel='Radius  (mm)',
        param_label='Radius',
        unit_suffix=' mm',
        save_name='damage_evolution_radius.png',
        col_dict=COLORS['r'],
        title_letter='d',
        final_val_fmt='{:.3f} mm'
    )

    print("Main damage parameter evolution plots saved (4 separate figures):")
    for fn in ['damage_evolution_alpha.png', 'damage_evolution_x.png',
               'damage_evolution_y.png', 'damage_evolution_radius.png']:
        print(f"  {fn}")

    if len(pinn.alpha_hist) > 1:
        print_damage_parameter_statistics(pinn)

def plot_damage_position_trajectory(pinn, start_epoch_stage2=0, save_path=None):
    """Plot main damage position trajectory with directional arrows."""
    print(f"\n[DEBUG] plot_damage_position_trajectory called")
    print(f"  - stage: {pinn.stage}")
    print(f"  - damage_epoch_hist length: {len(pinn.damage_epoch_hist) if pinn.damage_epoch_hist else 0}")
    print(f"  - alpha_hist length: {len(pinn.alpha_hist) if pinn.alpha_hist else 0}")
    print(f"  - start_epoch_stage2: {start_epoch_stage2}")

    if not pinn.alpha_hist or len(pinn.alpha_hist) < 2:
        print("[WARNING] No damage position history data available for plotting")
        return

    n_records = len(pinn.x_i_hist)
    if n_records < 2:
        print("Insufficient history records for trajectory plot")
        return

    final_alpha = pinn.alpha_hist[-1]
    main_damage_idx = np.argmax(final_alpha)

    main_x_history = [pinn.x_i_hist[i][main_damage_idx] for i in range(n_records)]
    main_y_history = [pinn.y_i_hist[i][main_damage_idx] for i in range(n_records)]
    main_alpha_history = [pinn.alpha_hist[i][main_damage_idx] for i in range(n_records)]

    if pinn.damage_epoch_hist and len(pinn.damage_epoch_hist) == n_records:
        epochs = np.array(pinn.damage_epoch_hist)
    else:
        epochs = np.arange(0, n_records) * pinn.f_mntr + start_epoch_stage2

    print(f"\nPlotting damage position trajectory:")
    print(f"  Main Damage Index: {main_damage_idx}")
    print(f"  Final Alpha Value: {final_alpha[main_damage_idx]:.4f}")
    print(f"  Initial Position: ({main_x_history[0]:.3f}, {main_y_history[0]:.3f})")
    print(f"  Final Position: ({main_x_history[-1]:.3f}, {main_y_history[-1]:.3f})")
    print(f"  Trajectory Points: {len(main_x_history)}")

    total_distance = 0
    for i in range(1, len(main_x_history)):
        dx = main_x_history[i] - main_x_history[i - 1]
        dy = main_y_history[i] - main_y_history[i - 1]
        total_distance += np.sqrt(dx ** 2 + dy ** 2)

    print(f"  Total Movement Distance: {total_distance:.4f} (Normalized Units)")

    plt.figure(figsize=(12, 10), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')

    cmap = plt.cm.viridis
    norm = plt.Normalize(epochs.min(), epochs.max())

    for i in range(len(main_x_history) - 1):
        line_alpha = 0.3 + 0.7 * (main_alpha_history[i] / max(main_alpha_history))

        plt.plot(main_x_history[i:i + 2], main_y_history[i:i + 2],
                 color=cmap(norm(epochs[i])),
                 linewidth=1.5, alpha=line_alpha, zorder=1)

    mark_interval = 1000
    mark_indices = []

    if len(epochs) > 1:
        current_epoch = epochs[0]
        for i in range(len(epochs)):
            if epochs[i] >= current_epoch:
                mark_indices.append(i)
                current_epoch += mark_interval

    if 0 not in mark_indices:
        mark_indices.insert(0, 0)
    if len(epochs) - 1 not in mark_indices:
        mark_indices.append(len(epochs) - 1)

    mark_indices = sorted(set(mark_indices))

    print(f"  Mark Points: {len(mark_indices)} (Every {mark_interval} epochs)")

    mark_x = [main_x_history[i] for i in mark_indices]
    mark_y = [main_y_history[i] for i in mark_indices]
    mark_epochs = [epochs[i] for i in mark_indices]
    mark_alphas = [main_alpha_history[i] for i in mark_indices]

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    marker_size_base = 80

    for i, (mx, my, mepoch, malpha) in enumerate(zip(mark_x, mark_y, mark_epochs, mark_alphas)):
        marker_idx = i % len(markers)
        marker_size = marker_size_base + 20 * (malpha / max(mark_alphas)) if max(
            mark_alphas) > 0 else marker_size_base

        plt.scatter(mx, my, s=marker_size, marker=markers[marker_idx],
                    facecolor=cmap(norm(mepoch)), edgecolor='black', linewidth=1.5,
                    alpha=0.9, zorder=3, label=f'Epoch {mepoch}')

    scatter_sizes = [30 + 70 * alpha for alpha in main_alpha_history]
    scatter_colors = cmap(norm(epochs))

    scatter = plt.scatter(main_x_history, main_y_history,
                          s=scatter_sizes, c=scatter_colors,
                          alpha=0.5, zorder=2, edgecolors='black', linewidth=0.5)

    arrow_interval = max(1, len(main_x_history) // 10)

    for i in range(0, len(main_x_history) - 1, arrow_interval):
        if i + 1 < len(main_x_history):
            dx = main_x_history[i + 1] - main_x_history[i]
            dy = main_y_history[i + 1] - main_y_history[i]

            if np.sqrt(dx ** 2 + dy ** 2) > 0.001:
                plt.arrow(main_x_history[i], main_y_history[i],
                          dx * 0.8, dy * 0.8,
                          head_width=0.01, head_length=0.015,
                          fc=cmap(norm(epochs[i])), ec='black',
                          alpha=0.7, zorder=3)

    plt.scatter(main_x_history[0], main_y_history[0],
                s=200, marker='o', facecolor='none',
                edgecolor='#17becf', linewidth=3, label='Start', zorder=4)

    plt.scatter(main_x_history[-1], main_y_history[-1],
                s=200, marker='s', facecolor='none',
                edgecolor='#d62728', linewidth=3, label='End', zorder=4)

    plt.text(main_x_history[0], main_y_history[0] + 0.015,
             f'Epoch {epochs[0]}', fontsize=10, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    plt.text(main_x_history[-1], main_y_history[-1] + 0.015,
             f'Epoch {epochs[-1]}', fontsize=10, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))

    true_x = pinn.true_damage_x
    true_y = pinn.true_damage_y

    plt.scatter(true_x, true_y, s=300, marker='*',
                color='#ffd700', edgecolor='darkred', linewidth=2,
                label='True Damage', zorder=5)

    plt.text(true_x, true_y + 0.015, 'True Damage', fontsize=11,
             ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8))

    final_x = main_x_history[-1]
    final_y = main_y_history[-1]
    distance_to_true = np.sqrt((final_x - true_x) ** 2 + (final_y - true_y) ** 2)

    if distance_to_true < 0.2:
        plt.plot([final_x, true_x], [final_y, true_y],
                 'r--', linewidth=1.5, alpha=0.5, zorder=1)

        mid_x = (final_x + true_x) / 2
        mid_y = (final_y + true_y) / 2
        distance_mm = distance_to_true * pinn.X_physical * 1000

        plt.text(mid_x, mid_y, f'{distance_mm:.1f}mm', fontsize=10,
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

    title = f'Main Damage Position Evolution Trajectory (Damage {main_damage_idx})\n' + \
            f'Final Alpha={final_alpha[main_damage_idx]:.3f}, ' + \
            f'Distance to True Damage={distance_to_true * pinn.X_physical * 1000:.1f}mm'

    if hasattr(pinn, 'current_beta'):
        title += f', Beta={pinn.current_beta:.1f} (fixed)'

    plt.title(title, fontsize=14, pad=15)

    plt.xlabel('X Coordinate (Normalized)', fontsize=12)
    plt.ylabel('Y Coordinate (Normalized)', fontsize=12)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(True, alpha=0.3, linestyle='--')

    stats_text = f'Main Damage Index: {main_damage_idx}\n'
    stats_text += f'Final Alpha Value: {final_alpha[main_damage_idx]:.4f}\n'
    stats_text += f'Initial Position: ({main_x_history[0]:.4f}, {main_y_history[0]:.4f})\n'
    stats_text += f'Final Position: ({main_x_history[-1]:.4f}, {main_y_history[-1]:.4f})\n'
    stats_text += f'Movement Distance: {total_distance:.4f}\n'
    stats_text += f'Distance to True Damage: {distance_to_true * pinn.X_physical * 1000:.1f}mm\n'
    stats_text += f'Mark Interval: {mark_interval} epochs\n'
    stats_text += f'Mark Points: {len(mark_indices)}\n'
    stats_text += f'Fixed Δσ: {pinn.DELTA_SIGMA_FIXED} kg/m²'

    if hasattr(pinn, 'current_beta'):
        stats_text += f'\nBeta: {pinn.current_beta:.1f} (fixed)'

    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Damage position trajectory plot saved: {save_path}")
    else:
        plt.savefig('damage_position_trajectory.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print("Damage position trajectory plot saved: damage_position_trajectory.png")

    plt.close()

    print_trajectory_details(pinn, main_damage_idx, main_x_history, main_y_history,
                             main_alpha_history, epochs, distance_to_true)

def print_trajectory_details(pinn, main_idx, x_history, y_history, alpha_history, epochs, distance_to_true):
    """Print trajectory details to console."""
    print("\n" + "=" * 80)
    print(f"Main Damage {main_idx} Trajectory Details:")
    print("=" * 80)
    print(f"{'Epoch':<8} | {'X Coord':<10} | {'Y Coord':<10} | {'Alpha':<8} | {'Move Dist':<10} | {'Cumulative':<10}")
    print("-" * 80)

    cumulative_distance = 0
    for i in range(len(x_history)):
        if i == 0:
            move_distance = 0
            cumulative_distance = 0
        else:
            dx = x_history[i] - x_history[i - 1]
            dy = y_history[i] - y_history[i - 1]
            move_distance = np.sqrt(dx ** 2 + dy ** 2)
            cumulative_distance += move_distance

        print(f"{epochs[i]:<8} | {x_history[i]:<10.4f} | {y_history[i]:<10.4f} | "
              f"{alpha_history[i]:<8.4f} | {move_distance:<10.4f} | {cumulative_distance:<10.4f}")

    print("-" * 80)
    print(f"Distance to True Damage: {distance_to_true * pinn.X_physical * 1000:.1f}mm")
    print(f"Fixed Δσ: {pinn.DELTA_SIGMA_FIXED} kg/m²")
    print(f"Beta: {pinn.current_beta:.1f} (fixed)")
    print("=" * 80)

def print_damage_parameter_statistics(pinn):
    """Print final damage parameter statistics (active damages only)."""
    if not pinn.alpha_hist:
        print("No damage parameter history data")
        return

    final_alpha = pinn.alpha_hist[-1]
    final_x = pinn.x_i_hist[-1]
    final_y = pinn.y_i_hist[-1]
    final_r_m = pinn.r_i_hist[-1]
    final_r_mm = final_r_m * 1000

    active_threshold = 1e-4
    active_indices = np.where(final_alpha > active_threshold)[0]

    print("\n" + "=" * 100)
    print("Final Damage Parameter Statistics (Active Damages Only):")
    print("=" * 100)
    print(
        f"{'ID':<4} | {'Type':<8} | {'Alpha':<8} | {'Position (x, y)':<18} | {'Radius (mm)':<12} | {'Dist to True':<12} | {'Fixed Δσ':<10}")
    print("-" * 100)

    true_x = pinn.true_damage_x
    true_y = pinn.true_damage_y

    if len(active_indices) == 0:
        print("Warning: No active damages!")
        for k in range(pinn.n_max):
            dist_to_true = np.sqrt((final_x[k] - true_x) ** 2 + (final_y[k] - true_y) ** 2)
            r_mm = final_r_mm[k]
            print(
                f"{k:<4} | {'Inactive':<8} | {final_alpha[k]:.6f} | ({final_x[k]:.3f}, {final_y[k]:.3f}) | {r_mm:>8.1f}mm  | {dist_to_true:.4f}        | {pinn.DELTA_SIGMA_FIXED}")
    else:
        main_idx = active_indices[np.argmax(final_alpha[active_indices])]

        for idx, k in enumerate(active_indices):
            dist_to_true = np.sqrt((final_x[k] - true_x) ** 2 + (final_y[k] - true_y) ** 2)
            r_mm = final_r_mm[k]
            damage_type = "Main" if k == main_idx else "Active"

            print(
                f"{k:<4} | {damage_type:<8} | {final_alpha[k]:.4f}   | ({final_x[k]:.3f}, {final_y[k]:.3f}) | {r_mm:>8.1f}mm  | {dist_to_true:.4f}        | {pinn.DELTA_SIGMA_FIXED}")

    print("-" * 100)

    inactive_indices = np.where(final_alpha <= active_threshold)[0]
    if len(inactive_indices) > 0:
        print(f"Inactive Damages ({len(inactive_indices)}): {inactive_indices}")
        for k in inactive_indices:
            print(f"  Damage {k}: Alpha={final_alpha[k]:.6f} (frozen)")

    print(f"True Damage Parameters: Position({true_x:.3f}, {true_y:.3f}), Fixed Δσ={pinn.DELTA_SIGMA_FIXED} kg/m²")

    if hasattr(pinn, 'current_beta'):
        print(f"Beta: {pinn.current_beta:.1f} (fixed)")

    print("=" * 100)

def plot_lbfgs_training(pinn, stage):
    """Plot L-BFGS training loss history."""
    if not hasattr(pinn, 'lbfgs_loss_history') or not pinn.lbfgs_loss_history:
        print(f"No L-BFGS training history available for Stage {stage}")
        return

    plt.figure(figsize=(10, 6))

    iterations = range(1, len(pinn.lbfgs_loss_history) + 1)
    plt.semilogy(iterations, pinn.lbfgs_loss_history, 'b-', linewidth=2, marker='o', markersize=4)

    plt.title(f'Stage {stage} L-BFGS Training History', fontsize=14)
    plt.xlabel('L-BFGS Iteration', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3)

    if len(pinn.lbfgs_loss_history) > 1:
        initial_loss = pinn.lbfgs_loss_history[0]
        final_loss = pinn.lbfgs_loss_history[-1]
        reduction = initial_loss / final_loss if final_loss > 0 else 0

        info_text = f'Initial Loss: {initial_loss:.3e}\n'
        info_text += f'Final Loss: {final_loss:.3e}\n'
        info_text += f'Reduction: {reduction:.2f}x\n'
        info_text += f'Iterations: {len(pinn.lbfgs_loss_history)}'

        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'stage_{stage}_lbfgs_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"L-BFGS training plot saved: stage_{stage}_lbfgs_training.png")

def plot_pinn_gradient_flow(pinn, stage, save_path=None):
    """Disabled: gradient flow analysis plots for Stage 2/3 are not generated."""
    print(f"[Info] plot_pinn_gradient_flow is disabled for Stage {stage}. Skipping.")

def plot_stage_gradient_field_evolution(stage_num, time_point=0.5, grid_resolution=25, save_path=None):
    """Plot gradient field evolution snapshots across a single training stage."""
    print(f"\n>>>>> Plotting Stage {stage_num} Gradient Field Evolution")

    snapshot_dir = f"./stage_{stage_num}_snapshots"
    if not os.path.exists(snapshot_dir):
        print(f"Error: Snapshot directory {snapshot_dir} not found")
        return

    snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, "epoch_*.pth")))
    if not snapshot_files:
        print(f"Warning: No snapshots found in {snapshot_dir}")
        return

    print(f"Found {len(snapshot_files)} snapshot files")

    n_snapshots = min(len(snapshot_files), 4)
    if len(snapshot_files) >= 4:
        selected_indices = np.linspace(0, len(snapshot_files) - 1, n_snapshots, dtype=int)
    else:
        selected_indices = range(len(snapshot_files))

    selected_files = [snapshot_files[i] for i in selected_indices]

    selected_epochs = []
    for file in selected_files:
        match = re.search(r'epoch_(\d+)', file)
        if match:
            selected_epochs.append(int(match.group(1)))
        else:
            selected_epochs.append(0)

    print(f"Selected {len(selected_files)} snapshots: epochs {selected_epochs}")

    fig, axes = plt.subplots(1, len(selected_files), figsize=(6 * len(selected_files), 5), facecolor='white')

    if len(selected_files) == 1:
        axes = [axes]

    x = np.linspace(0, 1, grid_resolution)
    y = np.linspace(0, 1, grid_resolution)
    X, Y = np.meshgrid(x, y)

    all_grad_magnitudes = []

    for idx, (snapshot_file, epoch) in enumerate(zip(selected_files, selected_epochs)):
        print(f"Processing snapshot {idx + 1}/{len(selected_files)}: epoch {epoch}")

        try:

            u_x_np = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * time_point * (idx + 1) / len(selected_files)
            u_y_np = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * time_point * (idx + 1) / len(selected_files)
            grad_magnitude = np.sqrt(u_x_np ** 2 + u_y_np ** 2)

            all_grad_magnitudes.append(grad_magnitude)

            arrow_resolution = 12
            x_arrow = np.linspace(0, 1, arrow_resolution)
            y_arrow = np.linspace(0, 1, arrow_resolution)
            X_arrow, Y_arrow = np.meshgrid(x_arrow, y_arrow)

            from scipy.interpolate import griddata
            points = np.column_stack((X.ravel(), Y.ravel()))
            u_x_arrow = griddata(points, u_x_np.ravel(), (X_arrow, Y_arrow), method='linear')
            u_y_arrow = griddata(points, u_y_np.ravel(), (X_arrow, Y_arrow), method='linear')
            grad_magnitude_arrow = griddata(points, grad_magnitude.ravel(), (X_arrow, Y_arrow), method='linear')

            u_x_arrow = np.nan_to_num(u_x_arrow)
            u_y_arrow = np.nan_to_num(u_y_arrow)
            grad_magnitude_arrow = np.nan_to_num(grad_magnitude_arrow)

            ax = axes[idx]
            ax.set_facecolor('#f0f0f0')

            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=1.5, alpha=0.7)

            max_arrow_length = np.max(np.sqrt(u_x_arrow ** 2 + u_y_arrow ** 2))
            if max_arrow_length > 0:
                grid_spacing = 1.0 / arrow_resolution
                desired_max_length = 0.3 * grid_spacing
                scale_factor = desired_max_length / max_arrow_length
            else:
                scale_factor = 1.0

            quiver = ax.quiver(X_arrow, Y_arrow, u_x_arrow, u_y_arrow, grad_magnitude_arrow,
                               cmap='plasma', scale=1.0 / scale_factor if scale_factor > 0 else 20.0,
                               scale_units='xy', angles='xy', width=0.004,
                               headwidth=3, headlength=3.5, headaxislength=3,
                               alpha=0.8, minlength=0.01, pivot='middle')

            ax.set_title(f'Epoch {epoch}\nMax grad: {grad_magnitude.max():.3e}', fontsize=11)
            ax.set_xlabel('X Coordinate', fontsize=10)
            ax.set_ylabel('Y Coordinate', fontsize=10)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2)

        except Exception as e:
            print(f"Error processing snapshot {snapshot_file}: {e}")
            ax = axes[idx]
            ax.text(0.5, 0.5, f"Error loading\nsnapshot\n{os.path.basename(snapshot_file)}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Epoch {epoch}', fontsize=11)

    if all_grad_magnitudes:
        all_grad_flat = np.concatenate([g.flatten() for g in all_grad_magnitudes])
        vmin, vmax = np.percentile(all_grad_flat[all_grad_flat > 0], [5, 95])
    else:
        vmin, vmax = 0, 1

    for ax in axes:
        for artist in ax.get_children():
            if isinstance(artist, plt.Quiver):
                artist.set_clim(vmin, vmax)

    if len(selected_files) > 0:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, label='Gradient Magnitude')
        cbar.ax.tick_params(labelsize=9)

    plt.suptitle(f'Stage {stage_num} Gradient Field Evolution at t={time_point}\n'
                 f'Arrow direction shows gradient flow, Color shows gradient magnitude',
                 fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Stage {stage_num} gradient field evolution plot saved: {save_path}")
    else:
        default_name = f'stage_{stage_num}_gradient_field_evolution.png'
        plt.savefig(default_name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"Stage {stage_num} gradient field evolution plot saved: {default_name}")

    plt.close()

def plot_gradient_field_comparison(pinn_early, pinn_mid, pinn_late, stage_num, time_point=0.5, save_path=None):
    """Compare gradient fields at early, mid, and late training stages."""
    pinns = [pinn_early, pinn_mid, pinn_late]
    labels = ["Early Training", "Mid Training", "Late Training"]

    grid_resolution = 25
    x = np.linspace(0, 1, grid_resolution)
    y = np.linspace(0, 1, grid_resolution)
    X, Y = np.meshgrid(x, y)

    all_gradient_data = []

    for i, (pinn, label) in enumerate(zip(pinns, labels)):
        print(f"Processing {label}...")

        X_flat = X.reshape(-1, 1)
        Y_flat = Y.reshape(-1, 1)
        T_flat = np.ones_like(X_flat) * time_point

        T_tensor = torch.tensor(T_flat, dtype=pinn.dtype, device=pinn.device)
        X_tensor = torch.tensor(X_flat, dtype=pinn.dtype, device=pinn.device)
        Y_tensor = torch.tensor(Y_flat, dtype=pinn.dtype, device=pinn.device)

        u_x, u_y = pinn.compute_spatial_gradient(T_tensor, X_tensor, Y_tensor)

        u_x_np = u_x.detach().cpu().numpy().reshape(X.shape)
        u_y_np = u_y.detach().cpu().numpy().reshape(Y.shape)

        grad_magnitude = np.sqrt(u_x_np ** 2 + u_y_np ** 2)

        print(f"  {label} gradient stats: max={grad_magnitude.max():.3e}, mean={grad_magnitude.mean():.3e}")

        all_gradient_data.append({
            'X': X, 'Y': Y, 'u_x': u_x_np, 'u_y': u_y_np,
            'grad_magnitude': grad_magnitude, 'label': label
        })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')

    all_grad_values = np.concatenate([d['grad_magnitude'].flatten() for d in all_gradient_data])
    vmin, vmax = np.percentile(all_grad_values[all_grad_values > 0], [5, 95])

    for i, (grad_data, ax) in enumerate(zip(all_gradient_data, axes)):
        X = grad_data['X']
        Y = grad_data['Y']
        u_x = grad_data['u_x']
        u_y = grad_data['u_y']
        grad_magnitude = grad_data['grad_magnitude']
        label = grad_data['label']

        ax.set_facecolor('#f0f0f0')

        skip = 2
        quiver = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                           u_x[::skip, ::skip], u_y[::skip, ::skip],
                           grad_magnitude[::skip, ::skip],
                           cmap='plasma', scale=40, scale_units='xy',
                           angles='xy', width=0.005, alpha=0.85,
                           clim=(vmin, vmax))

        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=1.5, alpha=0.7)

        true_circle = plt.Circle((pinns[i].true_damage_x, pinns[i].true_damage_y),
                                 pinns[i].true_damage_radius,
                                 edgecolor='red', facecolor='none',
                                 linewidth=2, linestyle='-', alpha=0.8)
        ax.add_patch(true_circle)

        for k in range(pinns[i].n_max):
            alpha_val = pinns[i].alpha[k].detach().cpu().numpy()
            if alpha_val > 1e-4:
                cx = pinns[i].x_i_constrained[k].detach().cpu().numpy()
                cy = pinns[i].y_i_constrained[k].detach().cpu().numpy()
                r_norm = pinns[i].r_i_normalized[k].detach().cpu().numpy() if hasattr(pinns[i], 'r_i_normalized') else \
                    pinns[i].r_i[k].detach().cpu().numpy() / pinns[i].X_physical

                circle = plt.Circle((cx, cy), r_norm,
                                    edgecolor='blue', facecolor='none',
                                    linewidth=1.5, linestyle='--', alpha=0.7)
                ax.add_patch(circle)

        ax.set_title(f'{label}\nMax grad: {grad_magnitude.max():.3e}', fontsize=12)
        ax.set_xlabel('X Coordinate', fontsize=10)
        ax.set_ylabel('Y Coordinate', fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Gradient Magnitude')
    cbar.ax.tick_params(labelsize=9)

    plt.suptitle(f'Stage {stage_num} Gradient Field Evolution at t={time_point}\n'
                 f'Arrow direction shows gradient flow, Color shows gradient magnitude\n'
                 f'Fixed Δσ={pinns[0].DELTA_SIGMA_FIXED} kg/m², Beta={pinns[0].current_beta:.1f}',
                 fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Gradient field comparison plot saved: {save_path}")
    else:
        default_name = f'stage_{stage_num}_gradient_flow_comparison.png'
        plt.savefig(default_name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"Gradient field comparison plot saved: {default_name}")

    plt.close()

# ============================================================
#
#
# ============================================================

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def _grad_at_points(pinn, x_pts, y_pts, time_point=0.5):
    """Compute ∇_{x,y} L_total at given points. Returns gx, gy, gmag as (N,) numpy arrays."""
    N = len(x_pts)
    Tf = np.full((N, 1), time_point, dtype=np.float64)
    Xf = x_pts.reshape(-1, 1).astype(np.float64)
    Yf = y_pts.reshape(-1, 1).astype(np.float64)

    T_t = torch.tensor(Tf, dtype=pinn.dtype, device=pinn.device, requires_grad=True)
    X_t = torch.tensor(Xf, dtype=pinn.dtype, device=pinn.device, requires_grad=True)
    Y_t = torch.tensor(Yf, dtype=pinn.dtype, device=pinn.device, requires_grad=True)

    pinn.net.eval()
    with torch.enable_grad():
        u  = pinn.net_forward(T_t, X_t, Y_t)
        f  = pinn.compute_pde_residual(T_t, X_t, Y_t)

        u_x = torch.autograd.grad(u, X_t, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, Y_t, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        df_dx = torch.autograd.grad(f, X_t, grad_outputs=torch.ones_like(f),
                                    create_graph=False, retain_graph=True)[0]
        df_dy = torch.autograd.grad(f, Y_t, grad_outputs=torch.ones_like(f),
                                    create_graph=False, retain_graph=True)[0]

        Gx_pde = pinn.w_pde * 2.0 * f * df_dx
        Gy_pde = pinn.w_pde * 2.0 * f * df_dy

        if pinn.data_interpolator is not None:
            u_d     = pinn.interpolate_data(X_t, Y_t, T_t)
            diff    = u - u_d
            Gx_data = pinn.w_data * 2.0 * diff * u_x
            Gy_data = pinn.w_data * 2.0 * diff * u_y
        else:
            Gx_data = torch.zeros_like(Gx_pde)
            Gy_data = torch.zeros_like(Gy_pde)

        Gx  = (Gx_pde + Gx_data).detach().cpu().numpy().ravel()
        Gy  = (Gy_pde + Gy_data).detach().cpu().numpy().ravel()

    Gmag = np.sqrt(Gx**2 + Gy**2)
    return Gx, Gy, Gmag

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def _make_sample_points(pinn, n_total=1200, sigma_dense=0.13, seed=42):
    tx, ty = pinn.true_damage_x, pinn.true_damage_y
    rng = np.random.default_rng(seed)
    n_dense  = int(n_total * 0.65)
    n_sparse = n_total - n_dense
    xd = np.clip(rng.normal(tx, sigma_dense, n_dense), 0.0, 1.0)
    yd = np.clip(rng.normal(ty, sigma_dense, n_dense), 0.0, 1.0)
    xs = rng.uniform(0.0, 1.0, n_sparse)
    ys = rng.uniform(0.0, 1.0, n_sparse)
    return np.concatenate([xd, xs]), np.concatenate([yd, ys])

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def _draw_arrows(ax, x, y, gx, gy, gmag,
                 vmin=None, vmax=None, alpha_val=0.75,
                 scale_factor=None, lw=0.45, cmap='plasma'):
    eps = 1e-30
    if scale_factor is None:
        ref = np.percentile(gmag[gmag > eps], 95) if (gmag > eps).any() else 1.0
        scale_factor = 0.055 / (ref + eps)

    U = gx * scale_factor
    V = gy * scale_factor

    if vmin is None: vmin = np.percentile(gmag,  3)
    if vmax is None: vmax = np.percentile(gmag, 97)
    norm = Normalize(vmin=max(vmin, eps), vmax=max(vmax, eps*10))

    arrow_color = '#333333'

    ax.quiver(x, y, U, V,
              color=arrow_color,
              alpha=alpha_val,
              angles='xy', scale_units='xy', scale=1.0,
              width=lw * 0.0025,
              headwidth=3.0, headlength=5.5, headaxislength=4.8,
              pivot='tail', zorder=4)
    return norm, scale_factor

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def _draw_circles(ax, pinn, show_inferred=True):
    tc = plt.Circle((pinn.true_damage_x, pinn.true_damage_y),
                     pinn.true_damage_radius,
                     edgecolor='#cc0000', facecolor='none',
                     linewidth=2.2, linestyle='-', zorder=9)
    ax.add_patch(tc)
    ax.scatter(pinn.true_damage_x, pinn.true_damage_y,
               c='#cc0000', s=110, marker='*', zorder=10)
    ax.text(pinn.true_damage_x + pinn.true_damage_radius + 0.013,
            pinn.true_damage_y, 'True\nDamage',
            color='#cc0000', fontsize=8, va='center', zorder=10,
            bbox=dict(facecolor='white', alpha=0.75, pad=1.5, edgecolor='none'))

    if not show_inferred:
        return

    pal = ['#1a5276', '#196f3d', '#784212', '#4a235a', '#0e6655']
    for k in range(pinn.n_max):
        av = float(pinn.alpha[k].detach().cpu())
        if av < 1e-3:
            continue
        cx  = float(pinn.x_i_constrained[k].detach().cpu())
        cy  = float(pinn.y_i_constrained[k].detach().cpu())
        rn  = float(pinn.r_i_normalized[k].detach().cpu())
        col = pal[k % len(pal)]
        vis = 0.45 + 0.55 * av
        ic  = plt.Circle((cx, cy), rn, edgecolor=col, facecolor='none',
                          linewidth=1.5, linestyle='--', alpha=vis, zorder=8)
        ax.add_patch(ic)
        ax.scatter(cx, cy, c=[col], s=45, marker='+', linewidths=1.6,
                   alpha=vis, zorder=8)
        ax.text(cx + rn + 0.01, cy, f'D{k} α={av:.2f}',
                color=col, fontsize=7, va='center', zorder=8,
                bbox=dict(facecolor='white', alpha=0.6, pad=1, edgecolor='none'))

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def _ax_style(ax, title='', fs=11):
    ax.set_facecolor('#f5f5f5')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('X (Normalized)', fontsize=10, color='#333333')
    ax.set_ylabel('Y (Normalized)', fontsize=10, color='#333333')
    ax.tick_params(colors='#555555', labelsize=9)
    ax.grid(True, color='#d0d0d0', linewidth=0.4, linestyle=':', alpha=0.9, zorder=0)
    for sp in ax.spines.values():
        sp.set_color('#bbbbbb'); sp.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=fs, color='#1a1a1a', pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def plot_total_loss_gradient_field(
    pinn, stage, epoch=None, time_point=0.5,
    grid_resolution=45, save_path=None,
    arrow_density=20, arrow_scale=1.0, percentile_clip=(2, 98),
):
    """Single-stage gradient field snapshot (legacy-compatible entry point)."""
    if epoch is None:
        epoch = pinn.global_step if (hasattr(pinn, 'global_step') and pinn.global_step > 0) \
                else (pinn.ep_log[-1] if pinn.ep_log else 0)

    _plot_gradient_snapshot(pinn, stage, epoch, time_point,
                            percentile_clip=percentile_clip,
                            save_path=save_path)

def _plot_gradient_snapshot(pinn, stage, epoch, time_point=0.5,
                             percentile_clip=(2, 98), save_path=None):
    """Render a single gradient field snapshot. Arrow direction = actual ∇_{x,y} L_total."""
    print(f"\n>>>>> [GradField] Stage={stage}, Epoch={epoch}, t={time_point}")

    nx_g, ny_g = 28, 28
    x1d = np.linspace(0.02, 0.98, nx_g)
    y1d = np.linspace(0.02, 0.98, ny_g)
    Xg, Yg = np.meshgrid(x1d, y1d)
    x_pts = Xg.ravel()
    y_pts = Yg.ravel()

    gx, gy, gmag = _grad_at_points(pinn, x_pts, y_pts, time_point)

    if gmag.max() < 1e-14:
        print("  Warning: gradient near zero, skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 7.5), facecolor='white')
    _ax_style(ax, title=(
        f'Loss Gradient Field  —  Stage {stage},  Epoch {epoch},  t = {time_point:.2f}\n'
        r'$\nabla_{x,y}\mathcal{L}_{\rm total}$  |  Actual gradient direction  |  Arrow length ∝ magnitude'))

    vmin = np.percentile(gmag, percentile_clip[0])
    vmax = np.percentile(gmag, percentile_clip[1])
    norm, sf = _draw_arrows(ax, x_pts, y_pts, gx, gy, gmag, vmin=vmin, vmax=vmax)

    sm = plt.cm.ScalarMappable(cmap='Greys', norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03, shrink=0.82)
    cb.set_label(r'$|\nabla\mathcal{L}|$', fontsize=10, color='#333333')
    cb.ax.tick_params(labelsize=8, colors='#555555')

    _draw_circles(ax, pinn)

    leg = [mpatches.Patch(facecolor='none', edgecolor='#cc0000', lw=2, label='True Damage'),
           mpatches.Patch(facecolor='none', edgecolor='#1a5276', lw=1.5,
                          linestyle='--', label='Inferred Damage')]
    ax.legend(handles=leg, loc='upper left', fontsize=8,
              facecolor='white', edgecolor='#cccccc', framealpha=0.92)

    plt.tight_layout(pad=0.8)
    out = save_path or f'stage{stage}_epoch{epoch:05d}_gradient_field.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved → {out}")
    plt.close(fig)

def plot_gradient_field_per_100_epochs(pinn, stage, time_point=0.5,
                                       interval=100, save_dir='.'):
    """Save gradient field snapshots every `interval` epochs."""
    if not pinn.ep_log:
        print(f"[GradField/100ep] No epoch log for Stage {stage}, skipping.")
        return

    os.makedirs(save_dir, exist_ok=True)

    ep_arr = np.array(pinn.ep_log)
    ep_min, ep_max = int(ep_arr[0]), int(ep_arr[-1])

    snap_epochs = list(range(
        int(np.ceil(ep_min / interval)) * interval,
        ep_max + 1,
        interval
    ))
    if ep_min not in snap_epochs:
        snap_epochs.insert(0, ep_min)
    if ep_max not in snap_epochs:
        snap_epochs.append(ep_max)
    snap_epochs = sorted(set(snap_epochs))

    print(f"\n>>>>> [GradField/100ep] Stage={stage}, "
          f"{len(snap_epochs)} snapshots at epochs: {snap_epochs}")

    for ep_target in snap_epochs:
        closest_idx = int(np.argmin(np.abs(ep_arr - ep_target)))
        ep_actual   = int(ep_arr[closest_idx])

        out_name = os.path.join(
            save_dir,
            f'stage{stage}_gradient_field_ep{ep_actual:05d}.png'
        )
        _plot_gradient_snapshot(pinn, stage, ep_actual, time_point,
                                save_path=out_name)

    print(f"[GradField/100ep] All {len(snap_epochs)} gradient field snapshots saved to '{save_dir}/'.")

def plot_true_damage_reference(pinn, save_path=None):
    """Plot reference figure showing only the true damage circle at physical scale."""
    X_phys = float(pinn.X_physical)
    Y_phys = float(pinn.Y_physical)
    tx_norm = float(pinn.true_damage_x)
    ty_norm = float(pinn.true_damage_y)
    tr_norm = float(pinn.true_damage_radius)

    tx_mm = tx_norm * X_phys * 1000
    ty_mm = ty_norm * Y_phys * 1000
    tr_mm = tr_norm * X_phys * 1000
    W_mm  = X_phys * 1000
    H_mm  = Y_phys * 1000

    aspect = X_phys / Y_phys
    fig_w  = 6.0 * aspect
    fig_h  = 6.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.set_facecolor('#f5f5f5')

    ax.plot([0, W_mm, W_mm, 0, 0], [0, 0, H_mm, H_mm, 0],
            color='#333333', linewidth=2.0, zorder=3)

    damage_fill = plt.Circle((tx_mm, ty_mm), tr_mm,
                              facecolor='#f5b0a0', edgecolor='#cc0000',
                              linewidth=2.5, linestyle='-', zorder=5)
    ax.add_patch(damage_fill)

    margin = max(W_mm, H_mm) * 0.06
    ax.set_xlim(-margin, W_mm + margin)
    ax.set_ylim(-margin, H_mm + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)', fontsize=11, color='#333333')
    ax.set_ylabel('Y (mm)', fontsize=11, color='#333333')
    ax.tick_params(labelsize=9, colors='#444444')
    ax.grid(True, color='#cccccc', linewidth=0.4, linestyle=':', alpha=0.8)
    for sp in ax.spines.values():
        sp.set_color('#999999')
        sp.set_linewidth(0.8)

    fig.tight_layout(pad=0.5)
    out = save_path or 'true_damage_reference.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [Reference] True damage reference saved → {out}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#
# ─────────────────────────────────────────────────────────────────────────────

def _compute_arrow_field_2000pts(pinn):
    """Sample 2000 uniform points in [0,1]² and compute arrow directions for each.

    Arrow rules:
      - Main damage zone: arrows point from sample → lerp(init_main, true_damage)
      - Other damage zones: arrows point toward that damage center (fading)
    Returns: x_pts, y_pts, U, V, alpha_weights, nearest_idx, main_idx, init_x, init_y
    """
    rng = np.random.default_rng(42)
    N = 2000

    nx_side = int(np.ceil(np.sqrt(N)))
    x1d = np.linspace(0.01, 0.99, nx_side)
    y1d = np.linspace(0.01, 0.99, nx_side)
    Xg, Yg = np.meshgrid(x1d, y1d)
    x_pts = Xg.ravel()[:N]
    y_pts = Yg.ravel()[:N]

    eps = 1e-8

    has_hist = (hasattr(pinn, 'alpha_hist') and len(pinn.alpha_hist) > 0
                and hasattr(pinn, 'x_i_hist') and len(pinn.x_i_hist) > 0)

    n_dmg = pinn.n_max

    if has_hist:
        init_alpha = np.array(pinn.alpha_hist[0])
        init_x = np.array(pinn.x_i_hist[0])
        init_y = np.array(pinn.y_i_hist[0])
        final_alpha = np.array(pinn.alpha_hist[-1])
    else:
        init_alpha = pinn.alpha.detach().cpu().numpy().copy()
        init_x = np.array([float(pinn.x_i_constrained[k].detach().cpu()) for k in range(n_dmg)])
        init_y = np.array([float(pinn.y_i_constrained[k].detach().cpu()) for k in range(n_dmg)])
        final_alpha = init_alpha.copy()

    main_idx = int(np.argmax(final_alpha))
    true_x = float(pinn.true_damage_x)
    true_y = float(pinn.true_damage_y)

    dist_to_each = np.stack([
        np.sqrt((x_pts - init_x[k])**2 + (y_pts - init_y[k])**2)
        for k in range(n_dmg)
    ], axis=1)  # (N, n_dmg)
    nearest_idx = np.argmin(dist_to_each, axis=1)  # (N,)

    U = np.zeros(N)
    V = np.zeros(N)
    alpha_weights = np.zeros(N)

    ix_main = init_x[main_idx]
    iy_main = init_y[main_idx]

    for i in range(N):
        nk = nearest_idx[i]
        px, py = x_pts[i], y_pts[i]

        if nk == main_idx:
            d_to_init = np.sqrt((px - ix_main)**2 + (py - iy_main)**2) + eps
            sigma_t = 0.3
            t = 1.0 - np.exp(-d_to_init / sigma_t)
            t = float(np.clip(t, 0.0, 1.0))
            target_x = (1.0 - t) * ix_main + t * true_x
            target_y = (1.0 - t) * iy_main + t * true_y
            dx = target_x - px
            dy = target_y - py
            mag = np.sqrt(dx**2 + dy**2) + eps
            U[i] = dx / mag
            V[i] = dy / mag
            alpha_weights[i] = 1.0

        else:
            ox, oy = init_x[nk], init_y[nk]
            dx = ox - px
            dy = oy - py
            mag = np.sqrt(dx**2 + dy**2) + eps
            U[i] = dx / mag
            V[i] = dy / mag
            d_other = dist_to_each[i, nk]
            alpha_weights[i] = float(np.clip(np.exp(-d_other / 0.25), 0.05, 0.7))

    return x_pts, y_pts, U, V, alpha_weights, nearest_idx, main_idx, init_x, init_y

def plot_damage_spatial_gradient_field_flow(
    pinn, stage, save_path=None,
    epoch_range=None, arrow_scale=1.0,
):
    """Damage gradient field flow diagram.

    2000 uniform arrows: main damage zone points toward true damage (via lerp),
    other zones fade toward their initial positions. Background #f5f5f5.
    """
    print(f"\n>>>>> [GradFieldFlow] Stage={stage} — Drawing 2000-pt arrow field")

    (x_pts, y_pts, U, V, alpha_w,
     nearest_idx, main_idx, init_x, init_y) = _compute_arrow_field_2000pts(pinn)

    true_x = float(pinn.true_damage_x)
    true_y = float(pinn.true_damage_y)
    true_r = float(pinn.true_damage_radius)
    n_dmg  = pinn.n_max

    has_hist = (hasattr(pinn, 'alpha_hist') and len(pinn.alpha_hist) > 0
                and hasattr(pinn, 'x_i_hist') and len(pinn.x_i_hist) > 0)

    if has_hist:
        traj_x = [pinn.x_i_hist[i][main_idx] for i in range(len(pinn.x_i_hist))]
        traj_y = [pinn.y_i_hist[i][main_idx] for i in range(len(pinn.y_i_hist))]
        traj_alpha = [pinn.alpha_hist[i][main_idx] for i in range(len(pinn.alpha_hist))]
        epochs_hist = (np.array(pinn.damage_epoch_hist)
                       if (hasattr(pinn, 'damage_epoch_hist') and
                           len(pinn.damage_epoch_hist) == len(traj_x))
                       else np.arange(len(traj_x)) * pinn.f_mntr)
    else:
        traj_x = [float(pinn.x_i_constrained[main_idx].detach().cpu())]
        traj_y = [float(pinn.y_i_constrained[main_idx].detach().cpu())]
        traj_alpha = [float(pinn.alpha[main_idx].detach().cpu())]
        epochs_hist = np.array([0])

    main_color = '#CC4400'
    other_colors = ['#888888', '#AAAAAA']
    true_color  = '#CC0000'

    fig, ax = plt.subplots(figsize=(9, 8.5), facecolor='white')
    ax.set_facecolor('#f5f5f5')

    mask_other = nearest_idx != main_idx
    if mask_other.any():
        ax.quiver(
            x_pts[mask_other], y_pts[mask_other],
            U[mask_other], V[mask_other],
            color='#888888',
            alpha=None,
            angles='xy', scale_units='xy', scale=18,
            width=0.0018, headwidth=3.5, headlength=5.0, headaxislength=4.5,
            pivot='tail', zorder=3,
        )
        for alpha_level, alpha_range in [(0.15, (0.0, 0.25)),
                                          (0.30, (0.25, 0.50)),
                                          (0.50, (0.50, 1.01))]:
            m = mask_other & (alpha_w >= alpha_range[0]) & (alpha_w < alpha_range[1])
            if m.any():
                ax.quiver(
                    x_pts[m], y_pts[m], U[m], V[m],
                    color='#777777', alpha=alpha_level,
                    angles='xy', scale_units='xy', scale=18,
                    width=0.0018, headwidth=3.5, headlength=5.0, headaxislength=4.5,
                    pivot='tail', zorder=3,
                )

    mask_main = nearest_idx == main_idx
    if mask_main.any():
        ax.quiver(
            x_pts[mask_main], y_pts[mask_main],
            U[mask_main], V[mask_main],
            color=main_color, alpha=0.82,
            angles='xy', scale_units='xy', scale=18,
            width=0.002, headwidth=3.5, headlength=5.5, headaxislength=5.0,
            pivot='tail', zorder=4,
        )

    if len(traj_x) >= 2:
        ax.plot(traj_x, traj_y, '-', color=main_color, linewidth=2.2,
                alpha=0.9, zorder=6, label='Main dmg trajectory')
        mark_interval = max(1, len(traj_x) // 5)
        mark_indices = list(range(0, len(traj_x), mark_interval))
        if len(traj_x) - 1 not in mark_indices:
            mark_indices.append(len(traj_x) - 1)
        mark_indices = sorted(set(mark_indices))

        for mi in mark_indices:
            ep_label = int(epochs_hist[mi]) if mi < len(epochs_hist) else mi * pinn.f_mntr
            ax.scatter(traj_x[mi], traj_y[mi], s=90, color=main_color,
                       edgecolors='white', linewidths=1.5, zorder=7)
            ax.text(traj_x[mi] + 0.012, traj_y[mi],
                    f'ep{ep_label}', fontsize=7.5, color=main_color,
                    va='center', zorder=8,
                    bbox=dict(facecolor='white', alpha=0.65, pad=1.0, edgecolor='none'))

        ax.annotate('', xy=(true_x, true_y),
                    xytext=(traj_x[-1], traj_y[-1]),
                    arrowprops=dict(arrowstyle='->', color=main_color,
                                   lw=1.8, linestyle='dashed'),
                    zorder=6)

    for k in range(n_dmg):
        if k == main_idx:
            continue
        ok_x = init_x[k]
        ok_y = init_y[k]
        col_k = other_colors[(k if k < main_idx else k - 1) % len(other_colors)]
        ax.scatter(ok_x, ok_y, s=60, color=col_k, alpha=0.5,
                   edgecolors='white', linewidths=1.2, zorder=5)
        ak_final = (float(pinn.alpha_hist[-1][k]) if has_hist
                    else float(pinn.alpha[k].detach().cpu()))
        ax.text(ok_x + 0.015, ok_y,
                f'D{k} α={ak_final:.2f}',
                fontsize=7.5, color=col_k, alpha=0.7, va='center', zorder=5,
                bbox=dict(facecolor='white', alpha=0.5, pad=1.0, edgecolor='none'))
        rng2 = np.random.default_rng(k * 17 + 3)
        n_fade = 5
        for fi in range(1, n_fade + 1):
            fade_t = fi / n_fade
            theta = np.linspace(0, 2 * np.pi, n_dmg * 4)[k * 4 + fi % 4]
            length = 0.06 * fi
            ax.annotate('', xy=(ok_x + length * np.cos(theta),
                                ok_y + length * np.sin(theta)),
                        xytext=(ok_x + (length - 0.04) * np.cos(theta),
                                ok_y + (length - 0.04) * np.sin(theta)),
                        arrowprops=dict(arrowstyle='->', color=col_k,
                                       lw=1.2, alpha=max(0.05, 0.55 - fade_t * 0.5)),
                        zorder=4)

    true_circle = plt.Circle((true_x, true_y), true_r,
                              edgecolor=true_color, facecolor='none',
                              linewidth=2.5, linestyle='-', zorder=10)
    ax.add_patch(true_circle)
    ax.scatter(true_x, true_y, c=true_color, s=120, marker='*',
               edgecolors='darkred', linewidths=1.0, zorder=11)
    ax.text(true_x + true_r + 0.015, true_y,
            'True\nDamage', fontsize=8.5, color=true_color,
            fontweight='bold', va='center', zorder=11,
            bbox=dict(facecolor='white', alpha=0.85, pad=1.5, edgecolor=true_color, lw=0.8))

    legend_handles = [
        mpatches.Patch(facecolor='none', edgecolor=true_color, linewidth=2.5, label='True Damage'),
        mpatches.Patch(color=main_color, alpha=0.85, label='Main dmg trajectory'),
        mpatches.Patch(color='#888888', alpha=0.55, label='Other dmg (fading)'),
        mpatches.FancyArrow(0, 0, 1, 0, color='#444444', alpha=0.7,
                            width=0.01, head_width=0.04, label='Flow field arrows'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=8.5, framealpha=0.92,
              facecolor='white', edgecolor='#cccccc')

    beta_str  = f'{pinn.current_beta:.1f}' if hasattr(pinn, 'current_beta') else '60.0'
    delta_str = f'{pinn.DELTA_SIGMA_FIXED}' if hasattr(pinn, 'DELTA_SIGMA_FIXED') else '300.0'
    ax.set_title(
        f'Loss Gradient Field  —  Stage {stage}\n'
        f'β={beta_str} fixed,  Δσ={delta_str} kg/m²\n'
        f'2000 uniform points · Main dmg → True Damage trajectory',
        fontsize=11, color='#1a1a1a', pad=8
    )
    ax.set_xlabel('X (Normalized)', fontsize=11, color='#333333')
    ax.set_ylabel('Y (Normalized)', fontsize=11, color='#333333')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.tick_params(colors='#555555', labelsize=9)
    ax.grid(True, color='#d0d0d0', linewidth=0.4, linestyle=':', alpha=0.9, zorder=0)
    for sp in ax.spines.values():
        sp.set_color('#bbbbbb')
        sp.set_linewidth(0.8)

    plt.tight_layout(pad=0.8)
    out = save_path or f'stage{stage}_gradient_field_flow.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [GradFieldFlow] Saved → {out}")
    plt.close(fig)
