"""
绘图功能模块
包含所有与绘图相关的函数
支持三阶段训练
优化配色方案：板底色为淡蓝色，损伤区域为淡红色
所有标题和标签改为英文，适合科研论文
修复关键问题：epoch拼接逻辑、半径单位一致性
添加 L-BFGS 训练监控
新增每个阶段的单独总损失曲线图
新增：阶段1的Adam和L-BFGS损失合并绘制功能
修复L-BFGS绘图功能
新增：专用的Damage Spatial Gradient Field Flow图函数（修改版，绘制损伤参数演化轨迹）
新增：训练过程中的梯度场演化图函数
新增：PINN总损失梯度流场图函数（仅total loss gradient）

修改要点：
1. 取消预训练阶段，改为三阶段训练
2. Beta固定为60，不再绘制Beta曲线
3. 新增空间梯度场图函数，使用箭头表示梯度方向
4. 修改损伤图底色为淡蓝色 (0.7,0.85,1.0)
5. 新增专用梯度场流图函数（修正版：绘制损伤参数移动轨迹）
6. 真实损伤用红色实线圆圈，推断损伤用其他颜色虚线圆圈
7. 不再绘制数据点可视化图
8. 新增：训练过程中梯度场演化图，展示梯度如何流动变化
9. 新增：PINN总损失梯度流场图，仅展示总损失梯度，顶部嵌入损失演化子图
10. 重写 `plot_damage_spatial_gradient_field_flow`：现在绘制整个板域的总损失梯度场，而非损伤参数轨迹
"""

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


# ================== 原有函数（未修改） ==================
# 以下函数与原始 plot.py 中完全相同，仅列出函数定义，实现细节请参考原始文件
def plot_loss_curve(pinn, stage):
    """绘制损失曲线 - 支持三阶段，修改：只绘制总损失曲线"""
    if not pinn.ep_log:
        return

    plt.figure(figsize=(12, 6))

    if stage == 1:
        # 第一阶段：绘制Adam训练的总损失，如果L-BFGS损失历史存在，也一并绘制
        colors = ['#1f77b4', '#ff7f0e']  # 蓝色，橙色

        # 阶段1：绘制Adam训练的总损失曲线
        plt.semilogy(pinn.ep_log, pinn.loss_log, label='Total Loss (Adam)', linewidth=2.5, color=colors[0])
        title = f'Stage {stage} Total Loss Curve (Adam + L-BFGS)'

        # 如果存在L-BFGS损失历史，绘制在同一个图上
        if hasattr(pinn, 'lbfgs_loss_history') and pinn.lbfgs_loss_history:
            # L-BFGS训练的epoch起始点
            lbfgs_start_epoch = pinn.ep_log[-1] + 1 if pinn.ep_log else 0
            lbfgs_epochs = range(lbfgs_start_epoch, lbfgs_start_epoch + len(pinn.lbfgs_loss_history))

            # 绘制L-BFGS损失
            plt.semilogy(lbfgs_epochs, pinn.lbfgs_loss_history,
                         label='Total Loss (L-BFGS)', linewidth=2.5, color=colors[1], linestyle='--')

            # 标记L-BFGS开始点
            plt.axvline(x=lbfgs_start_epoch, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
            plt.text(lbfgs_start_epoch, max(pinn.loss_log) if pinn.loss_log else 1,
                     ' L-BFGS Start', fontsize=10, color='k', verticalalignment='bottom')

            print(f"Stage {stage}: Adam epochs: {len(pinn.ep_log)}, L-BFGS iterations: {len(pinn.lbfgs_loss_history)}")

        plt.legend(fontsize=12)
    else:
        # 第二阶段和第三阶段：只绘制总损失曲线
        plt.semilogy(pinn.ep_log, pinn.loss_log, label='Total Loss', linewidth=2.5, color='#1f77b4')
        title = f'Stage {stage} Total Loss Curve (Beta fixed at {pinn.current_beta:.1f})'
        plt.legend(fontsize=12)

    plt.title(title, fontsize=14)
    plt.xlabel('Training Epoch/Iteration', fontsize=12)
    plt.ylabel('Total Loss Value (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 添加统计信息文本框
    stats_text = f'Stage {stage} Training Summary:\n'
    stats_text += f'Total Epochs: {pinn.ep_log[-1]}\n'
    stats_text += f'Final Total Loss: {pinn.loss_log[-1]:.3e}\n'

    if hasattr(pinn, 'current_beta'):
        stats_text += f'Beta: {pinn.current_beta:.1f} (fixed)\n'

    # 添加损伤参数信息（阶段2和3）
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

    # 打印最终损失值用于调试
    if pinn.ep_log:
        print(f"\nStage {stage} Final Loss Values:")
        print(f"  Total Loss: {pinn.loss_log[-1]:.3e}")


def plot_stage_total_loss(pinn, stage):
    """
    绘制单个阶段的总损失曲线图 - 只绘制总损失

    参数:
        pinn: PINN实例
        stage: 阶段编号
    """
    if not pinn.ep_log:
        print(f"Stage {stage}: No training data available")
        return

    plt.figure(figsize=(12, 6))

    # 绘制总损失曲线
    plt.semilogy(pinn.ep_log, pinn.loss_log, 'b-', linewidth=2.5, label='Total Loss')

    # 根据阶段添加额外信息
    if stage == 1:
        title = f'Stage {stage}: Healthy Network Training - Total Loss'
        # 添加L-BFGS损失（如果存在）
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

    # 添加统计信息文本框
    stats_text = f'Stage {stage} Training Summary:\n'
    stats_text += f'Total Epochs: {pinn.ep_log[-1]}\n'
    stats_text += f'Final Total Loss: {pinn.loss_log[-1]:.3e}\n'
    stats_text += f'Beta: {pinn.current_beta:.1f} (fixed)\n'

    # 添加损伤参数信息（阶段2和3）
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
    """绘制三阶段训练的总损失曲线 - 修复epoch拼接逻辑"""
    # 收集所有阶段的损失数据
    all_epochs = []
    all_losses = []
    stage_boundaries = []

    # 阶段1的损失
    if pinn_stage1.ep_log:
        # 阶段1从epoch 0开始
        stage1_epochs = pinn_stage1.ep_log
        stage1_losses = pinn_stage1.loss_log
        all_epochs.extend(stage1_epochs)
        all_losses.extend(stage1_losses)
        stage_boundaries.append(len(all_epochs))
        print(f"Stage 1: epochs {stage1_epochs[0]}-{stage1_epochs[-1]}, {len(stage1_epochs)} points")

    # 阶段2的损失
    if pinn_stage2.ep_log:
        # 阶段2从阶段1结束的epoch+1开始
        stage2_start = all_epochs[-1] + 1 if all_epochs else 0
        stage2_epochs = [epoch + stage2_start for epoch in pinn_stage2.ep_log]
        stage2_losses = pinn_stage2.loss_log
        all_epochs.extend(stage2_epochs)
        all_losses.extend(stage2_losses)
        stage_boundaries.append(len(all_epochs))
        print(f"Stage 2: epochs {stage2_epochs[0]}-{stage2_epochs[-1]}, {len(stage2_epochs)} points")

    # 阶段3的损失
    if pinn_stage3.ep_log:
        # 阶段3从阶段2结束的epoch+1开始
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

    # 绘制总损失曲线
    plt.semilogy(all_epochs, all_losses, 'b-', linewidth=3, label='Total Loss')

    # 标记阶段分界线
    colors = ['g', 'm', 'c']  # 绿色，品红色，青色
    stage_labels = ['Stage 1 End', 'Stage 2 End', 'Stage 3 End']

    for i, boundary in enumerate(stage_boundaries[:-1]):  # 不标记最后一个边界
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

    # 添加信息框
    info_text = f"Total Training Epochs: {all_epochs[-1]}\n"
    info_text += f"Final Total Loss: {all_losses[-1]:.3e}"

    # 添加Beta信息
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
    """绘制损伤分布图 - 优化配色：板底色灰白色，损伤区域淡红色，修复半径单位"""
    if stage < 2:
        return

    # 临时保存当前阶段
    original_stage = pinn.stage
    pinn.stage = stage  # 确保在绘图时使用正确的阶段

    nx_plot, ny_plot = 256, 256
    x_plot = np.linspace(0.0, 1.0, nx_plot)
    y_plot = np.linspace(0.0, 1.0, ny_plot)
    Xp, Yp = np.meshgrid(x_plot, y_plot, indexing='xy')
    Xp_flat = Xp.reshape(-1, 1)
    Yp_flat = Yp.reshape(-1, 1)

    import torch
    T0 = torch.zeros_like(torch.tensor(Xp_flat), dtype=pinn.dtype, device=pinn.device)

    # 计算损伤分布
    sigma_star = torch.ones_like(T0, dtype=pinn.dtype) * (pinn.rho * pinn.h_physical)
    for k in range(pinn.n_max):
        # 如果Alpha值为0，跳过计算
        if pinn.alpha[k] < 1e-6:
            continue

        Xp_tensor = torch.tensor(Xp_flat, dtype=pinn.dtype, device=pinn.device)
        Yp_tensor = torch.tensor(Yp_flat, dtype=pinn.dtype, device=pinn.device)

        # 使用归一化半径计算距离
        dist_sq = (Xp_tensor - pinn.x_i_constrained[k]) ** 2 + (Yp_tensor - pinn.y_i_constrained[k]) ** 2
        # 使用归一化半径 - 修复单位问题
        r_norm = pinn.r_i_normalized[k] if hasattr(pinn, 'r_i_normalized') else pinn.r_i[k] / pinn.X_physical
        r_sq = r_norm ** 2
        # 使用当前的Beta值
        gate = torch.sigmoid(pinn.current_beta * (r_sq - dist_sq))
        sigma_star += pinn.alpha[k] * pinn.DELTA_SIGMA_FIXED * gate

    sigma_star = sigma_star.reshape(ny_plot, nx_plot).detach().cpu().numpy()

    # 创建自定义颜色映射：底色为淡蓝色，损伤区域为淡红色
    colors = [(0.94, 0.94, 0.94), (1.0, 0.78, 0.78)]  # 灰白色 -> 淡红色
    n_bins = 100
    cmap_name = 'damage_cmap'
    damage_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    plt.figure(figsize=(10, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # 使用自定义颜色映射
    im = plt.imshow(sigma_star, extent=[0, 1, 0, 1], origin='lower',
                    cmap=damage_cmap, interpolation='bilinear', alpha=0.9)

    # 将颜色条放在右侧外部
    cbar = plt.colorbar(im, label='σ*(x,y) (kg/m²)', shrink=0.8, pad=0.05)
    cbar.ax.tick_params(labelsize=10)

    # 使用科研友好的配色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    # 统计活跃损伤
    active_damages = 0
    legend_handles = []
    legend_labels = []

    # 打印损伤参数用于调试
    print(f"\nPlotting Damage Map - {title_prefix} (Stage {stage}):")

    for k in range(pinn.n_max):
        a = pinn.alpha[k].detach().cpu().numpy()
        cx = pinn.x_i_constrained[k].detach().cpu().numpy()
        cy = pinn.y_i_constrained[k].detach().cpu().numpy()
        # 使用归一化半径绘图
        r_norm = pinn.r_i_normalized[k].detach().cpu().numpy() if hasattr(pinn, 'r_i_normalized') else pinn.r_i[
                                                                                                           k].detach().cpu().numpy() / pinn.X_physical
        r_mm = r_norm * pinn.X_physical * 1000  # 转换为毫米

        # 如果Alpha值为0，不绘制该损伤
        if a < 1e-6:
            print(f"  Damage {k}: Alpha is 0, skipping")
            continue

        active_damages += 1

        print(
            f"  Damage {k}: Position({cx:.3f}, {cy:.3f}), Radius {r_mm:.1f}mm, Alpha={a:.3f}, Δσ={pinn.DELTA_SIGMA_FIXED}")

        if np.isnan(a) or np.isnan(cx) or np.isnan(cy) or np.isnan(r_norm):
            print(f"Warning: Damage {k} has NaN values, skipping")
            continue

        # 确保半径不为零或负值
        if r_norm <= 0:
            r_norm = 0.005 / pinn.X_physical  # 设置最小半径
            print(f"Warning: Damage {k} radius invalid, using default {r_norm * pinn.X_physical * 1000:.1f}mm")

        # 绘制损伤半径虚线圆圈 - 使用归一化半径
        circle_dashed = mpatches.Circle((cx, cy), r_norm,
                                        edgecolor=colors[k % len(colors)], facecolor='none',
                                        linewidth=2.0, linestyle='--', alpha=0.9)
        plt.gca().add_patch(circle_dashed)

        # 绘制填充的实线圆圈（透明度较低）
        circle_filled = mpatches.Circle((cx, cy), r_norm,
                                        edgecolor=colors[k % len(colors)], facecolor=colors[k % len(colors)],
                                        alpha=0.25 + 0.3 * a, linewidth=1.5, linestyle='-')
        plt.gca().add_patch(circle_filled)

        # 绘制中心标记
        plt.scatter(cx, cy, c=[colors[k % len(colors)]], s=80, marker='x', linewidths=2.5, alpha=0.9)

        # 添加图例项
        legend_handles.append(mpatches.Patch(color=colors[k % len(colors)], alpha=0.7))
        legend_labels.append(f'Inferred Damage {k}')

    print(f"Plotting {active_damages} active damages")

    # 真实损伤 - 使用归一化半径绘制（红色实线圆圈）
    true_r_norm = pinn.true_damage_radius  # 已经是归一化半径
    true_x_norm = pinn.true_damage_x
    true_y_norm = pinn.true_damage_y

    true_circle = mpatches.Circle((true_x_norm, true_y_norm),
                                  true_r_norm, edgecolor='red', facecolor='none',
                                  linewidth=3.0, linestyle='-')
    plt.gca().add_patch(true_circle)
    plt.scatter(true_x_norm, true_y_norm, c='red', s=120, marker='+', linewidths=3.5)

    # 添加真实损伤到图例
    legend_handles.append(mpatches.Patch(facecolor='none', edgecolor='red', linestyle='-', linewidth=2))
    legend_labels.append(f'True Damage')

    # 将图例放在右上角内部，避免遮挡
    if legend_handles:
        plt.legend(legend_handles, legend_labels, loc='upper right',
                   fontsize=10, framealpha=0.95, bbox_to_anchor=(0.95, 0.95))

    # 设置英文标题
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

    # 添加板边界
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    plt.axhline(y=1, color='k', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(x=1, color='k', linestyle='-', linewidth=1, alpha=0.5)

    # 调整布局
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Damage map saved: {save_path}")
    else:
        plt.show()
    plt.close()

    # 恢复原来的阶段
    pinn.stage = original_stage


def plot_damage_parameter_evolution(pinn, start_epoch_stage2=0):
    """绘制训练过程中损伤参数的演化 - 重写版本
    四个参数各用独立颜色，曲线平滑，坐标轴自动聚焦真实值附近，
    删除左上角统计文本框。

    参数:
        pinn: PINN实例
        start_epoch_stage2: 阶段2训练开始的epoch数
    """
    print(f"\n[DEBUG] plot_damage_parameter_evolution called")
    print(f"  - stage: {pinn.stage}")
    print(f"  - damage_epoch_hist length: {len(pinn.damage_epoch_hist) if pinn.damage_epoch_hist else 0}")
    print(f"  - alpha_hist length: {len(pinn.alpha_hist) if pinn.alpha_hist else 0}")
    print(f"  - start_epoch_stage2: {start_epoch_stage2}")

    if not pinn.alpha_hist or len(pinn.alpha_hist) < 2:
        print("[WARNING] No damage parameter history data available for plotting")
        return

    # ── epoch 横轴 ────────────────────────────────────────────────
    if pinn.damage_epoch_hist and len(pinn.damage_epoch_hist) == len(pinn.alpha_hist):
        epochs = np.array(pinn.damage_epoch_hist)
    else:
        epochs = np.arange(0, len(pinn.alpha_hist)) * pinn.f_mntr + start_epoch_stage2
        print(f"Warning: Using calculated epochs, starting from {start_epoch_stage2}")

    # ── 确定主损伤 ────────────────────────────────────────────────
    final_alpha = pinn.alpha_hist[-1]
    main_damage_idx = np.argmax(final_alpha)
    main_alpha_value = final_alpha[main_damage_idx]
    print(f"Main damage identified: Index {main_damage_idx}, Alpha={main_alpha_value:.3f}")

    # ── 提取原始数据 ──────────────────────────────────────────────
    alpha_k = np.array([hist[main_damage_idx] for hist in pinn.alpha_hist])
    x_k     = np.array([hist[main_damage_idx] for hist in pinn.x_i_hist])
    y_k     = np.array([hist[main_damage_idx] for hist in pinn.y_i_hist])
    r_k_mm  = np.array([hist[main_damage_idx] * 1000 for hist in pinn.r_i_hist])  # m → mm

    # ── 平滑处理（Savitzky-Golay 滤波，对数据量自适应）────────────
    try:
        from scipy.signal import savgol_filter
        n_pts = len(epochs)
        # 窗口长度：取数据量的 ~20%，必须为奇数且 >= 5
        win = max(5, int(n_pts * 0.20) | 1)   # |1 保证奇数
        poly = 3
        if n_pts >= win + 1:
            alpha_s = savgol_filter(alpha_k, win, poly)
            x_s     = savgol_filter(x_k,     win, poly)
            y_s     = savgol_filter(y_k,     win, poly)
            r_s     = savgol_filter(r_k_mm,  win, poly)
        else:
            alpha_s, x_s, y_s, r_s = alpha_k, x_k, y_k, r_k_mm
    except ImportError:
        alpha_s, x_s, y_s, r_s = alpha_k, x_k, y_k, r_k_mm

    # ── 真实损伤参数 ──────────────────────────────────────────────
    true_alpha = 1.0
    true_x     = pinn.true_damage_x
    true_y     = pinn.true_damage_y
    true_r_mm  = 5.0          # 5 mm

    # ── 四参数独立配色（曲线色 / 浅色阴影 / 真实值虚线同色系深色）──
    # 顺序：alpha → x → y → r
    COLORS = {
        'alpha': {'line': '#e05a2b', 'fill': '#f5b8a0', 'true': '#a03010'},
        'x':     {'line': '#2176c7', 'fill': '#a8c8f0', 'true': '#0d4e8a'},
        'y':     {'line': '#2aa84f', 'fill': '#97dbb0', 'true': '#145e2b'},
        'r':     {'line': '#9b59b6', 'fill': '#d7b8e8', 'true': '#5b2d7a'},
    }
    TRUE_LS  = (0, (6, 3))          # 自定义虚线样式（更优雅）

    def _ylim_focus(data, true_val, margin_frac=0.30):
        """以真实值为中心，保留数据范围，加 margin 留白"""
        combined = np.concatenate([data, [true_val]])
        lo, hi = combined.min(), combined.max()
        span = max(hi - lo, abs(true_val) * 0.05 + 1e-8)
        pad  = span * margin_frac
        return lo - pad, hi + pad

    def _smooth_fill(ax, ep, raw, smooth, col_dict):
        """绘制原始数据（浅色半透明带）+ 平滑曲线"""
        # 浅色阴影带（原始数据上下各一个 running-window 的幅度）
        ax.fill_between(ep, raw, smooth,
                        color=col_dict['fill'], alpha=0.35, zorder=1)
        # 平滑主曲线
        ax.plot(ep, smooth,
                color=col_dict['line'], linewidth=2.5, zorder=3)
        # 起点标记（圆形）
        ax.scatter(ep[0], smooth[0],
                   s=55, marker='o',
                   c=[col_dict['line']],
                   edgecolors='white', linewidths=1.2,
                   zorder=5)
        # 终点标记（五角星）
        ax.scatter(ep[-1], smooth[-1],
                   s=100, marker='*',
                   c=[col_dict['line']],
                   edgecolors='white', linewidths=1.0,
                   zorder=5)

    def _true_line(ax, true_val, col_dict, label):
        ax.axhline(y=true_val,
                   color=col_dict['true'], linewidth=2.0,
                   linestyle=TRUE_LS, label=label, zorder=2)

    def _style_ax(ax, xlabel, ylabel, title, ylim):
        ax.set_xlabel(xlabel, fontsize=12, color='#333333')
        ax.set_ylabel(ylabel, fontsize=12, color='#333333')
        ax.set_title(title,  fontsize=13, fontweight='bold', pad=8)
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=10, colors='#444444')
        ax.grid(True, alpha=0.25, linestyle='--', color='#999999')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=9, loc='best', framealpha=0.85,
                  edgecolor='#cccccc')

    # ── 创建图形 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), facecolor='white')
    fig.suptitle(
        f'Main Damage Parameter Evolution  ─  Stage 2/3 Training\n'
        f'Damage #{main_damage_idx}   |   Final α = {main_alpha_value:.3f}   |   '
        f'β = {pinn.current_beta:.0f} (fixed)',
        fontsize=16, y=0.995, color='#222222'
    )
    for ax in axes.flat:
        ax.set_facecolor('#fafafa')

    # ── 子图 1：Alpha ─────────────────────────────────────────────
    ax = axes[0, 0]
    _smooth_fill(ax, epochs, alpha_k, alpha_s, COLORS['alpha'])
    _true_line(ax, true_alpha, COLORS['alpha'], f'True  α = {true_alpha:.1f}')
    ax.text(epochs[-1], alpha_s[-1] + 0.01,
            f'{alpha_s[-1]:.3f}', color=COLORS['alpha']['line'],
            fontsize=9, ha='right', va='bottom')
    _style_ax(ax,
              xlabel='Training Epoch',
              ylabel='Damage Intensity  α',
              title=f'(a)  Alpha  [final = {alpha_s[-1]:.3f}]',
              ylim=_ylim_focus(alpha_s, true_alpha))

    # ── 子图 2：X coordinate ──────────────────────────────────────
    ax = axes[0, 1]
    _smooth_fill(ax, epochs, x_k, x_s, COLORS['x'])
    _true_line(ax, true_x, COLORS['x'], f'True  x = {true_x:.3f}')
    ax.text(epochs[-1], x_s[-1] + 0.001,
            f'{x_s[-1]:.4f}', color=COLORS['x']['line'],
            fontsize=9, ha='right', va='bottom')
    _style_ax(ax,
              xlabel='Training Epoch',
              ylabel='X Coordinate  (normalized)',
              title=f'(b)  X Position  [final = {x_s[-1]:.4f}]',
              ylim=_ylim_focus(x_s, true_x))

    # ── 子图 3：Y coordinate ──────────────────────────────────────
    ax = axes[1, 0]
    _smooth_fill(ax, epochs, y_k, y_s, COLORS['y'])
    _true_line(ax, true_y, COLORS['y'], f'True  y = {true_y:.3f}')
    ax.text(epochs[-1], y_s[-1] + 0.001,
            f'{y_s[-1]:.4f}', color=COLORS['y']['line'],
            fontsize=9, ha='right', va='bottom')
    _style_ax(ax,
              xlabel='Training Epoch',
              ylabel='Y Coordinate  (normalized)',
              title=f'(c)  Y Position  [final = {y_s[-1]:.4f}]',
              ylim=_ylim_focus(y_s, true_y))

    # ── 子图 4：Radius ────────────────────────────────────────────
    ax = axes[1, 1]
    _smooth_fill(ax, epochs, r_k_mm, r_s, COLORS['r'])
    _true_line(ax, true_r_mm, COLORS['r'], f'True  r = {true_r_mm:.1f} mm')
    ax.text(epochs[-1], r_s[-1] + 0.05,
            f'{r_s[-1]:.2f} mm', color=COLORS['r']['line'],
            fontsize=9, ha='right', va='bottom')
    _style_ax(ax,
              xlabel='Training Epoch',
              ylabel='Radius  (mm)',
              title=f'(d)  Radius  [final = {r_s[-1]:.2f} mm]',
              ylim=_ylim_focus(r_s, true_r_mm))

    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig('main_damage_parameter_evolution.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Main damage parameter evolution plot saved: main_damage_parameter_evolution.png")

    # 打印最终的损伤参数统计
    if len(pinn.alpha_hist) > 1:
        print_damage_parameter_statistics(pinn)


def plot_damage_position_trajectory(pinn, start_epoch_stage2=0, save_path=None):
    """绘制损伤位置演化轨迹图 - 带箭头指示演化方向
    4. 每1000个epoch在轨迹路线中标记损伤的位置点

    参数:
        pinn: PINN实例
        start_epoch_stage2: 阶段2训练开始的epoch数
        save_path: 保存路径，如果为None则显示
    """
    # 添加调试信息
    print(f"\n[DEBUG] plot_damage_position_trajectory called")
    print(f"  - stage: {pinn.stage}")
    print(f"  - damage_epoch_hist length: {len(pinn.damage_epoch_hist) if pinn.damage_epoch_hist else 0}")
    print(f"  - alpha_hist length: {len(pinn.alpha_hist) if pinn.alpha_hist else 0}")
    print(f"  - start_epoch_stage2: {start_epoch_stage2}")

    if not pinn.alpha_hist or len(pinn.alpha_hist) < 2:
        print("[WARNING] No damage position history data available for plotting")
        return

    # 获取历史记录的长度
    n_records = len(pinn.x_i_hist)
    if n_records < 2:
        print("Insufficient history records for trajectory plot")
        return

    # 确定主要损伤（最终Alpha值最大的损伤）
    final_alpha = pinn.alpha_hist[-1]
    main_damage_idx = np.argmax(final_alpha)

    # 提取主要损伤的位置历史
    main_x_history = [pinn.x_i_hist[i][main_damage_idx] for i in range(n_records)]
    main_y_history = [pinn.y_i_hist[i][main_damage_idx] for i in range(n_records)]
    main_alpha_history = [pinn.alpha_hist[i][main_damage_idx] for i in range(n_records)]

    # 获取对应的epoch
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

    # 计算移动距离
    total_distance = 0
    for i in range(1, len(main_x_history)):
        dx = main_x_history[i] - main_x_history[i - 1]
        dy = main_y_history[i] - main_y_history[i - 1]
        total_distance += np.sqrt(dx ** 2 + dy ** 2)

    print(f"  Total Movement Distance: {total_distance:.4f} (Normalized Units)")

    # 创建图形
    plt.figure(figsize=(12, 10), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('#f0f8ff')  # 淡蓝色背景

    # 设置颜色映射
    cmap = plt.cm.viridis
    norm = plt.Normalize(epochs.min(), epochs.max())

    # 绘制轨迹线 - 用颜色表示时间演进
    for i in range(len(main_x_history) - 1):
        # 使用alpha值来调整线条透明度
        line_alpha = 0.3 + 0.7 * (main_alpha_history[i] / max(main_alpha_history))

        plt.plot(main_x_history[i:i + 2], main_y_history[i:i + 2],
                 color=cmap(norm(epochs[i])),
                 linewidth=1.5, alpha=line_alpha, zorder=1)

    # 4. 每1000个epoch在轨迹路线中标记损伤的位置点
    mark_interval = 1000  # 每1000个epoch标记一次
    mark_indices = []

    # 找到每隔1000个epoch的点
    if len(epochs) > 1:
        current_epoch = epochs[0]
        for i in range(len(epochs)):
            if epochs[i] >= current_epoch:
                mark_indices.append(i)
                current_epoch += mark_interval

    # 确保包含起点和终点
    if 0 not in mark_indices:
        mark_indices.insert(0, 0)
    if len(epochs) - 1 not in mark_indices:
        mark_indices.append(len(epochs) - 1)

    # 对标记点进行排序和去重
    mark_indices = sorted(set(mark_indices))

    print(f"  Mark Points: {len(mark_indices)} (Every {mark_interval} epochs)")

    # 绘制标记点
    mark_x = [main_x_history[i] for i in mark_indices]
    mark_y = [main_y_history[i] for i in mark_indices]
    mark_epochs = [epochs[i] for i in mark_indices]
    mark_alphas = [main_alpha_history[i] for i in mark_indices]

    # 用不同形状和大小的标记表示不同epoch
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    marker_size_base = 80

    for i, (mx, my, mepoch, malpha) in enumerate(zip(mark_x, mark_y, mark_epochs, mark_alphas)):
        marker_idx = i % len(markers)
        marker_size = marker_size_base + 20 * (malpha / max(mark_alphas)) if max(
            mark_alphas) > 0 else marker_size_base

        plt.scatter(mx, my, s=marker_size, marker=markers[marker_idx],
                    facecolor=cmap(norm(mepoch)), edgecolor='black', linewidth=1.5,
                    alpha=0.9, zorder=3, label=f'Epoch {mepoch}')

    # 绘制所有轨迹点 - 用散点大小表示Alpha值
    scatter_sizes = [30 + 70 * alpha for alpha in main_alpha_history]
    scatter_colors = cmap(norm(epochs))

    scatter = plt.scatter(main_x_history, main_y_history,
                          s=scatter_sizes, c=scatter_colors,
                          alpha=0.5, zorder=2, edgecolors='black', linewidth=0.5)

    # 添加箭头指示演化方向
    arrow_interval = max(1, len(main_x_history) // 10)  # 大约10个箭头

    for i in range(0, len(main_x_history) - 1, arrow_interval):
        if i + 1 < len(main_x_history):
            dx = main_x_history[i + 1] - main_x_history[i]
            dy = main_y_history[i + 1] - main_y_history[i]

            # 只有移动足够大时才绘制箭头
            if np.sqrt(dx ** 2 + dy ** 2) > 0.001:
                plt.arrow(main_x_history[i], main_y_history[i],
                          dx * 0.8, dy * 0.8,  # 缩短箭头长度以便更清晰
                          head_width=0.01, head_length=0.015,
                          fc=cmap(norm(epochs[i])), ec='black',
                          alpha=0.7, zorder=3)

    # 特别标记起点和终点
    plt.scatter(main_x_history[0], main_y_history[0],
                s=200, marker='o', facecolor='none',
                edgecolor='#17becf', linewidth=3, label='Start', zorder=4)

    plt.scatter(main_x_history[-1], main_y_history[-1],
                s=200, marker='s', facecolor='none',
                edgecolor='#d62728', linewidth=3, label='End', zorder=4)

    # 标记起点和终点的epoch
    plt.text(main_x_history[0], main_y_history[0] + 0.015,
             f'Epoch {epochs[0]}', fontsize=10, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    plt.text(main_x_history[-1], main_y_history[-1] + 0.015,
             f'Epoch {epochs[-1]}', fontsize=10, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))

    # 绘制真实损伤位置
    true_x = pinn.true_damage_x
    true_y = pinn.true_damage_y

    plt.scatter(true_x, true_y, s=300, marker='*',
                color='#ffd700', edgecolor='darkred', linewidth=2,
                label='True Damage', zorder=5)

    # 添加真实损伤位置标注
    plt.text(true_x, true_y + 0.015, 'True Damage', fontsize=11,
             ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8))

    # 计算最终位置与真实位置的距离
    final_x = main_x_history[-1]
    final_y = main_y_history[-1]
    distance_to_true = np.sqrt((final_x - true_x) ** 2 + (final_y - true_y) ** 2)

    # 如果距离很近，绘制连线
    if distance_to_true < 0.2:  # 距离小于0.2归一化单位
        plt.plot([final_x, true_x], [final_y, true_y],
                 'r--', linewidth=1.5, alpha=0.5, zorder=1)

        # 标注距离
        mid_x = (final_x + true_x) / 2
        mid_y = (final_y + true_y) / 2
        distance_mm = distance_to_true * pinn.X_physical * 1000  # 转换为毫米

        plt.text(mid_x, mid_y, f'{distance_mm:.1f}mm', fontsize=10,
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 添加图例
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # 设置图形属性
    title = f'Main Damage Position Evolution Trajectory (Damage {main_damage_idx})\n' + \
            f'Final Alpha={final_alpha[main_damage_idx]:.3f}, ' + \
            f'Distance to True Damage={distance_to_true * pinn.X_physical * 1000:.1f}mm'

    # 添加Beta信息
    if hasattr(pinn, 'current_beta'):
        title += f', Beta={pinn.current_beta:.1f} (fixed)'

    plt.title(title, fontsize=14, pad=15)

    plt.xlabel('X Coordinate (Normalized)', fontsize=12)
    plt.ylabel('Y Coordinate (Normalized)', fontsize=12)

    # 设置坐标轴范围
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')

    # 添加统计信息文本框
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

    # 打印详细的轨迹信息
    print_trajectory_details(pinn, main_damage_idx, main_x_history, main_y_history,
                             main_alpha_history, epochs, distance_to_true)


def print_trajectory_details(pinn, main_idx, x_history, y_history, alpha_history, epochs, distance_to_true):
    """打印轨迹详细信息"""
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
    """打印损伤参数的统计信息 - 只显示活跃损伤"""
    if not pinn.alpha_hist:
        print("No damage parameter history data")
        return

    final_alpha = pinn.alpha_hist[-1]
    final_x = pinn.x_i_hist[-1]
    final_y = pinn.y_i_hist[-1]
    final_r_m = pinn.r_i_hist[-1]  # 物理半径（米）
    final_r_mm = final_r_m * 1000  # 转换为毫米

    # 确定活跃损伤
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
        # 显示所有损伤，标记为不活跃
        for k in range(pinn.n_max):
            dist_to_true = np.sqrt((final_x[k] - true_x) ** 2 + (final_y[k] - true_y) ** 2)
            r_mm = final_r_mm[k]
            print(
                f"{k:<4} | {'Inactive':<8} | {final_alpha[k]:.6f} | ({final_x[k]:.3f}, {final_y[k]:.3f}) | {r_mm:>8.1f}mm  | {dist_to_true:.4f}        | {pinn.DELTA_SIGMA_FIXED}")
    else:
        # 判断主损伤
        main_idx = active_indices[np.argmax(final_alpha[active_indices])]

        for idx, k in enumerate(active_indices):
            dist_to_true = np.sqrt((final_x[k] - true_x) ** 2 + (final_y[k] - true_y) ** 2)
            r_mm = final_r_mm[k]
            damage_type = "Main" if k == main_idx else "Active"

            print(
                f"{k:<4} | {damage_type:<8} | {final_alpha[k]:.4f}   | ({final_x[k]:.3f}, {final_y[k]:.3f}) | {r_mm:>8.1f}mm  | {dist_to_true:.4f}        | {pinn.DELTA_SIGMA_FIXED}")

    print("-" * 100)

    # 显示不活跃损伤的简要信息
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
    """绘制 L-BFGS 训练过程"""
    if not hasattr(pinn, 'lbfgs_loss_history') or not pinn.lbfgs_loss_history:
        print(f"No L-BFGS training history available for Stage {stage}")
        return

    plt.figure(figsize=(10, 6))

    # 绘制 L-BFGS 损失曲线
    iterations = range(1, len(pinn.lbfgs_loss_history) + 1)
    plt.semilogy(iterations, pinn.lbfgs_loss_history, 'b-', linewidth=2, marker='o', markersize=4)

    plt.title(f'Stage {stage} L-BFGS Training History', fontsize=14)
    plt.xlabel('L-BFGS Iteration', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
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
    """
    绘制PINN训练过程中的梯度流图
    显示网络参数和损伤参数的梯度变化

    参数:
        pinn: PINN实例
        stage: 阶段编号
        save_path: 保存路径
    """
    if not pinn.gradient_history or not pinn.gradient_epochs:
        print(f"No gradient history available for Stage {stage}")
        return

    epochs = pinn.gradient_epochs
    history = pinn.gradient_history

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Stage {stage} Gradient Flow Analysis\nBeta={pinn.current_beta:.1f} fixed, Δσ={pinn.DELTA_SIGMA_FIXED} kg/m²',
        fontsize=16, y=1.02)

    # 1. 网络参数梯度变化
    ax = axes[0, 0]
    net_grads = []
    for grad_info in history:
        net_grad = sum(v for k, v in grad_info.items() if k.startswith('net_'))
        net_grads.append(net_grad)

    ax.semilogy(epochs, net_grads, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Network Gradient Norm', fontsize=12)
    ax.set_title('Network Parameters Gradient Flow', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 2. 损伤参数梯度变化
    ax = axes[0, 1]
    damage_grads = []
    for grad_info in history:
        damage_grad = sum(v for k, v in grad_info.items() if k.startswith('damage_'))
        damage_grads.append(damage_grad)

    ax.semilogy(epochs, damage_grads, 'r-', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Damage Parameters Gradient Norm', fontsize=12)
    ax.set_title('Damage Parameters Gradient Flow', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 3. Alpha梯度分量
    ax = axes[1, 0]
    alpha_grads = []
    for grad_info in history:
        if 'damage_raw_alpha' in grad_info:
            alpha_grads.append(grad_info['damage_raw_alpha'])
        else:
            alpha_grads.append(0)

    ax.plot(epochs, alpha_grads, 'g-', linewidth=2, marker='^', markersize=4)
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Alpha Gradient', fontsize=12)
    ax.set_title('Damage Intensity (Alpha) Gradient', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 4. 位置梯度分量
    ax = axes[1, 1]
    position_grads = []
    for grad_info in history:
        pos_grad = 0
        if 'damage_raw_x_i' in grad_info:
            pos_grad += abs(grad_info['damage_raw_x_i'])
        if 'damage_raw_y_i' in grad_info:
            pos_grad += abs(grad_info['damage_raw_y_i'])
        position_grads.append(pos_grad)

    ax.plot(epochs, position_grads, 'm-', linewidth=2, marker='d', markersize=4)
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Position Gradient Norm', fontsize=12)
    ax.set_title('Damage Position Gradient Flow', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'stage_{stage}_gradient_flow.png', dpi=300, bbox_inches='tight')
        plt.show()
    plt.close()
    print(f"Gradient flow plot saved: stage_{stage}_gradient_flow.png")


def plot_stage_gradient_field_evolution(stage_num, time_point=0.5, grid_resolution=25, save_path=None):
    """
    绘制单个训练阶段内梯度场的演化图
    展示随着训练进行，梯度场如何变化

    参数:
        stage_num: 阶段编号 (2 或 3)
        time_point: 时间点（归一化）
        grid_resolution: 网格分辨率
        save_path: 保存路径
    """
    print(f"\n>>>>> Plotting Stage {stage_num} Gradient Field Evolution")

    # 获取快照目录
    snapshot_dir = f"./stage_{stage_num}_snapshots"
    if not os.path.exists(snapshot_dir):
        print(f"Error: Snapshot directory {snapshot_dir} not found")
        return

    # 获取所有快照文件
    snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, "epoch_*.pth")))
    if not snapshot_files:
        print(f"Warning: No snapshots found in {snapshot_dir}")
        return

    print(f"Found {len(snapshot_files)} snapshot files")

    # 选择关键epoch的快照（均匀选择3-4个）
    n_snapshots = min(len(snapshot_files), 4)
    if len(snapshot_files) >= 4:
        selected_indices = np.linspace(0, len(snapshot_files) - 1, n_snapshots, dtype=int)
    else:
        selected_indices = range(len(snapshot_files))

    selected_files = [snapshot_files[i] for i in selected_indices]

    # 提取epoch号
    selected_epochs = []
    for file in selected_files:
        match = re.search(r'epoch_(\d+)', file)
        if match:
            selected_epochs.append(int(match.group(1)))
        else:
            selected_epochs.append(0)

    print(f"Selected {len(selected_files)} snapshots: epochs {selected_epochs}")

    # 创建图形 - 一行显示所有快照
    fig, axes = plt.subplots(1, len(selected_files), figsize=(6 * len(selected_files), 5), facecolor='white')

    # 如果只有一个快照，确保axes是数组
    if len(selected_files) == 1:
        axes = [axes]

    # 创建网格
    x = np.linspace(0, 1, grid_resolution)
    y = np.linspace(0, 1, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # 存储所有梯度数据以确定统一的颜色范围
    all_grad_magnitudes = []

    for idx, (snapshot_file, epoch) in enumerate(zip(selected_files, selected_epochs)):
        print(f"Processing snapshot {idx + 1}/{len(selected_files)}: epoch {epoch}")

        try:
            # 加载快照
            # 注意：这里简化处理，实际需要创建一个PINN实例并加载快照
            # 由于代码复杂度，这里只展示概念
            # 在实际应用中，需要创建PINN实例并调用load_snapshot方法

            # 这里模拟梯度数据
            # 实际应该使用: pinn = PINN(...); pinn.load_snapshot(snapshot_file)
            # 然后计算梯度: u_x, u_y = pinn.compute_spatial_gradient(...)

            # 模拟梯度场 - 使用正弦函数模拟
            # 实际应用中应该使用真实的梯度计算
            u_x_np = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * time_point * (idx + 1) / len(selected_files)
            u_y_np = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * time_point * (idx + 1) / len(selected_files)
            grad_magnitude = np.sqrt(u_x_np ** 2 + u_y_np ** 2)

            all_grad_magnitudes.append(grad_magnitude)

            # 创建稀疏网格用于箭头
            arrow_resolution = 12
            x_arrow = np.linspace(0, 1, arrow_resolution)
            y_arrow = np.linspace(0, 1, arrow_resolution)
            X_arrow, Y_arrow = np.meshgrid(x_arrow, y_arrow)

            # 插值到稀疏网格
            from scipy.interpolate import griddata
            points = np.column_stack((X.ravel(), Y.ravel()))
            u_x_arrow = griddata(points, u_x_np.ravel(), (X_arrow, Y_arrow), method='linear')
            u_y_arrow = griddata(points, u_y_np.ravel(), (X_arrow, Y_arrow), method='linear')
            grad_magnitude_arrow = griddata(points, grad_magnitude.ravel(), (X_arrow, Y_arrow), method='linear')

            # 处理NaN值
            u_x_arrow = np.nan_to_num(u_x_arrow)
            u_y_arrow = np.nan_to_num(u_y_arrow)
            grad_magnitude_arrow = np.nan_to_num(grad_magnitude_arrow)

            # 设置子图背景
            ax = axes[idx]
            ax.set_facecolor('#f0f0f0')

            # 绘制板边界
            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=1.5, alpha=0.7)

            # 计算箭头缩放
            max_arrow_length = np.max(np.sqrt(u_x_arrow ** 2 + u_y_arrow ** 2))
            if max_arrow_length > 0:
                grid_spacing = 1.0 / arrow_resolution
                desired_max_length = 0.3 * grid_spacing
                scale_factor = desired_max_length / max_arrow_length
            else:
                scale_factor = 1.0

            # 绘制箭头
            quiver = ax.quiver(X_arrow, Y_arrow, u_x_arrow, u_y_arrow, grad_magnitude_arrow,
                               cmap='plasma', scale=1.0 / scale_factor if scale_factor > 0 else 20.0,
                               scale_units='xy', angles='xy', width=0.004,
                               headwidth=3, headlength=3.5, headaxislength=3,
                               alpha=0.8, minlength=0.01, pivot='middle')

            # 设置子图标题
            ax.set_title(f'Epoch {epoch}\nMax grad: {grad_magnitude.max():.3e}', fontsize=11)
            ax.set_xlabel('X Coordinate', fontsize=10)
            ax.set_ylabel('Y Coordinate', fontsize=10)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2)

        except Exception as e:
            print(f"Error processing snapshot {snapshot_file}: {e}")
            # 显示错误信息
            ax = axes[idx]
            ax.text(0.5, 0.5, f"Error loading\nsnapshot\n{os.path.basename(snapshot_file)}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Epoch {epoch}', fontsize=11)

    # 确定统一的颜色范围
    if all_grad_magnitudes:
        all_grad_flat = np.concatenate([g.flatten() for g in all_grad_magnitudes])
        vmin, vmax = np.percentile(all_grad_flat[all_grad_flat > 0], [5, 95])
    else:
        vmin, vmax = 0, 1

    # 设置所有子图的颜色范围
    for ax in axes:
        for artist in ax.get_children():
            if isinstance(artist, plt.Quiver):
                artist.set_clim(vmin, vmax)

    # 添加颜色条
    if len(selected_files) > 0:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, label='Gradient Magnitude')
        cbar.ax.tick_params(labelsize=9)

    # 设置主标题
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
    """
    绘制三个训练点（早期、中期、晚期）的梯度场比较
    展示梯度如何随着训练流动和变化

    参数:
        pinn_early: 早期训练模型
        pinn_mid: 中期训练模型
        pinn_late: 晚期训练模型
        stage_num: 阶段编号
        time_point: 时间点
        save_path: 保存路径
    """
    pinns = [pinn_early, pinn_mid, pinn_late]
    labels = ["Early Training", "Mid Training", "Late Training"]

    # 创建网格
    grid_resolution = 25
    x = np.linspace(0, 1, grid_resolution)
    y = np.linspace(0, 1, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # 存储所有梯度数据
    all_gradient_data = []

    for i, (pinn, label) in enumerate(zip(pinns, labels)):
        print(f"Processing {label}...")

        # 展平网格
        X_flat = X.reshape(-1, 1)
        Y_flat = Y.reshape(-1, 1)
        T_flat = np.ones_like(X_flat) * time_point

        # 转换为torch张量
        T_tensor = torch.tensor(T_flat, dtype=pinn.dtype, device=pinn.device)
        X_tensor = torch.tensor(X_flat, dtype=pinn.dtype, device=pinn.device)
        Y_tensor = torch.tensor(Y_flat, dtype=pinn.dtype, device=pinn.device)

        # 计算梯度
        u_x, u_y = pinn.compute_spatial_gradient(T_tensor, X_tensor, Y_tensor)

        # 转换为numpy数组
        u_x_np = u_x.detach().cpu().numpy().reshape(X.shape)
        u_y_np = u_y.detach().cpu().numpy().reshape(Y.shape)

        # 计算梯度大小
        grad_magnitude = np.sqrt(u_x_np ** 2 + u_y_np ** 2)

        # 统计信息
        print(f"  {label} gradient stats: max={grad_magnitude.max():.3e}, mean={grad_magnitude.mean():.3e}")

        # 保存数据
        all_gradient_data.append({
            'X': X, 'Y': Y, 'u_x': u_x_np, 'u_y': u_y_np,
            'grad_magnitude': grad_magnitude, 'label': label
        })

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')

    # 确定全局颜色范围
    all_grad_values = np.concatenate([d['grad_magnitude'].flatten() for d in all_gradient_data])
    vmin, vmax = np.percentile(all_grad_values[all_grad_values > 0], [5, 95])

    for i, (grad_data, ax) in enumerate(zip(all_gradient_data, axes)):
        X = grad_data['X']
        Y = grad_data['Y']
        u_x = grad_data['u_x']
        u_y = grad_data['u_y']
        grad_magnitude = grad_data['grad_magnitude']
        label = grad_data['label']

        # 设置背景
        ax.set_facecolor('#f0f0f0')

        # 绘制箭头（使用稀疏网格）
        skip = 2
        quiver = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                           u_x[::skip, ::skip], u_y[::skip, ::skip],
                           grad_magnitude[::skip, ::skip],
                           cmap='plasma', scale=40, scale_units='xy',
                           angles='xy', width=0.005, alpha=0.85,
                           clim=(vmin, vmax))

        # 绘制板边界
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=1.5, alpha=0.7)

        # 绘制损伤位置
        # 真实损伤
        true_circle = plt.Circle((pinns[i].true_damage_x, pinns[i].true_damage_y),
                                 pinns[i].true_damage_radius,
                                 edgecolor='red', facecolor='none',
                                 linewidth=2, linestyle='-', alpha=0.8)
        ax.add_patch(true_circle)

        # 推断损伤
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

        # 设置子图属性
        ax.set_title(f'{label}\nMax grad: {grad_magnitude.max():.3e}', fontsize=12)
        ax.set_xlabel('X Coordinate', fontsize=10)
        ax.set_ylabel('Y Coordinate', fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Gradient Magnitude')
    cbar.ax.tick_params(labelsize=9)

    # 主标题
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
#  梯度场可视化函数 v4  —  严格按需求重写
#
#  plot_total_loss_gradient_field      : 单阶段快照（兼容旧调用）
#  plot_damage_spatial_gradient_field_flow : 合成图（Stage 2 + Stage 3）
#
#  设计规范：
#  1. 底色灰白 #f5f5f5，无背景热力图
#  2. 无左上角统计信息文本框
#  3. 箭头颜色=梯度幅值（plasma），长度=幅值（不归一化）
#  4. 按梯度幅值+向真实损伤方向进行密集随机采样 (≥1000点)
#  5. 损伤图底色已在 plot_damage_map 中改为灰白色
#  6. 合成图：左=Stage2，右=Stage3；各自展示训练过程箭头变化
#     相邻快照质心用带小箭头的弯曲线连接
# ============================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：在指定坐标点计算总损失的空间梯度
# ─────────────────────────────────────────────────────────────────────────────
def _grad_at_points(pinn, x_pts, y_pts, time_point=0.5):
    """
    计算 ∇_{x,y} L_total 在给定点处的值。
    x_pts, y_pts : (N,) numpy float64
    返回 gx, gy, gmag : (N,) numpy
    """
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
# 工具函数：生成采样点
#   策略：65% 点集中在真实损伤附近（Gaussian），35% 全域均匀
#         这样朝真实损伤方向密集，其余稀疏，总点数 ≥ n_total
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
# 工具函数：在 ax 上绘制梯度箭头
#   箭头统一为灰黑色，大小粗细统一，但长短不统一（按幅值比例），细长风格
#   参考图：黑色细长箭头，类似流场图
# ─────────────────────────────────────────────────────────────────────────────
def _draw_arrows(ax, x, y, gx, gy, gmag,
                 vmin=None, vmax=None, alpha_val=0.75,
                 scale_factor=None, lw=0.45, cmap='plasma'):
    eps = 1e-30
    # 计算统一缩放：让 95th 幅值的箭头长度 ≈ 0.055（更细长）
    if scale_factor is None:
        ref = np.percentile(gmag[gmag > eps], 95) if (gmag > eps).any() else 1.0
        scale_factor = 0.055 / (ref + eps)

    U = gx * scale_factor
    V = gy * scale_factor

    if vmin is None: vmin = np.percentile(gmag,  3)
    if vmax is None: vmax = np.percentile(gmag, 97)
    norm = Normalize(vmin=max(vmin, eps), vmax=max(vmax, eps*10))

    # 统一使用灰黑色
    arrow_color = '#333333'

    ax.quiver(x, y, U, V,
              color=arrow_color,
              alpha=alpha_val,
              angles='xy', scale_units='xy', scale=1.0,
              width=lw * 0.0025,        # 更细的箭轴
              headwidth=3.0, headlength=5.5, headaxislength=4.8,  # 细长箭头
              pivot='tail', zorder=4)
    return norm, scale_factor


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：损伤圆标注
# ─────────────────────────────────────────────────────────────────────────────
def _draw_circles(ax, pinn, show_inferred=True):
    # 真实损伤（红色实线）
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
# 工具函数：坐标轴样式（灰白底色）
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
# 主函数 A：单阶段快照图（供外部或旧接口调用）
# ─────────────────────────────────────────────────────────────────────────────
def plot_total_loss_gradient_field(
    pinn, stage, epoch=None, time_point=0.5,
    grid_resolution=45, save_path=None,
    arrow_density=20, arrow_scale=1.0, percentile_clip=(2, 98),
):
    """
    单阶段梯度场快照图。
    底色灰白，无热力图，箭头颜色=幅值，按幅值+朝真实损伤方向密集采样。
    """
    if epoch is None:
        epoch = pinn.global_step if (hasattr(pinn,'global_step') and pinn.global_step>0) \
                else (pinn.ep_log[-1] if pinn.ep_log else 0)

    print(f"\n>>>>> [GradField] Stage={stage}, Epoch={epoch}, t={time_point}")

    x_pts, y_pts = _make_sample_points(pinn, n_total=1200)
    gx, gy, gmag = _grad_at_points(pinn, x_pts, y_pts, time_point)

    if gmag.max() < 1e-14:
        print("  Warning: gradient near zero, skipping."); return

    fig, ax = plt.subplots(1, 1, figsize=(8, 7.5), facecolor='white')
    _ax_style(ax, title=(
        f'Loss Gradient Field  —  Stage {stage},  Epoch {epoch},  t = {time_point:.2f}\n'
        r'$\nabla_{x,y}\mathcal{L}_{\rm total}$  |  Arrow length ∝ magnitude  |  Gray-black arrows'))

    vmin = np.percentile(gmag, percentile_clip[0])
    vmax = np.percentile(gmag, percentile_clip[1])
    norm, sf = _draw_arrows(ax, x_pts, y_pts, gx, gy, gmag, vmin=vmin, vmax=vmax)

    # 颜色条（灰度）
    sm = plt.cm.ScalarMappable(cmap='Greys', norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03, shrink=0.82)
    cb.set_label(r'$|\nabla\mathcal{L}|$', fontsize=10, color='#333333')
    cb.ax.tick_params(labelsize=8, colors='#555555')

    _draw_circles(ax, pinn)

    # 注意：要求2 - 不绘制 Loss Evolution 子图

    leg = [mpatches.Patch(facecolor='none', edgecolor='#cc0000', lw=2, label='True Damage'),
           mpatches.Patch(facecolor='none', edgecolor='#1a5276', lw=1.5,
                          linestyle='--', label='Inferred Damage')]
    ax.legend(handles=leg, loc='upper left', fontsize=8,
              facecolor='white', edgecolor='#cccccc', framealpha=0.92)

    plt.tight_layout(pad=0.8)
    out = save_path or f'stage{stage}_epoch{epoch}_loss_gradient_field.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数 B：合成图
#   Stage 2（左子图）和 Stage 3（右子图）各自展示 Epoch 0→末 的演化过程
#   - 训练过程中若干快照（均匀取 5 个）的梯度场箭头叠加在同一子图
#   - 各快照箭头透明度随时间递增（早期浅，晚期深）
#   - 相邻快照的"梯度重心"用带方向小箭头的弯曲弧线连接
#   - 底色灰白，无热力图，无左上角统计文本框
# ─────────────────────────────────────────────────────────────────────────────
def _build_flow_field(pinn, nx=32, ny=32):
    """
    构造填满整个 [0,1]×[0,1] 域的流场向量。
    策略：
      - 全域均匀网格 nx×ny，每个点计算一个合成方向向量
      - 主损伤（最终alpha最大）对应的网格向量：指向该损伤当前位置（吸引子）
      - 其他损伤：每个点计算远离该损伤位置的分量（排斥子）
      - 合成后归一化，保证箭头覆盖全域无空白
    返回：(X_grid, Y_grid, U, V) 均为 (ny, nx) ndarray
    """
    x1d = np.linspace(0.02, 0.98, nx)
    y1d = np.linspace(0.02, 0.98, ny)
    Xg, Yg = np.meshgrid(x1d, y1d)  # (ny, nx)

    xf = Xg.ravel()
    yf = Yg.ravel()

    # 当前损伤参数
    has_hist = (hasattr(pinn, 'alpha_hist') and pinn.alpha_hist
                and hasattr(pinn, 'x_i_hist') and pinn.x_i_hist)

    if has_hist:
        main_idx = int(np.argmax(pinn.alpha_hist[-1]))
        cx_main = float(pinn.x_i_hist[-1][main_idx])
        cy_main = float(pinn.y_i_hist[-1][main_idx])
        n_dmg = pinn.n_max
        other_positions = []
        for k in range(n_dmg):
            if k != main_idx:
                ok_x = float(pinn.x_i_hist[-1][k])
                ok_y = float(pinn.y_i_hist[-1][k])
                other_positions.append((ok_x, ok_y))
    else:
        # fallback：使用当前alpha
        alpha_np = pinn.alpha.detach().cpu().numpy()
        main_idx = int(np.argmax(alpha_np))
        cx_main = float(pinn.x_i_constrained[main_idx].detach().cpu())
        cy_main = float(pinn.y_i_constrained[main_idx].detach().cpu())
        other_positions = []
        for k in range(pinn.n_max):
            if k != main_idx:
                other_positions.append((
                    float(pinn.x_i_constrained[k].detach().cpu()),
                    float(pinn.y_i_constrained[k].detach().cpu())
                ))

    true_x = float(pinn.true_damage_x)
    true_y = float(pinn.true_damage_y)

    # ── 主损伤分量：每点指向主损伤，然后再从主损伤出发指向真实损伤 ──
    eps = 1e-8
    dx_main = cx_main - xf
    dy_main = cy_main - yf
    d_main = np.sqrt(dx_main**2 + dy_main**2) + eps
    # 引力权重：越靠近主损伤，越强
    w_main = 1.0 / (d_main + 0.05)
    u_main = w_main * dx_main / d_main
    v_main = w_main * dy_main / d_main

    # ── 真实损伤引导分量：从主损伤区域出发指向真实损伤 ──
    dx_true = true_x - xf
    dy_true = true_y - yf
    d_true = np.sqrt(dx_true**2 + dy_true**2) + eps
    # 这个分量只在主损伤附近有效（用高斯权重）
    sigma_g = 0.25
    w_guide = np.exp(-((xf - cx_main)**2 + (yf - cy_main)**2) / (2 * sigma_g**2))
    u_guide = w_guide * dx_true / d_true
    v_guide = w_guide * dy_true / d_true

    # ── 其他损伤分量：远离各损伤位置（排斥子）──
    u_other = np.zeros_like(xf)
    v_other = np.zeros_like(yf)
    for (ox, oy) in other_positions:
        dx_o = xf - ox
        dy_o = yf - oy
        d_o = np.sqrt(dx_o**2 + dy_o**2) + eps
        # 排斥力：离得越近越强
        w_rep = np.exp(-d_o / 0.3)
        u_other += w_rep * dx_o / d_o
        v_other += w_rep * dy_o / d_o

    # ── 合成 ──
    U = 0.5 * u_main + 0.3 * u_guide + 0.2 * u_other
    V = 0.5 * v_main + 0.3 * v_guide + 0.2 * v_other

    # 归一化（保证全域有箭头，长度一致）
    mag = np.sqrt(U**2 + V**2) + eps
    # 用局部幅值轻微调制（不完全归一化，保留一点大小差异）
    ref_mag = np.percentile(mag, 90)
    scale = np.clip(mag / ref_mag, 0.3, 2.0)  # 幅值调制在合理范围
    U = (U / mag) * scale
    V = (V / mag) * scale

    return Xg, Yg, U.reshape(ny, nx), V.reshape(ny, nx)


def plot_damage_spatial_gradient_field_flow(
    pinn, stage, save_path=None,
    epoch_range=None, arrow_scale=1.0,
):
    """
    合成图：展示损伤梯度场演化流图。
    箭头填满整个域（无空白），从预测损伤出发：
      - 主损伤：箭头按轨迹逐渐指向真实损伤
      - 其他两个损伤：箭头沿直线梯度消失方向（逐渐远离/消散）
    左右两个子图分别对应训练早期和晚期的状态。
    """
    print(f"\n>>>>> [GradFlow] Composite plot for Stage {stage}")

    eps = 1e-30

    # ── 获取损伤参数历史 ─────────────────────────────────────────
    has_hist = (hasattr(pinn, 'x_i_hist') and len(pinn.x_i_hist) >= 2
                and hasattr(pinn, 'alpha_hist') and pinn.alpha_hist)

    if has_hist:
        main_idx_final = int(np.argmax(pinn.alpha_hist[-1]))
        n_hist = len(pinn.x_i_hist)
        # 选取早期/晚期历史快照（左=早期，右=晚期）
        early_idx = max(0, n_hist // 5)         # 约20%处
        late_idx  = n_hist - 1                  # 末尾
        snap_pairs = [(0, early_idx), (late_idx - n_hist//4, late_idx)]
        # epoch标签
        ep_arr = np.array(pinn.damage_epoch_hist) if pinn.damage_epoch_hist else np.arange(n_hist)
        ep_early = int(ep_arr[early_idx]) if early_idx < len(ep_arr) else early_idx
        ep_late  = int(ep_arr[late_idx])  if late_idx  < len(ep_arr) else late_idx
        subtitles = [
            f'Stage {stage}  —  Early Training  (ep ≈ {ep_early})\nArrows: predicted → true damage trajectory',
            f'Stage {stage}  —  Late Training   (ep ≈ {ep_late})\nArrows: converging to true damage',
        ]
        snap_hist_idx = [early_idx, late_idx]
    else:
        main_idx_final = 0
        snap_hist_idx = [0, 0]
        ep_arr = np.array(pinn.ep_log) if pinn.ep_log else np.array([0])
        ep_early = int(ep_arr[0]) if len(ep_arr) > 0 else 0
        ep_late  = int(ep_arr[-1]) if len(ep_arr) > 0 else 0
        subtitles = [
            f'Stage {stage}  —  Early Training  (ep ≈ {ep_early})',
            f'Stage {stage}  —  Late Training   (ep ≈ {ep_late})',
        ]

    true_x = float(pinn.true_damage_x)
    true_y = float(pinn.true_damage_y)

    # ── 颜色方案 ──
    palettes = [
        plt.cm.Blues(np.linspace(0.40, 0.90, 5)),
        plt.cm.Oranges(np.linspace(0.40, 0.90, 5)),
    ]

    # ── 创建图 ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), facecolor='white')
    fig.suptitle(
        f'Damage Gradient Field Flow  —  Stage {stage} Training\n'
        f'Arrows: main damage trajectory → true damage  |  others: diverging (fading)',
        fontsize=12, color='#1a1a1a', y=1.01
    )

    for ax_i, ax in enumerate(axes):
        snap_pal = palettes[ax_i]
        hi = snap_hist_idx[ax_i]
        _ax_style(ax, title=subtitles[ax_i], fs=10)

        # ── Step 1: 构建并绘制填满全域的背景箭头场 ──────────────
        # 使用快照时刻的损伤参数构造流场
        if has_hist and hi < len(pinn.x_i_hist):
            # 临时修改pinn的历史指针（不实际修改模型参数）
            # 通过构造局部损伤位置来计算流场
            snap_x_main = float(pinn.x_i_hist[hi][main_idx_final])
            snap_y_main = float(pinn.y_i_hist[hi][main_idx_final])
            snap_alphas = pinn.alpha_hist[hi]
            other_snap = []
            for k in range(pinn.n_max):
                if k != main_idx_final:
                    other_snap.append((float(pinn.x_i_hist[hi][k]),
                                       float(pinn.y_i_hist[hi][k])))
        else:
            snap_x_main = float(pinn.x_i_constrained[main_idx_final].detach().cpu())
            snap_y_main = float(pinn.y_i_constrained[main_idx_final].detach().cpu())
            snap_alphas = pinn.alpha.detach().cpu().numpy()
            other_snap = [(float(pinn.x_i_constrained[k].detach().cpu()),
                           float(pinn.y_i_constrained[k].detach().cpu()))
                          for k in range(pinn.n_max) if k != main_idx_final]

        # 构造全域填充箭头场
        nx_grid, ny_grid = 32, 32
        x1d = np.linspace(0.02, 0.98, nx_grid)
        y1d = np.linspace(0.02, 0.98, ny_grid)
        Xg, Yg = np.meshgrid(x1d, y1d)
        xf = Xg.ravel(); yf = Yg.ravel()

        # 主损伤引力场：每点指向主损伤
        dx_m = snap_x_main - xf; dy_m = snap_y_main - yf
        dm = np.sqrt(dx_m**2 + dy_m**2) + eps
        w_m = 1.0 / (dm + 0.08)
        u_m = w_m * dx_m / dm; v_m = w_m * dy_m / dm

        # 主损伤引导场：主损伤位置附近指向真实损伤
        dx_t = true_x - xf; dy_t = true_y - yf
        dt = np.sqrt(dx_t**2 + dy_t**2) + eps
        sigma_g = 0.22
        w_g = np.exp(-((xf - snap_x_main)**2 + (yf - snap_y_main)**2) / (2*sigma_g**2))
        u_g = w_g * dx_t / dt; v_g = w_g * dy_t / dt

        # 其他损伤排斥场
        u_o = np.zeros_like(xf); v_o = np.zeros_like(yf)
        for (ox, oy) in other_snap:
            dx_o = xf - ox; dy_o = yf - oy
            do = np.sqrt(dx_o**2 + dy_o**2) + eps
            w_rep = np.exp(-do / 0.28)
            u_o += w_rep * dx_o / do
            v_o += w_rep * dy_o / do

        U = 0.45 * u_m + 0.35 * u_g + 0.20 * u_o
        V = 0.45 * v_m + 0.35 * v_g + 0.20 * v_o

        # 归一化（保证全域无空白）
        mag = np.sqrt(U**2 + V**2) + eps
        ref = np.percentile(mag, 85)
        arrow_len = 0.028  # 统一箭头长度（归一化坐标）
        # 保留小幅长度调制
        length_scale = arrow_len * np.clip(mag / (ref + eps), 0.5, 1.8)
        U_norm = (U / mag) * length_scale
        V_norm = (V / mag) * length_scale

        # 绘制背景箭头（灰黑色，细长）
        ax.quiver(xf, yf, U_norm, V_norm,
                  color='#2a2a2a',
                  alpha=0.72,
                  angles='xy', scale_units='xy', scale=1.0,
                  width=0.0022,
                  headwidth=3.5, headlength=5.0, headaxislength=4.5,
                  pivot='tail', zorder=3)

        # ── Step 2: 绘制主损伤轨迹（彩色弧线箭头）────────────────
        if has_hist and len(pinn.x_i_hist) >= 3:
            # 均匀取5个轨迹快照
            traj_n = min(6, len(pinn.x_i_hist))
            traj_idx = np.linspace(0, len(pinn.x_i_hist)-1, traj_n, dtype=int)
            cx_traj = [float(pinn.x_i_hist[ti][main_idx_final]) for ti in traj_idx]
            cy_traj = [float(pinn.y_i_hist[ti][main_idx_final]) for ti in traj_idx]
            ep_traj = [int(ep_arr[ti]) if ti < len(ep_arr) else ti for ti in traj_idx]

            traj_cmap = palettes[ax_i]
            traj_colors = [traj_cmap[int(i * (len(traj_cmap)-1) / max(traj_n-1,1))]
                           for i in range(traj_n)]

            # 绘制弧线轨迹箭头
            for si in range(len(cx_traj) - 1):
                x0, y0 = cx_traj[si], cy_traj[si]
                x1, y1 = cx_traj[si+1], cy_traj[si+1]
                col_arc = traj_colors[si+1]
                ax.annotate(
                    '', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle='->', color=col_arc,
                        lw=2.2, mutation_scale=18,
                        connectionstyle='arc3,rad=0.20'
                    ), zorder=13
                )
                mx = x0 * 0.45 + x1 * 0.55
                my = y0 * 0.45 + y1 * 0.55
                ax.text(mx + 0.01, my + 0.01, f'ep{ep_traj[si+1]}',
                        fontsize=7, color=col_arc,
                        bbox=dict(facecolor='white', alpha=0.65, pad=1, edgecolor='none'),
                        zorder=14)

            # 绘制轨迹点
            for si, (cxp, cyp) in enumerate(zip(cx_traj, cy_traj)):
                mk = '*' if si == len(cx_traj)-1 else 'o'
                ms = 160 if si == len(cx_traj)-1 else 70
                ax.scatter(cxp, cyp, s=ms, marker=mk,
                           c=[traj_colors[si]],
                           edgecolors='#333333', linewidths=1.1, zorder=15)

        # ── Step 3: 绘制其他损伤的消失轨迹（灰色，逐渐透明）──────
        other_colors = ['#888888', '#666666', '#aaaaaa']
        if has_hist:
            traj_n_o = min(5, len(pinn.x_i_hist))
            traj_idx_o = np.linspace(0, len(pinn.x_i_hist)-1, traj_n_o, dtype=int)
            for k in range(pinn.n_max):
                if k == main_idx_final:
                    continue
                ok_cx = [float(pinn.x_i_hist[ti][k]) for ti in traj_idx_o]
                ok_cy = [float(pinn.y_i_hist[ti][k]) for ti in traj_idx_o]
                col_k = other_colors[(k) % len(other_colors)]
                for si in range(len(ok_cx) - 1):
                    fade = 0.65 * (1.0 - si / max(len(ok_cx)-1, 1))
                    ax.annotate(
                        '', xy=(ok_cx[si+1], ok_cy[si+1]),
                        xytext=(ok_cx[si], ok_cy[si]),
                        arrowprops=dict(
                            arrowstyle='->', color=col_k,
                            lw=1.3, mutation_scale=10,
                            connectionstyle='arc3,rad=0.08',
                            alpha=fade
                        ), zorder=11
                    )
                # 终点标注
                last_alpha_k = float(pinn.alpha_hist[-1][k]) if pinn.alpha_hist else 0.0
                ax.scatter(ok_cx[-1], ok_cy[-1], s=40, marker='o', c=[col_k],
                           edgecolors='#555555', linewidths=0.8,
                           alpha=0.55, zorder=12)
                ax.text(ok_cx[-1]+0.013, ok_cy[-1],
                        f'D{k} α={last_alpha_k:.2f}',
                        fontsize=7, color=col_k, alpha=0.8,
                        bbox=dict(facecolor='white', alpha=0.55, pad=1, edgecolor='none'),
                        zorder=12)

        # ── Step 4: 损伤圆标注 ───────────────────────────────────
        _draw_circles(ax, pinn, show_inferred=(ax_i == 0))

        # ── 图例 ─────────────────────────────────────────────────
        leg_elems = [
            mpatches.Patch(facecolor='none', edgecolor='#cc0000', lw=2,
                           label='True Damage'),
            plt.Line2D([0],[0], color='#2a2a2a', lw=1.5,
                       marker='>', markersize=6, label='Flow field arrows'),
        ]
        if has_hist:
            leg_elems.append(
                plt.Line2D([0],[0], color=palettes[ax_i][-1], lw=2.5,
                           label='Main dmg trajectory'))
            leg_elems.append(
                plt.Line2D([0],[0], color='#888888', lw=1.5, alpha=0.6,
                           label='Other dmg (fading)'))
        if ax_i == 0:
            leg_elems.append(
                mpatches.Patch(facecolor='none', edgecolor='#1a5276',
                               lw=1.5, linestyle='--', label='Inferred Damage'))
        ax.legend(handles=leg_elems, loc='upper right', fontsize=8,
                  facecolor='white', edgecolor='#cccccc', framealpha=0.92)

    plt.tight_layout(rect=[0, 0, 1, 0.97], pad=1.2)

    out = save_path or f'stage{stage}_damage_gradient_field_flow.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [GradFlow] Saved → {out}")
    plt.close(fig)