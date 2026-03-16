"""
pinn.py  —  PINN for Kirchhoff-Love plate damage identification
Three-stage training: healthy network → joint optimisation → refinement.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    _SCIPY = True
except ImportError:
    _SCIPY = False

from network import DNN
from sampler import lhs_points, adaptive_points
import config as C


class PINN:
    def __init__(self,
                 t_data, x_data, y_data, u_data,
                 t_pde,  x_pde,  y_pde,
                 device='cuda'):

        self.device = device
        self.dtype  = torch.float64
        torch.manual_seed(C.R_SEED);  torch.cuda.manual_seed(C.R_SEED)
        np.random.seed(C.R_SEED)

        # ── Physical parameters ──────────────────────────────────────────────
        self.E          = C.E
        self.nu         = C.NU
        self.rho        = C.RHO0
        self.h          = C.H_PLATE
        self.Lx         = C.X_PHYSICAL
        self.Ly         = C.Y_PHYSICAL
        self.T_phys     = C.T_PHYSICAL
        self.D          = self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))
        self.pde_coeff  = (self.rho * self.h * self.Lx**4) / (self.D * self.T_phys**2)
        self.delta_sig  = C.DELTA_SIGMA
        self.beta       = C.BETA
        self.n_max      = C.K_MAX

        # ── Data tensors ─────────────────────────────────────────────────────
        def _t(arr):
            return arr.to(device, dtype=self.dtype) if arr is not None else None

        self.t_data, self.x_data, self.y_data, self.u_data = (
            _t(t_data), _t(x_data), _t(y_data), _t(u_data))
        self.t_pde, self.x_pde, self.y_pde = _t(t_pde), _t(x_pde), _t(y_pde)

        self._update_feature_bounds()

        # ── Network ──────────────────────────────────────────────────────────
        layers = [C.F_IN] + (C.DEPTH - 1) * [C.WIDTH] + [C.F_OUT]
        self.net = DNN(layers, C.W_INIT, C.B_INIT, C.ACTIVATION,
                       C.LAAF, self.dtype).to(device)

        # ── Damage parameters (reparameterised) ──────────────────────────────
        self.raw_alpha = nn.Parameter(torch.empty(self.n_max, dtype=self.dtype).uniform_(-2., 1.))
        self.raw_x_i   = nn.Parameter(torch.empty(self.n_max, dtype=self.dtype).uniform_(-2., 2.))
        self.raw_y_i   = nn.Parameter(torch.empty(self.n_max, dtype=self.dtype).uniform_(-2., 2.))
        self.raw_r_i   = nn.Parameter(torch.empty(self.n_max, dtype=self.dtype).uniform_(-2., 1.))

        # ── Training state ───────────────────────────────────────────────────
        self.stage       = 0
        self.global_step = 0
        self.w_pde       = C.W_PDE
        self.w_data      = C.W_DATA_S1
        self.w_reg_r     = C.W_REG_R_S2
        self.w_reg_g     = C.W_REG_G_S2
        self.adaptive_on = False
        self.last_adapt  = 0

        # ── Logging ──────────────────────────────────────────────────────────
        self.ep_log          = [];  self.loss_log       = []
        self.loss_pde_log    = [];  self.loss_data_log  = []
        self.loss_reg_r_log  = [];  self.loss_reg_g_log = []
        self.beta_log        = []
        self.damage_ep_hist  = [];  self.alpha_hist     = []
        self.x_i_hist        = [];  self.y_i_hist       = []
        self.r_i_hist        = []
        self.lbfgs_loss_hist = [];  self.lbfgs_pde_hist = []
        self.lbfgs_data_hist = [];  self.lbfgs_grad_norms = []
        self.gradient_history  = [];  self.gradient_epochs = []

        # ── Data interpolator (for gradient-field visualisation) ──────────────
        self._data_interp   = None
        self._nearest_interp = None
        if self.t_data is not None and _SCIPY:
            pts = np.hstack([
                self.t_data.detach().cpu().numpy(),
                self.x_data.detach().cpu().numpy(),
                self.y_data.detach().cpu().numpy()])
            vals = self.u_data.detach().cpu().numpy().flatten()
            self._data_interp = LinearNDInterpolator(pts, vals)

        print(f"PINN initialised | net {layers} | data {self.t_data.shape[0] if self.t_data is not None else 0}"
              f" | PDE {self.t_pde.shape[0] if self.t_pde is not None else 0} | device {device}")
        print(f"  D = {self.D:.4e} N·m,  pde_coeff = {self.pde_coeff:.4e}")

    # ── Feature bounds ────────────────────────────────────────────────────────

    def _update_feature_bounds(self):
        parts = []
        if self.t_pde  is not None: parts.append(torch.cat([self.t_pde,  self.t_data  if self.t_data  is not None else self.t_pde], 0))
        if self.t_data is not None and self.t_pde is None: parts.append(self.t_data)
        if not parts:
            self.lb = torch.zeros(3, dtype=self.dtype, device=self.device)
            self.ub = torch.ones (3, dtype=self.dtype, device=self.device)
            return
        def _mm(tensors, fn): return fn(torch.cat(tensors, 0), dim=0)[0]
        tx = (torch.cat([self.t_pde,  self.t_data],  0) if (self.t_pde  is not None and self.t_data  is not None)
              else (self.t_pde  if self.t_pde  is not None else self.t_data))
        xx = (torch.cat([self.x_pde,  self.x_data],  0) if (self.x_pde  is not None and self.x_data  is not None)
              else (self.x_pde  if self.x_pde  is not None else self.x_data))
        yx = (torch.cat([self.y_pde,  self.y_data],  0) if (self.y_pde  is not None and self.y_data  is not None)
              else (self.y_pde  if self.y_pde  is not None else self.y_data))
        bounds = torch.cat([tx, xx, yx], dim=1)
        self.lb = torch.min (bounds, dim=0)[0].to(self.device, dtype=self.dtype)
        self.ub = torch.max (bounds, dim=0)[0].to(self.device, dtype=self.dtype)

    # ── Damage parameter properties ───────────────────────────────────────────

    @property
    def alpha(self):
        if self.stage == 1:
            return torch.zeros(self.n_max, dtype=self.dtype, device=self.device)
        return torch.sigmoid(self.raw_alpha)

    @property
    def r_i(self):
        """Physical radius (m)."""
        return torch.nn.functional.softplus(self.raw_r_i) * 0.014 + 0.001

    @property
    def r_i_norm(self):
        """Normalised radius."""
        return self.r_i / self.Lx

    @property
    def x_i(self):
        m = 0.04
        return torch.sigmoid(self.raw_x_i) * (1.0 - 2*m) + m

    @property
    def y_i(self):
        m = 0.04
        return torch.sigmoid(self.raw_y_i) * (1.0 - 2*m) + m

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def _build_optimizer(self):
        lr_net = C.LR_NET if self.stage == 1 else C.LR_DAMAGE * 0.1
        dam_params = [self.raw_alpha, self.raw_x_i, self.raw_y_i, self.raw_r_i]
        if self.stage == 1:
            for p in dam_params: p.requires_grad_(False)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr_net)
        else:
            for p in dam_params: p.requires_grad_(True)
            self.optimizer = torch.optim.Adam([
                {'params': self.net.parameters(), 'lr': lr_net},
                {'params': dam_params,            'lr': C.LR_DAMAGE}])

    def set_stage(self, stage):
        self.stage = stage
        self._build_optimizer()
        self.adaptive_on = (stage >= 2)
        print(f"Stage {stage} set | net_lr={C.LR_NET if stage==1 else C.LR_DAMAGE*0.1:.2e}"
              + (f" | damage_lr={C.LR_DAMAGE:.2e}" if stage >= 2 else ""))

    # ── Damage initialisation ─────────────────────────────────────────────────

    def init_damage_params(self, positions=None, radii_mm=None, alphas=None):
        """Set damage parameters to the given initial guesses."""
        positions = positions or C.INIT_POSITIONS
        radii_mm  = radii_mm  or C.INIT_RADII_MM
        alphas    = alphas    or C.INIT_ALPHA

        m, s = 0.04, 1.0 - 2*0.04

        def _logit(v): return float(np.log(np.clip(v,1e-5,1-1e-5) / (1 - np.clip(v,1e-5,1-1e-5))))

        raw_a, raw_x, raw_y, raw_r = [], [], [], []
        for k in range(self.n_max):
            raw_a.append(_logit(alphas[k] if k < len(alphas) else 0.1))
            xn, yn = (positions[k] if k < len(positions) else (0.5, 0.5))
            xn = np.clip((xn - m) / s, 1e-5, 1-1e-5)
            yn = np.clip((yn - m) / s, 1e-5, 1-1e-5)
            raw_x.append(_logit(xn));  raw_y.append(_logit(yn))
            r_m = (radii_mm[k] / 1000.0 if k < len(radii_mm) else 0.005)
            v = np.clip((r_m - 0.001) / 0.014, 1e-6, None)
            raw_r.append(float(np.log(np.exp(v) - 1.0 + 1e-9)))

        def _pt(lst): return torch.tensor(lst, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            self.raw_alpha.data = _pt(raw_a)
            self.raw_x_i.data   = _pt(raw_x)
            self.raw_y_i.data   = _pt(raw_y)
            self.raw_r_i.data   = _pt(raw_r)

        self._print_damage_state("Damage initialisation")

    def _print_damage_state(self, label=""):
        a  = self.alpha.detach().cpu().numpy()
        xi = self.x_i.detach().cpu().numpy()
        yi = self.y_i.detach().cpu().numpy()
        ri = self.r_i.detach().cpu().numpy() * 1000
        print(f"\n{label}")
        print(f"  {'#':<3} {'α':>6} {'x':>7} {'y':>7} {'r (mm)':>8}")
        for k in range(self.n_max):
            print(f"  {k:<3} {a[k]:6.3f} {xi[k]:7.4f} {yi[k]:7.4f} {ri[k]:8.2f}")

    # ── PDE-point update ──────────────────────────────────────────────────────

    def _refresh_pde_points(self, epoch):
        if not self.adaptive_on: return
        if epoch - self.last_adapt < C.ADAPTIVE_UPDATE_INTERVAL: return

        a  = self.alpha.detach().cpu().numpy()
        xc = self.x_i.detach().cpu().numpy()
        yc = self.y_i.detach().cpu().numpy()
        rc = self.r_i_norm.detach().cpu().numpy()

        t, x, y = adaptive_points(
            C.N_PDE, self.stage, epoch, xc, yc, rc, a,
            early_phase=C.S2_EARLY_PHASE)

        def _ten(arr): return torch.tensor(arr, dtype=self.dtype, device=self.device)
        self.t_pde, self.x_pde, self.y_pde = _ten(t), _ten(x), _ten(y)
        self._update_feature_bounds()
        self.last_adapt = epoch

    # ── Forward + PDE residual ────────────────────────────────────────────────

    def _scale(self, X):
        return 2.0 * (X - self.lb) / (self.ub - self.lb + 1e-12) - 1.0

    def forward(self, t, x, y):
        return self.net(self._scale(torch.cat([t, x, y], dim=1)))

    def pde_residual(self, t, x, y):
        t.requires_grad_(True);  x.requires_grad_(True);  y.requires_grad_(True)
        u = self.forward(t, x, y)

        def _g(f, v, cg=True):
            return torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f),
                                       create_graph=cg)[0]

        u_t  = _g(u,  t)
        u_x  = _g(u,  x)
        u_y  = _g(u,  y)
        u_tt = _g(u_t, t)
        u_xx = _g(u_x, x)
        u_yy = _g(u_y, y)
        u_xy = _g(u_x, y)
        u_xxx = _g(u_xx, x);  u_xxy = _g(u_xx, y)
        u_xyy = _g(u_xy, y);  u_yyy = _g(u_yy, y)
        u_xxxx = _g(u_xxx, x)
        u_xxyy = _g(u_xxy, y)
        u_yyyy = _g(u_yyy, y)

        nabla4 = u_xxxx + 2.0 * u_xxyy + u_yyyy

        if self.stage == 1:
            return nabla4 + self.pde_coeff * u_tt

        sig = torch.ones_like(u) * (self.rho * self.h)
        for k in range(self.n_max):
            d2   = (x - self.x_i[k])**2 + (y - self.y_i[k])**2
            gate = torch.sigmoid(self.beta * (self.r_i_norm[k]**2 - d2))
            sig  = sig + self.alpha[k] * self.delta_sig * gate

        coeff = sig * self.Lx**4 / (self.D * self.T_phys**2)
        return nabla4 + coeff * u_tt

    # ── Regularisation ────────────────────────────────────────────────────────

    def _sparsity_loss(self, epoch):
        """
        Encourage exactly one dominant damage site (max α → 1) while
        suppressing the others via L1 regularisation.
        Switches on gradually after a burn-in period.
        """
        if self.stage == 1: return torch.zeros(1, dtype=self.dtype, device=self.device).squeeze()

        a = self.alpha
        max_a, max_idx = torch.max(a, 0)

        # burn-in: lightly encourage the dominant site to grow
        if epoch < 2000:
            main   = (1.0 - max_a)**2
            others = a.clone();  others[max_idx] = 0.0
            return 1e-4 * (main + 0.1 * torch.sum(others**2))

        # later: stronger sparsification
        w = min(1e-3, 1e-4 + (epoch - 2000) / 3000 * 9e-4)
        others = a.clone();  others[max_idx] = 0.0
        return w * ((1.0 - max_a)**2 + 0.5 * torch.sum(torch.abs(others)))

    def _radius_loss(self):
        """Soft regulariser on damage radii (physical metres)."""
        r = self.r_i
        max_idx = torch.argmax(self.alpha)
        main_pen  = (r[max_idx] - 0.0048)**2
        other_pen = torch.sum((r - 0.005)**2) - (r[max_idx] - 0.005)**2
        return self.w_reg_r * (2.0 * main_pen + 0.5 * other_pen)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def compute_loss(self, t_p, x_p, y_p, t_d, x_d, y_d, u_d, epoch=0):
        loss_pde = (torch.mean(self.pde_residual(t_p, x_p, y_p)**2)
                    if t_p is not None else torch.zeros(1, dtype=self.dtype, device=self.device).squeeze())

        loss_data = (torch.mean((u_d - self.forward(t_d, x_d, y_d))**2)
                     if t_d is not None else torch.zeros(1, dtype=self.dtype, device=self.device).squeeze())

        if self.stage >= 2:
            loss_reg_g = self._sparsity_loss(epoch)
            loss_reg_r = self._radius_loss()
        else:
            loss_reg_g = torch.zeros(1, dtype=self.dtype, device=self.device).squeeze()
            loss_reg_r = torch.zeros(1, dtype=self.dtype, device=self.device).squeeze()

        total = (self.w_pde * loss_pde
                 + self.w_data * loss_data
                 + loss_reg_r + loss_reg_g)
        return total, loss_pde, loss_data, loss_reg_r, loss_reg_g

    # ── Gradient recording ────────────────────────────────────────────────────

    def _record_gradients(self, epoch):
        if self.stage < 2: return
        if not (epoch < 100 or epoch % 500 == 0):  return
        info = {}
        for name, p in self.net.named_parameters():
            if p.grad is not None:
                info[f"net_{name}"] = p.grad.norm(2).item()
        for name, p in [('raw_alpha', self.raw_alpha), ('raw_x_i', self.raw_x_i),
                         ('raw_y_i',  self.raw_y_i),   ('raw_r_i', self.raw_r_i)]:
            if p.grad is not None:
                info[f"dmg_{name}"] = p.grad.norm(2).item()
        self.gradient_history.append(info)
        self.gradient_epochs.append(epoch)

    # ── Adam training loop ────────────────────────────────────────────────────

    def train(self, epochs, batch, lr=1e-3, stage=1,
              start_epoch=0, joint_training=False):
        self.set_stage(stage)
        t0 = time.time()

        for ep in range(1, epochs + 1):
            self.global_step += 1
            g_ep = start_epoch + ep

            self._refresh_pde_points(g_ep)

            n_pde  = self.t_pde.shape[0]  if self.t_pde  is not None else 0
            n_data = self.t_data.shape[0] if self.t_data is not None else 0
            n_bat  = max(max(n_pde, n_data) // max(batch, 1), 1)

            ep_loss = ep_pde = ep_data = ep_rr = ep_rg = 0.0

            for _ in range(n_bat):
                self.optimizer.zero_grad()

                def _batch(N, *tensors):
                    if N == 0: return (None,) * len(tensors)
                    idx = torch.randint(0, N, (min(batch, N),), device=self.device)
                    return tuple(t[idx] for t in tensors)

                tp, xp, yp = _batch(n_pde,  self.t_pde,  self.x_pde,  self.y_pde)
                td, xd, yd, ud = _batch(n_data, self.t_data, self.x_data, self.y_data, self.u_data)

                if tp is None and td is None: continue

                loss, lpde, ldata, lrr, lrg = self.compute_loss(
                    tp, xp, yp, td, xd, yd, ud, g_ep)

                if torch.isnan(loss): continue
                loss.backward()
                self._record_gradients(g_ep)
                self.optimizer.step()

                ep_loss += loss.item()  / n_bat
                ep_pde  += lpde.item()  / n_bat
                ep_data += ldata.item() / n_bat
                ep_rr   += lrr.item()   / n_bat
                ep_rg   += lrg.item()   / n_bat

            # Logging
            if g_ep % C.F_MNTR == 0:
                self.ep_log.append(g_ep)
                self.loss_log.append(ep_loss);   self.loss_pde_log.append(ep_pde)
                self.loss_data_log.append(ep_data)
                self.loss_reg_r_log.append(ep_rr); self.loss_reg_g_log.append(ep_rg)
                self.beta_log.append(self.beta)

                if stage >= 2:
                    self.damage_ep_hist.append(g_ep)
                    self.alpha_hist.append(self.alpha.detach().cpu().numpy())
                    self.x_i_hist.append(self.x_i.detach().cpu().numpy())
                    self.y_i_hist.append(self.y_i.detach().cpu().numpy())
                    self.r_i_hist.append(self.r_i.detach().cpu().numpy())

                    a_np = self.alpha.detach().cpu().numpy()
                    k_max = int(np.argmax(a_np))
                    print(f"S{stage} | ep {g_ep:5d} | loss {ep_loss:.3e} | pde {ep_pde:.3e} | "
                          f"data {ep_data:.3e} | "
                          f"α_max={a_np[k_max]:.3f} r={self.r_i[k_max].item()*1e3:.1f}mm "
                          f"x={self.x_i[k_max].item():.3f} y={self.y_i[k_max].item():.3f} "
                          f"| {time.time()-t0:.1f}s")
                else:
                    print(f"S{stage} | ep {g_ep:5d} | loss {ep_loss:.3e} | "
                          f"pde {ep_pde:.3e} | data {ep_data:.3e} | {time.time()-t0:.1f}s")
                t0 = time.time()

            # Periodic damage map
            if stage >= 2 and (g_ep % 100 == 0 or g_ep == start_epoch + epochs):
                try:
                    from plot import plot_damage_map
                    os.makedirs("./damage_evolution", exist_ok=True)
                    plot_damage_map(self, f"Epoch {g_ep}",
                                    f"Stage {stage}",
                                    stage=stage,
                                    save_path=f"./damage_evolution/damage_{g_ep:05d}.png")
                except Exception:
                    pass

        print(f"Stage {stage} done | final loss {ep_loss:.3e}")
        return ep_loss

    # ── L-BFGS (Stage 1 only) ─────────────────────────────────────────────────

    def train_lbfgs(self, epochs=1000, tol=1e-8, max_iter=50,
                    subset=C.PDE_LBFGS_SUBSET, lr=0.1):
        if self.stage != 1:
            print("L-BFGS is only used in Stage 1."); return 0.0

        params = list(self.net.parameters())
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter,
                                 tolerance_grad=tol, tolerance_change=tol,
                                 history_size=100, line_search_fn='strong_wolfe')
        N_pde = len(self.t_pde)
        use_sub = min(subset, N_pde)
        losses, pde_l, data_l, gnorms = [], [], [], []
        best_loss, best_params = float('inf'), None

        def closure():
            opt.zero_grad()
            idx = torch.randperm(N_pde, device=self.device)[:use_sub]
            loss, lpde, ldata, _, _ = self.compute_loss(
                self.t_pde[idx], self.x_pde[idx], self.y_pde[idx],
                self.t_data, self.x_data, self.y_data, self.u_data,
                self.global_step)
            if not torch.isnan(loss): loss.backward()
            g = sum(p.grad.norm(2).item()**2 for p in params if p.grad is not None) ** 0.5
            closure.vals = (loss.item(), lpde.item(), ldata.item(), g)
            return loss

        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.global_step += 1
            opt.step(closure)
            lv, pv, dv, gv = closure.vals
            losses.append(lv);  pde_l.append(pv);  data_l.append(dv);  gnorms.append(gv)
            if lv < best_loss:
                best_loss = lv
                best_params = [p.clone().detach() for p in params]
            if ep % 10 == 0:
                print(f"L-BFGS {ep:4d}/{epochs} | loss {lv:.3e} | pde {pv:.3e} | "
                      f"data {dv:.3e} | gnorm {gv:.3e} | {time.time()-t0:.1f}s")
            if ep > 10 and abs(losses[-1] - losses[-2]) < tol * max(1.0, losses[-2]): break

        if best_params:
            for p, bp in zip(params, best_params): p.data.copy_(bp)
        self.lbfgs_loss_hist = losses;  self.lbfgs_pde_hist = pde_l
        self.lbfgs_data_hist = data_l;  self.lbfgs_grad_norms = gnorms
        print(f"L-BFGS done | best loss {best_loss:.6e} | {len(losses)} iters")
        return losses[-1]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_model(self, path):
        torch.save({
            'net': self.net.state_dict(),
            'raw_alpha': self.raw_alpha.data,
            'raw_x_i':  self.raw_x_i.data,
            'raw_y_i':  self.raw_y_i.data,
            'raw_r_i':  self.raw_r_i.data,
            'stage':    self.stage,
            'ep_log':   self.ep_log,
            'loss_log': self.loss_log, 'loss_pde_log': self.loss_pde_log,
            'loss_data_log': self.loss_data_log,
            'loss_reg_r_log': self.loss_reg_r_log,
            'loss_reg_g_log': self.loss_reg_g_log,
            'beta_log': self.beta_log,
            'damage_ep_hist': self.damage_ep_hist,
            'alpha_hist': self.alpha_hist, 'x_i_hist': self.x_i_hist,
            'y_i_hist':  self.y_i_hist,   'r_i_hist': self.r_i_hist,
            'lbfgs_loss_hist': self.lbfgs_loss_hist,
            'lbfgs_pde_hist':  self.lbfgs_pde_hist,
            'lbfgs_data_hist': self.lbfgs_data_hist,
            'lbfgs_grad_norms': self.lbfgs_grad_norms,
            'gradient_history': self.gradient_history,
            'gradient_epochs':  self.gradient_epochs,
        }, path)
        print(f"Model saved → {path}")

    def load_model(self, path):
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck['net'])
        self.raw_alpha.data = ck['raw_alpha']
        self.raw_x_i.data   = ck['raw_x_i']
        self.raw_y_i.data   = ck['raw_y_i']
        self.raw_r_i.data   = ck['raw_r_i']   # bug fix: was loading raw_x_i here
        self.stage          = ck['stage']
        for attr in ('ep_log','loss_log','loss_pde_log','loss_data_log',
                     'loss_reg_r_log','loss_reg_g_log','beta_log',
                     'damage_ep_hist','alpha_hist','x_i_hist','y_i_hist','r_i_hist',
                     'lbfgs_loss_hist','lbfgs_pde_hist','lbfgs_data_hist',
                     'lbfgs_grad_norms','gradient_history','gradient_epochs'):
            setattr(self, attr, ck.get(attr, []))
        print(f"Model loaded ← {path} (stage {self.stage})")

    def save_damage_params(self, path):
        np.savez(path,
                 raw_alpha=self.raw_alpha.detach().cpu().numpy(),
                 raw_x_i  =self.raw_x_i.detach().cpu().numpy(),
                 raw_y_i  =self.raw_y_i.detach().cpu().numpy(),
                 raw_r_i  =self.raw_r_i.detach().cpu().numpy(),
                 alpha    =self.alpha.detach().cpu().numpy(),
                 x_i      =self.x_i.detach().cpu().numpy(),
                 y_i      =self.y_i.detach().cpu().numpy(),
                 r_i      =self.r_i.detach().cpu().numpy(),
                 r_i_norm =self.r_i_norm.detach().cpu().numpy())
        print(f"Damage params saved → {path}")

    # ── Spatial gradient (for plotting) ───────────────────────────────────────

    def spatial_gradient(self, t, x, y):
        t.requires_grad_(True);  x.requires_grad_(True);  y.requires_grad_(True)
        u = self.forward(t, x, y)
        gs = torch.autograd.grad(u, [x, y],
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=False, retain_graph=True,
                                  allow_unused=True)
        ux = gs[0] if gs[0] is not None else torch.zeros_like(x)
        uy = gs[1] if gs[1] is not None else torch.zeros_like(y)
        return ux.detach(), uy.detach()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self):
        """Return RMSE on training data and final damage-parameter summary."""
        with torch.no_grad():
            u_pred = self.forward(self.t_data, self.x_data, self.y_data)
            rmse = torch.sqrt(torch.mean((self.u_data - u_pred)**2)).item()

        a  = self.alpha.detach().cpu().numpy()
        xi = self.x_i.detach().cpu().numpy()
        yi = self.y_i.detach().cpu().numpy()
        ri = self.r_i.detach().cpu().numpy()

        k_max = int(np.argmax(a))
        results = dict(rmse_data=rmse,
                       main_alpha=float(a[k_max]),
                       main_x=float(xi[k_max]),
                       main_y=float(yi[k_max]),
                       main_r_mm=float(ri[k_max] * 1000))

        if self.loss_log:
            results['final_total_loss'] = self.loss_log[-1]
            results['final_pde_loss']   = self.loss_pde_log[-1]
            results['final_data_loss']  = self.loss_data_log[-1]
        return results
