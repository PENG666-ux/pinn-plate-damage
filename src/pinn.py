"""Physics-informed model for inverse damage identification in thin plates."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:  # package import
    from . import config as C
    from .network import DNN
    from .sampling import adaptive_points, boundary_points
except ImportError:  # direct ``python src/main.py`` execution
    import config as C
    from network import DNN
    from sampling import adaptive_points, boundary_points


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed NumPy/PyTorch and request deterministic kernels when available."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic


class PINN:
    """Three-stage PINN with bounded, differentiable damage parameters.

    Coordinates and time are dimensionless in ``[0, 1]``.  Physical plate
    dimensions enter the rectangular-plate operator explicitly.  The network
    learns standardized deflection, making PDE/data loss weights independent of
    the displacement unit used by the measurement files.
    """

    def __init__(self, t_data, x_data, y_data, u_data,
                 t_pde, x_pde, y_pde, device="cpu"):
        self.device = torch.device(device)
        self.dtype = torch.float64
        set_reproducible_seed(C.R_SEED, C.DETERMINISTIC)

        self.E, self.nu, self.rho, self.h = C.E, C.NU, C.RHO0, C.H_PLATE
        self.Lx, self.Ly, self.T_phys = C.X_PHYSICAL, C.Y_PHYSICAL, C.T_PHYSICAL
        self.D = self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))
        self.aspect2 = (self.Lx / self.Ly) ** 2
        self.pde_scale = self.rho * self.h * self.Lx**4 / (self.D * self.T_phys**2)
        self.delta_sig, self.beta, self.n_max = C.DELTA_SIGMA, C.BETA, C.K_MAX
        self._validate_physics()

        def tensor(value):
            if value is None:
                return None
            result = torch.as_tensor(value, dtype=self.dtype, device=self.device)
            return result.reshape(-1, 1)

        self.t_data, self.x_data, self.y_data, self.u_data = map(
            tensor, (t_data, x_data, y_data, u_data))
        self.t_pde, self.x_pde, self.y_pde = map(tensor, (t_pde, x_pde, y_pde))
        self._validate_coordinates()

        if self.u_data is None or not len(self.u_data):
            self.u_mean = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            self.u_scale = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        else:
            self.u_mean = self.u_data.mean()
            scale = self.u_data.std(unbiased=False)
            self.u_scale = torch.where(scale > torch.finfo(self.dtype).eps,
                                       scale, torch.ones_like(scale))

        # Bounds are part of the mathematical normalization, not estimated from
        # a changing collocation cloud.
        self.lb = torch.tensor([0.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
        self.ub = torch.tensor([1.0, 1.0, 1.0], dtype=self.dtype, device=self.device)
        layers = [C.F_IN] + (C.DEPTH - 1) * [C.WIDTH] + [C.F_OUT]
        self.net = DNN(layers, C.W_INIT, C.B_INIT, C.ACTIVATION,
                       C.LAAF, self.dtype).to(self.device)

        self.raw_alpha = nn.Parameter(torch.zeros(self.n_max, dtype=self.dtype, device=self.device))
        self.raw_x_i = nn.Parameter(torch.zeros(self.n_max, dtype=self.dtype, device=self.device))
        self.raw_y_i = nn.Parameter(torch.zeros(self.n_max, dtype=self.dtype, device=self.device))
        self.raw_r_i = nn.Parameter(torch.zeros(self.n_max, dtype=self.dtype, device=self.device))
        self.init_damage_params()

        t_bc, x_bc, y_bc = boundary_points(C.N_BC, seed=C.R_SEED + 1)
        self.t_bc, self.x_bc, self.y_bc = map(tensor, (t_bc, x_bc, y_bc))
        self.stage = 0
        self.global_step = 0
        self.last_adapt = 0
        self.w_pde, self.w_bc, self.w_data = C.W_PDE, C.W_BC, C.W_DATA_S1
        self.w_reg_r, self.w_reg_g = C.W_REG_R_S2, C.W_REG_G_S2
        self.history = {key: [] for key in
                        ("epoch", "total", "pde", "boundary", "data",
                         "radius_regularization", "sparsity", "validation_rmse")}
        self.damage_history = []
        self.lbfgs_history = []
        self.optimizer = None

    def _validate_physics(self):
        positive = {"E": self.E, "rho": self.rho, "h": self.h, "Lx": self.Lx,
                    "Ly": self.Ly, "T_phys": self.T_phys, "delta_sigma": self.delta_sig}
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid or not (-1.0 < self.nu < 0.5):
            raise ValueError(f"Invalid physical parameters: {invalid}, nu={self.nu}")
        if C.R_MIN_MM <= 0 or C.R_MIN_MM >= C.R_MAX_MM:
            raise ValueError("Require 0 < R_MIN_MM < R_MAX_MM")

    def _validate_coordinates(self):
        for name, value in (("t_data", self.t_data), ("x_data", self.x_data),
                            ("y_data", self.y_data), ("t_pde", self.t_pde),
                            ("x_pde", self.x_pde), ("y_pde", self.y_pde)):
            if value is not None and (not torch.isfinite(value).all() or
                                      value.min() < -1e-12 or value.max() > 1 + 1e-12):
                raise ValueError(f"{name} must contain finite normalized values in [0, 1]")
        sizes = [len(v) for v in (self.t_data, self.x_data, self.y_data, self.u_data) if v is not None]
        if sizes and len(set(sizes)) != 1:
            raise ValueError("data tensors must have equal lengths")

    @property
    def alpha(self):
        return (torch.zeros(self.n_max, dtype=self.dtype, device=self.device)
                if self.stage == 1 else torch.sigmoid(self.raw_alpha))

    @property
    def x_i(self):
        return 0.04 + 0.92 * torch.sigmoid(self.raw_x_i)

    @property
    def y_i(self):
        return 0.04 + 0.92 * torch.sigmoid(self.raw_y_i)

    @property
    def r_i(self):
        """Physical radii in metres, strictly bounded by configuration."""
        lower, span = C.R_MIN_MM / 1000.0, (C.R_MAX_MM - C.R_MIN_MM) / 1000.0
        return lower + span * torch.sigmoid(self.raw_r_i)

    @property
    def r_i_norm(self):
        return self.r_i / self.Lx

    # Compatibility aliases used by older plotting scripts.
    @property
    def current_beta(self): return self.beta
    @property
    def DELTA_SIGMA_FIXED(self): return self.delta_sig
    @property
    def X_physical(self): return self.Lx
    @property
    def Y_physical(self): return self.Ly

    @staticmethod
    def _logit(value):
        value = np.clip(value, 1e-8, 1.0 - 1e-8)
        return float(np.log(value / (1.0 - value)))

    def init_damage_params(self, positions=None, radii_mm=None, alphas=None):
        positions = C.INIT_POSITIONS if positions is None else positions
        radii_mm = C.INIT_RADII_MM if radii_mm is None else radii_mm
        alphas = C.INIT_ALPHA if alphas is None else alphas
        if not (len(positions) == len(radii_mm) == len(alphas) == self.n_max):
            raise ValueError("Initial position, radius, and alpha lists must match K_MAX")
        radius_span = C.R_MAX_MM - C.R_MIN_MM
        raw_x = [self._logit((x - 0.04) / 0.92) for x, _ in positions]
        raw_y = [self._logit((y - 0.04) / 0.92) for _, y in positions]
        raw_r = [self._logit((r - C.R_MIN_MM) / radius_span) for r in radii_mm]
        values = ([self._logit(a) for a in alphas], raw_x, raw_y, raw_r)
        with torch.no_grad():
            for parameter, value in zip((self.raw_alpha, self.raw_x_i,
                                         self.raw_y_i, self.raw_r_i), values):
                parameter.copy_(torch.tensor(value, dtype=self.dtype, device=self.device))

    def set_stage(self, stage):
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")
        self.stage = stage
        damage_parameters = [self.raw_alpha, self.raw_x_i, self.raw_y_i, self.raw_r_i]
        for parameter in damage_parameters:
            parameter.requires_grad_(stage >= 2)
        if stage == 1:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=C.LR_NET)
        else:
            self.optimizer = torch.optim.Adam([
                {"params": self.net.parameters(), "lr": C.LR_DAMAGE * 0.1},
                {"params": damage_parameters, "lr": C.LR_DAMAGE},
            ])

    def _scale(self, inputs):
        return 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0

    def normalized_forward(self, t, x, y):
        return self.net(self._scale(torch.cat([t, x, y], dim=1)))

    def forward(self, t, x, y):
        return self.u_mean + self.u_scale * self.normalized_forward(t, x, y)

    @staticmethod
    def _gradient(output, variable):
        return torch.autograd.grad(output, variable, torch.ones_like(output),
                                   create_graph=True, retain_graph=True)[0]

    def pde_residual(self, t, x, y):
        t = t.detach().clone().requires_grad_(True)
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        u = self.normalized_forward(t, x, y)
        u_t = self._gradient(u, t); u_tt = self._gradient(u_t, t)
        u_x = self._gradient(u, x); u_xx = self._gradient(u_x, x)
        u_y = self._gradient(u, y); u_yy = self._gradient(u_y, y)
        u_xxx = self._gradient(u_xx, x); u_xxxx = self._gradient(u_xxx, x)
        u_xxy = self._gradient(u_xx, y); u_xxyy = self._gradient(u_xxy, y)
        u_yyy = self._gradient(u_yy, y); u_yyyy = self._gradient(u_yyy, y)
        biharmonic = u_xxxx + 2.0 * self.aspect2 * u_xxyy + self.aspect2**2 * u_yyyy

        sigma = torch.full_like(u, self.rho * self.h)
        if self.stage >= 2:
            for k in range(self.n_max):
                distance = torch.sqrt((x - self.x_i[k])**2 + (y - self.y_i[k])**2 + 1e-16)
                # Signed-distance gate: beta has an interpretable transition
                # width in normalized coordinates and avoids the near-0.5 gate
                # produced by beta*(r^2-d^2) at realistic plate radii.
                gate = torch.sigmoid(self.beta * (self.r_i_norm[k] - distance))
                sigma = sigma + self.alpha[k] * self.delta_sig * gate
        inertia = sigma * self.Lx**4 / (self.D * self.T_phys**2)
        return (biharmonic + inertia * u_tt) / max(1.0, self.pde_scale)

    def boundary_residuals(self, t, x, y):
        """Simply-supported essential and bending-moment residuals."""
        if C.BOUNDARY_CONDITION != "simply_supported":
            raise NotImplementedError(f"Unsupported boundary condition: {C.BOUNDARY_CONDITION}")
        t = t.detach().clone().requires_grad_(True)
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        z = self.normalized_forward(t, x, y)
        z_xx = self._gradient(self._gradient(z, x), x)
        z_yy = self._gradient(self._gradient(z, y), y)
        on_x = torch.isclose(x, torch.zeros_like(x)) | torch.isclose(x, torch.ones_like(x))
        moment_x = z_xx + self.nu * self.aspect2 * z_yy
        moment_y = self.aspect2 * z_yy + self.nu * z_xx
        moment = torch.where(on_x, moment_x, moment_y)
        displacement = self.forward(t, x, y) / self.u_scale
        return displacement, moment

    def _regularization(self):
        if self.stage == 1:
            zero = torch.zeros((), dtype=self.dtype, device=self.device)
            return zero, zero
        # Weighted area discourages diffuse/duplicate sites without encoding a
        # privileged radius or forcing exactly one damage location.
        radius = self.w_reg_r * torch.sum(self.alpha * (self.r_i / (C.R_MAX_MM / 1000.0)) ** 2)
        sparsity = self.w_reg_g * torch.sum(self.alpha)
        return radius, sparsity

    def compute_loss(self, t_p, x_p, y_p, t_d, x_d, y_d, u_d,
                     t_b=None, x_b=None, y_b=None):
        zero = torch.zeros((), dtype=self.dtype, device=self.device)
        pde = self.pde_residual(t_p, x_p, y_p).square().mean() if t_p is not None else zero
        data = (((self.forward(t_d, x_d, y_d) - u_d) / self.u_scale).square().mean()
                if t_d is not None else zero)
        if t_b is not None:
            displacement, moment = self.boundary_residuals(t_b, x_b, y_b)
            boundary = displacement.square().mean() + moment.square().mean()
        else:
            boundary = zero
        radius, sparsity = self._regularization()
        total = (self.w_pde * pde + self.w_bc * boundary +
                 self.w_data * data + radius + sparsity)
        return total, pde, boundary, data, radius, sparsity

    def _refresh_pde_points(self, epoch):
        if self.stage < 2 or epoch - self.last_adapt < C.ADAPTIVE_UPDATE_INTERVAL:
            return
        values = [v.detach().cpu().numpy() for v in
                  (self.x_i, self.y_i, self.r_i_norm, self.alpha)]
        arrays = adaptive_points(C.N_PDE, self.stage, epoch, *values,
                                 early_phase=C.S2_EARLY_PHASE,
                                 seed=C.R_SEED + epoch)
        self.t_pde, self.x_pde, self.y_pde = [
            torch.tensor(a, dtype=self.dtype, device=self.device) for a in arrays]
        self.last_adapt = epoch

    def train(self, epochs, batch, stage=1, start_epoch=0, validation_data=None,
              **_ignored):
        self.set_stage(stage)
        if epochs < 1 or batch < 1:
            raise ValueError("epochs and batch must be positive")
        started = time.time()
        for local_epoch in range(1, epochs + 1):
            epoch = start_epoch + local_epoch
            self.global_step += 1
            self._refresh_pde_points(epoch)
            n_pde, n_data, n_bc = len(self.t_pde), len(self.t_data), len(self.t_bc)
            n_batches = max(1, math.ceil(max(n_pde, n_data, n_bc) / batch))
            sums = np.zeros(6)
            for _ in range(n_batches):
                self.optimizer.zero_grad(set_to_none=True)

                def take(size, *arrays):
                    indices = torch.randint(size, (min(batch, size),), device=self.device)
                    return [array[indices] for array in arrays]

                tp, xp, yp = take(n_pde, self.t_pde, self.x_pde, self.y_pde)
                td, xd, yd, ud = take(n_data, self.t_data, self.x_data, self.y_data, self.u_data)
                tb, xb, yb = take(n_bc, self.t_bc, self.x_bc, self.y_bc)
                losses = self.compute_loss(tp, xp, yp, td, xd, yd, ud, tb, xb, yb)
                if not torch.isfinite(losses[0]):
                    raise FloatingPointError(f"Non-finite loss at stage {stage}, epoch {epoch}")
                losses[0].backward()
                trainable = [parameter for group in self.optimizer.param_groups
                             for parameter in group["params"] if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1e4)
                self.optimizer.step()
                sums += np.array([value.detach().item() for value in losses]) / n_batches

            if epoch % C.F_MNTR == 0 or local_epoch == epochs:
                val_rmse = (self.evaluate(validation_data)["rmse"]
                            if validation_data is not None else float("nan"))
                for key, value in zip(("total", "pde", "boundary", "data",
                                       "radius_regularization", "sparsity"), sums):
                    self.history[key].append(float(value))
                self.history["epoch"].append(epoch)
                self.history["validation_rmse"].append(val_rmse)
                self.damage_history.append(self.damage_parameters())
                print(f"S{stage} epoch={epoch} loss={sums[0]:.3e} pde={sums[1]:.3e} "
                      f"bc={sums[2]:.3e} data={sums[3]:.3e} val_rmse={val_rmse:.3e} "
                      f"elapsed={time.time() - started:.1f}s")
        return float(sums[0])

    def train_lbfgs(self, epochs=1000, tol=1e-8, max_iter=20,
                    subset=C.PDE_LBFGS_SUBSET, validation_data=None, lr=0.5):
        """Deterministic L-BFGS: a fixed subset is reused by every closure."""
        if self.stage != 1:
            raise RuntimeError("L-BFGS is only valid in Stage 1")
        generator = torch.Generator(device=self.device).manual_seed(C.R_SEED + 10)
        p_idx = torch.randperm(len(self.t_pde), generator=generator, device=self.device)[:subset]
        b_idx = torch.randperm(len(self.t_bc), generator=generator, device=self.device)[:min(subset, len(self.t_bc))]
        optimizer = torch.optim.LBFGS(self.net.parameters(), lr=lr, max_iter=max_iter,
                                      tolerance_grad=tol, tolerance_change=tol,
                                      history_size=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            losses = self.compute_loss(self.t_pde[p_idx], self.x_pde[p_idx], self.y_pde[p_idx],
                                       self.t_data, self.x_data, self.y_data, self.u_data,
                                       self.t_bc[b_idx], self.x_bc[b_idx], self.y_bc[b_idx])
            if not torch.isfinite(losses[0]):
                raise FloatingPointError("Non-finite L-BFGS objective")
            losses[0].backward()
            closure.values = [float(v.detach()) for v in losses]
            return losses[0]

        previous = None
        for epoch in range(1, epochs + 1):
            optimizer.step(closure)
            current = closure.values[0]
            self.lbfgs_history.append({"iteration": epoch, "total": current,
                                       "components": closure.values[1:]})
            if previous is not None and abs(current - previous) <= tol * max(1.0, abs(previous)):
                break
            previous = current
        return float(self.lbfgs_history[-1]["total"])

    def damage_parameters(self):
        return {"alpha": self.alpha.detach().cpu().tolist(),
                "x": self.x_i.detach().cpu().tolist(),
                "y": self.y_i.detach().cpu().tolist(),
                "radius_mm": (self.r_i.detach().cpu() * 1000.0).tolist()}

    def evaluate(self, dataset=None):
        """Return physical-unit RMSE, MAE, relative L2, and damage estimates."""
        t, x, y, u = dataset if dataset is not None else (
            self.t_data, self.x_data, self.y_data, self.u_data)
        t, x, y, u = [torch.as_tensor(v, dtype=self.dtype, device=self.device).reshape(-1, 1)
                      for v in (t, x, y, u)]
        with torch.no_grad():
            error = self.forward(t, x, y) - u
            rmse = torch.sqrt(error.square().mean()).item()
            mae = error.abs().mean().item()
            reference_norm = torch.linalg.vector_norm(u)
            relative_l2 = (None if reference_norm <= 100 * torch.finfo(self.dtype).eps else
                           (torch.linalg.vector_norm(error) / reference_norm).item())
        return {"rmse": rmse, "mae": mae, "relative_l2": relative_l2,
                "n": len(u), "damage": self.damage_parameters()}

    def _config_hash(self):
        payload = json.dumps(C.get_config(), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": C.SCHEMA_VERSION, "config_sha256": self._config_hash(),
                    "python": platform.python_version(), "torch": torch.__version__,
                    "net": self.net.state_dict(), "damage": {
                        "raw_alpha": self.raw_alpha.detach(), "raw_x_i": self.raw_x_i.detach(),
                        "raw_y_i": self.raw_y_i.detach(), "raw_r_i": self.raw_r_i.detach()},
                    "u_mean": self.u_mean.detach(), "u_scale": self.u_scale.detach(),
                    "stage": self.stage, "global_step": self.global_step,
                    "history": self.history, "damage_history": self.damage_history,
                    "lbfgs_history": self.lbfgs_history,
                    "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                    "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, path)

    def load_model(self, path, strict_config=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("schema_version") != C.SCHEMA_VERSION:
            raise ValueError("Checkpoint schema is incompatible with this code version")
        if strict_config and checkpoint.get("config_sha256") != self._config_hash():
            raise ValueError("Checkpoint configuration differs from the active configuration")
        self.net.load_state_dict(checkpoint["net"])
        with torch.no_grad():
            for name, parameter in (("raw_alpha", self.raw_alpha), ("raw_x_i", self.raw_x_i),
                                    ("raw_y_i", self.raw_y_i), ("raw_r_i", self.raw_r_i)):
                parameter.copy_(checkpoint["damage"][name].to(self.device))
            self.u_mean.copy_(checkpoint["u_mean"].to(self.device))
            self.u_scale.copy_(checkpoint["u_scale"].to(self.device))
        self.stage = int(checkpoint["stage"]); self.global_step = int(checkpoint["global_step"])
        self.history = checkpoint["history"]; self.damage_history = checkpoint["damage_history"]
        self.lbfgs_history = checkpoint.get("lbfgs_history", [])
        self.set_stage(self.stage)
        if checkpoint.get("optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())

    def save_damage_params(self, path):
        params = self.damage_parameters()
        np.savez(path, **{key: np.asarray(value) for key, value in params.items()})

    def spatial_gradient(self, t, x, y):
        t = t.detach().clone().requires_grad_(True)
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        prediction = self.forward(t, x, y)
        ux, uy = torch.autograd.grad(prediction, (x, y), torch.ones_like(prediction))
        return ux.detach(), uy.detach()
