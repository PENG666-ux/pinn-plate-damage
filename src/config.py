# config.py  —  Kirchhoff-Love plate damage identification PINN
# All hyper-parameters and physical constants in one place.
# Edit DATA_HEALTHY / DATA_DAMAGE to point to your COMSOL output folders.

import os

# ── Network architecture ───────────────────────────────────────────────────────
F_IN        = 3          # inputs: (t, x, y)
F_OUT       = 1          # output: normalised deflection w
WIDTH       = 50
DEPTH       = 5
W_INIT      = "xavier"
B_INIT      = "zeros"
ACTIVATION  = "tanh"
LAAF        = False

# ── Optimiser ─────────────────────────────────────────────────────────────────
LR_NET      = 1e-3
LR_DAMAGE   = 1e-3       # network lr in Stage 2/3 = LR_DAMAGE * 0.1
F_SCL       = "minmax"

# ── Physical constants ────────────────────────────────────────────────────────
E           = 6.90e10    # Young's modulus (Pa)
NU          = 0.33       # Poisson's ratio
RHO0        = 2700.0     # background density (kg/m3)
H_PLATE     = 0.001      # plate thickness (m)
X_PHYSICAL  = 0.3        # plate length (m)
Y_PHYSICAL  = 0.3        # plate width  (m)
T_PHYSICAL  = 8e-5       # physical time scale (s)
DELTA_SIGMA = 300.0      # fixed areal-density change of damage (kg/m2)
BETA        = 60.0       # sigmoid sharpness (fixed)

# ── Damage parameterisation ────────────────────────────────────────────────────
K_MAX          = 3
INIT_POSITIONS = [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]
INIT_RADII_MM  = [9.0, 3.0, 6.0]
INIT_ALPHA     = [0.5, 0.5, 0.5]

# ── Loss weights ───────────────────────────────────────────────────────────────
W_PDE       = 1.0
W_DATA_S1   = 10000.0
W_DATA_S2   = 100.0
W_DATA_S3   = 100.0
W_REG_R_S2  = 1e-5
W_REG_G_S2  = 1e-4
W_REG_R_S3  = 1e-5
W_REG_G_S3  = 1e-4

# ── Sampling ───────────────────────────────────────────────────────────────────
N_PDE            = 20000
PDE_LBFGS_SUBSET = 5000
GRID_SIZE        = 15
N_TIME_POINTS    = 80
SAMPLING_TOL     = 0.01

# ── Training schedule ─────────────────────────────────────────────────────────
S1_ADAM_EPOCHS           = 9000
S1_LBFGS_EPOCHS          = 1000
S2_EPOCHS                = 1000
S3_EPOCHS                = 1000
ADAPTIVE_UPDATE_INTERVAL = 500
S2_EARLY_PHASE           = 2000

# ── Misc ───────────────────────────────────────────────────────────────────────
F_MNTR = 50
R_SEED = 1234

# ── Data paths ─────────────────────────────────────────────────────────────────
# Point to folders containing COMSOL-exported CSV files (columns: t, x, y, u).
_HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_HEALTHY = os.path.join(_HERE, "..", "data", "healthy")
DATA_DAMAGE  = os.path.join(_HERE, "..", "data", "damaged")


def get_config():
    return dict(
        f_in=F_IN, f_out=F_OUT, width=WIDTH, depth=DEPTH,
        w_init=W_INIT, b_init=B_INIT, act=ACTIVATION, laaf=LAAF,
        f_scl=F_SCL, lr=LR_NET, lr_damage=LR_DAMAGE,
        E=E, nu=NU, rho0=RHO0, h=H_PLATE,
        X_physical=X_PHYSICAL, Y_physical=Y_PHYSICAL, T_physical=T_PHYSICAL,
        delta_sigma=DELTA_SIGMA, beta=BETA, K_max=K_MAX,
        init_positions=INIT_POSITIONS, init_radii_mm=INIT_RADII_MM, init_alpha=INIT_ALPHA,
        w_pde=W_PDE,
        w_data_s1=W_DATA_S1, w_data_s2=W_DATA_S2, w_data_s3=W_DATA_S3,
        w_reg_r_s2=W_REG_R_S2, w_reg_g_s2=W_REG_G_S2,
        w_reg_r_s3=W_REG_R_S3, w_reg_g_s3=W_REG_G_S3,
        N_PDE=N_PDE, pde_lbfgs_subset=PDE_LBFGS_SUBSET,
        grid_size=GRID_SIZE, num_time_points=N_TIME_POINTS, sampling_tol=SAMPLING_TOL,
        s1_adam=S1_ADAM_EPOCHS, s1_lbfgs=S1_LBFGS_EPOCHS,
        s2_epochs=S2_EPOCHS, s3_epochs=S3_EPOCHS,
        adaptive_interval=ADAPTIVE_UPDATE_INTERVAL, s2_early_phase=S2_EARLY_PHASE,
        f_mntr=F_MNTR, r_seed=R_SEED, t_max_star=1.0,
        data_healthy=DATA_HEALTHY, data_damage=DATA_DAMAGE,
    )
