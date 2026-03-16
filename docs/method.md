# Method: PINN-Based Damage Identification for Kirchhoff-Love Plates

## 1. Governing Equation

The out-of-plane deflection $w(t, x, y)$ of a thin plate obeys the
Kirchhoff-Love equation:

$$
\nabla^4 w + \frac{\sigma^*(x,y)\, L^4}{D\, T^2}\, \frac{\partial^2 w}{\partial t^2} = 0
$$

where $D = \frac{E h^3}{12(1-\nu^2)}$ is the bending stiffness, $L$ the plate
side length, and $T$ the physical time scale used for normalisation.

For a **healthy** plate the effective areal density is constant:

$$\sigma^*(x,y) = \rho h$$

For a **damaged** plate, each damage site $k$ introduces a localised mass
perturbation modelled by a smooth indicator function:

$$
\sigma^*(x,y) = \rho h \;+\;
\sum_{k=1}^{K} \alpha_k\, \Delta\sigma\;
\text{sigmoid}\!\left[\beta\!\left(r_k^2 - d_k^2(x,y)\right)\right]
$$

where $d_k^2(x,y) = (x - x_k)^2 + (y - y_k)^2$ is the squared distance from
damage centre $k$, $r_k$ its radius, $\alpha_k \in [0,1]$ the intensity, and
$\beta$ the sharpness of the indicator.

The inverse problem is to recover $\{x_k, y_k, r_k, \alpha_k\}_{k=1}^K$ from
spatially sparse, multi-snapshot measurements of $w$.

---

## 2. Neural Network Ansatz

A fully-connected network $\mathcal{N}_\theta : (t, x, y) \mapsto \hat{w}$
approximates the deflection field.  
Architecture: 5 hidden layers × 50 neurons, tanh activation, Xavier
initialisation, min-max feature scaling to $[-1, 1]$.

---

## 3. Three-Stage Training

### Stage 1 – Healthy Baseline

Using measurements from the undamaged plate, minimise:

$$\mathcal{L}_1 = w_\text{data}\,\mathcal{L}_\text{data} + w_\text{pde}\,\mathcal{L}_\text{PDE}$$

$$\mathcal{L}_\text{data} = \frac{1}{N_d}\sum_i\!\left(\hat{w}_i - w_i^\text{meas}\right)^2, \qquad
\mathcal{L}_\text{PDE} = \frac{1}{N_c}\sum_j f(\hat{w})^2_j$$

All $\alpha_k = 0$ (damage parameters are frozen). Optimiser: Adam (9 000
epochs) followed by L-BFGS with strong Wolfe line search (1 000 iterations).

### Stage 2 – Joint Optimisation

The network weights are initialised from Stage 1 and the damage parameters are
unfrozen. Both sets of parameters are updated simultaneously using measurements
from the damaged plate:

$$\mathcal{L}_2 = w_\text{data}\,\mathcal{L}_\text{data}
+ w_\text{pde}\,\mathcal{L}_\text{PDE}
+ \mathcal{L}_\text{reg,r}
+ \mathcal{L}_\text{sparsity}$$

- **Radius regularisation** $\mathcal{L}_\text{reg,r}$: soft penalty that
  keeps the primary damage radius in a physically reasonable band.
- **Sparsity regularisation** $\mathcal{L}_\text{sparsity}$: encourages one
  dominant $\alpha_k \to 1$ while suppressing the rest toward $0$,
  implementing an $\ell_1$-like prior on the damage configuration.

The network learning rate is set to $0.1 \times$ the damage-parameter learning
rate to prevent the network from "absorbing" the damage signal.

### Stage 3 – Damage Refinement

Starting from Stage-2 parameters, training continues with:

- 80 % of collocation points concentrated near the current damage estimate
  (high-resolution adaptive sampling).
- The network learning rate is unchanged; the damage parameters continue to
  evolve.

---

## 4. Adaptive Collocation Sampling

In Stages 2 and 3 the collocation set is refreshed every
`ADAPTIVE_UPDATE_INTERVAL` epochs according to the current damage estimate:

| Stage / Phase | Damage region | Global uniform | Boundary |
|---------------|--------------|----------------|---------|
| Stage 2 early | 0 %  (LHS)   | 80 %           | 20 %    |
| Stage 2 late  | 60 %         | 30 %           | 10 %    |
| Stage 3       | 80 %         | 15 %           | 5 %     |

Damage-focused points are drawn from a Gaussian centred on each active site,
with the inner 70 % sampled uniformly inside the disc and the outer 30 %
sampled from an annular Gaussian.

---

## 5. Parameter Constraints

Raw network outputs are passed through smooth bijections to enforce physical
bounds:

| Parameter | Constraint | Mapping |
|-----------|-----------|---------|
| $\alpha_k$ | $\in (0, 1)$ | $\alpha_k = \text{sigmoid}(\tilde\alpha_k)$ |
| $x_k, y_k$ | $\in [0.04, 0.96]$ (away from boundary) | sigmoid + affine |
| $r_k$     | $\in [1\,\text{mm},\; 15\,\text{mm}]$ | softplus + affine |

---

## 6. Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| $\text{RMSE}_\text{data}$ | Root-mean-square error on held-out measurements |
| $e_\text{pos}$ | Euclidean distance (mm) between inferred and reference damage centre |
| $e_\text{size}$ | Relative radius error $|r_\text{inf} - r_\text{ref}| / r_\text{ref}$ |
