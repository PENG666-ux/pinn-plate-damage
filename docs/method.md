# Method and implementation

This document describes the paper-informed method implemented here. References such as “p. 7, Eq. (18)” refer to the physical PDF pages of the supplied 24-page U-PINN manuscript. The source-to-implementation map and unresolved settings are recorded in [paper_alignment.md](paper_alignment.md).

## Plate equation and damage field

The transverse displacement of the healthy plate satisfies

$$
\sigma_0 w_{tt}+D\nabla^4w=0,\quad
\sigma_0=\rho h,\quad D=\frac{Eh^3}{12(1-\nu^2)},
\quad \nabla^4w=w_{xxxx}+2w_{xxyy}+w_{yyyy}.
$$

This is the Kirchhoff–Love flexural-wave approximation, not a full three-dimensional elastodynamic model (p. 3, Eq. (1)). An unknown damage configuration is represented by a bounded pool of candidates:

$$
\sigma^*(x,y)=\sigma_0+
\sum_{i=1}^{K_{\max}}\alpha_i\Delta\sigma_i S_i(x,y),
\quad
S_i=\frac{1}{1+\exp[\beta(d_i-r_i)]},
\quad d_i=\sqrt{(x-x_i)^2+(y-y_i)^2}.
$$

Each candidate has a gate $\alpha_i\in[0,1]$, center $(x_i,y_i)$, and nominal radius $r_i$ (pp. 3–4, Eqs. (5)–(7)). $\Delta\sigma_i$ is a prescribed areal-density increment; the manuscript does not supply its numerical value. It must be calibrated for a real experiment. The code's configured value is an assumption, not a recovered paper parameter.

The damage field multiplies acceleration:

$$
\sigma^*(x,y)w_{tt}+D\nabla^4w=0.
$$

It does not modify bending stiffness. The scattered-field approximation in the manuscript's Eq. (9) motivates this model; training uses the complete damaged-state residual of Eq. (18).

## Coordinates, units, and derivatives

Let physical coordinates be transformed as

$$
\bar x=(x-x_{\min})/L_x,\quad
\bar y=(y-y_{\min})/L_y,\quad
\bar t=(t-t_{\min})/L_t,\quad
\bar w=w/U,
$$

where $U$ is one common max-absolute displacement scale. Coordinate origins and scales must be shared by simulation, healthy, and damaged data. The CSV interface expects these already normalized coordinates and amplitudes. Material quantities use SI units: $E$ in Pa, $h,L_x,L_y$ in m, $L_t$ in s, and $\sigma$ in kg/m².

Dividing the physical equation by $\sigma_0U/L_t^2$ gives

$$
\mathcal R=
\frac{\sigma^*}{\sigma_0}\bar w_{\bar t\bar t}
+\frac{DL_t^2}{\sigma_0L_x^4}\bar w_{\bar x\bar x\bar x\bar x}
+\frac{2DL_t^2}{\sigma_0L_x^2L_y^2}\bar w_{\bar x\bar x\bar y\bar y}
+\frac{DL_t^2}{\sigma_0L_y^4}\bar w_{\bar y\bar y\bar y\bar y}.
$$

This explicit form removes ambiguity from the abbreviated coefficient in Appendix 4 (pp. 20–21) and handles rectangular plates. A healthy density field must reduce exactly to the healthy residual.

Distances used by the damage indicator retain the physical aspect ratio: separately normalized $x$ and $y$ are converted to a common length scale, $L_{\mathrm{ref}}=L_x$, before computing $d_i$. Thus $d_i/L_x=\sqrt{(\bar x-\bar x_i)^2+(L_y/L_x)^2(\bar y-\bar y_i)^2}$, and the indicator uses $r_i/L_x$ as its radius. The configured $\beta$ multiplies these dimensionless distances; its value cannot be transferred unchanged between meters, millimeters, and normalized coordinates.

The wavefield network has five hidden layers of 50 <code>tanh</code> units, accepts coordinates in <code>(t,x,y)</code> order, and returns one normalized displacement. It internally maps the fixed [0,1] coordinate domain to [-1,1] before its linear layers; derivatives are still taken with respect to the external [0,1] coordinates, retaining this affine map in the differentiation graph. Double precision and hierarchical automatic differentiation compute second derivatives before the fourth-order spatial terms. Freezing network weights during final inversion still permits differentiation with respect to the coordinate inputs.

## Losses

For observations and independent collocation points:

$$
\mathcal L_{\mathrm{data}}
=\frac{1}{N_d}\sum_j|\hat w_j-w_j|^2,\qquad
\mathcal L_{\mathrm{PDE}}
=\frac{1}{N_c}\sum_j|\mathcal R_j|^2.
$$

Stage I minimizes $\lambda_d\mathcal L_{\mathrm{data}}+\lambda_p\mathcal L_{\mathrm{PDE}}$ with the healthy density. The manuscript incorporates any observed initial/boundary samples into the unified data term; it does not impose separate idealized initial/boundary losses (p. 5, Eqs. (11)–(14)).

Stage II minimizes

$$
\mathcal L_{\mathrm{II}}
=\lambda_d\mathcal L_{\mathrm{data}}
+\lambda_p\mathcal L_{\mathrm{PDE}}
+\lambda_r\sum_i(1-\alpha_i)r_i^2
+\lambda_s\sum_i\alpha_i(1-\alpha_i).
$$

The radius term suppresses expansion of weakly activated candidates. The gate term penalizes intermediate values and vanishes at both 0 and 1 (pp. 7–8, Eqs. (19)–(21)). It is a binary-gate penalty, not an $\ell_1$ penalty or proof of minimum-cardinality selection. The implemented radius penalty uses $(r_i/L_x)^2$, so its coefficient must be interpreted together with that scale. The manuscript does not clearly specify the radius units used within its regularizer; matching the displayed numerical coefficient alone cannot establish an identical objective.

## Optimization and transfer

1. Optionally pretrain the entire wavefield network on healthy simulation data.
2. If pretraining was provided, transfer to healthy experimental observations by training only the final hidden layer and output layer. Without simulation input, learn the healthy field directly; that run omits the paper's simulation-to-experiment adaptation.
3. Initialize the damaged-state model from the healthy model. First update candidate gates and centers with fixed radii, while allowing the network tail to adapt at a lower learning rate.
4. Release the radii and continue updating all damage parameters and the same network tail.
5. Freeze the entire wavefield network and refine only the damage parameters.

The last three steps are the three sub-stages of **Stage II**, rather than three independent scientific stages (pp. 5–8). Internal implementation stage numbers are explained in [quickstart.md](quickstart.md).

The implementation uses Adam with stage-specific learning rates; healthy direct fitting/pretraining is additionally refined with L-BFGS. For the two joint inversion sub-stages the damage learning rates default to 1e−3 and 5e−4, while the wavefield-tail rate is 5e−5; final damage-only refinement uses 1e−4. These are piecewise-constant rates, not a reproduction of the manuscript's incompletely specified inverse-time decay schedule.

Gates use a sigmoid mapping to (0,1), centers use an affine sigmoid mapping to (0.04,0.96), and radii use $r_i=1\,\mathrm{mm}+14\,\mathrm{mm}\,\operatorname{softplus}(z_i)$. The radius therefore has a lower bound of 1 mm and **no upper bound**; 15 mm is not a maximum. These implementation bounds are engineering assumptions.

Collocation points are independent of measurement samples. The current sampler mixes global Latin hypercube points with edge points, and adds damage-focused samples later in inversion. Edge points enforce the PDE, not a boundary-condition loss. This mixture and adaptive refresh are engineering extensions; the supplied manuscript describes independent LHS collocation and does not establish these additions. Sampling neighborhoods use normalized coordinates and need not be physical circles on a rectangular plate, whereas the density masks preserve the physical aspect ratio.

## Interpreting results

The data RMSE quantifies displacement fit. Training-observation RMSE is not a held-out generalization metric, and a small data loss alone does not validate damage identification. Localization error requires independent reference centers in physical units:

$$
e_{\mathrm{pos},i}=\sqrt{(x_i-x_{i,\mathrm{true}})^2+(y_i-y_{i,\mathrm{true}})^2},\qquad
e_{\mathrm{size},i}=\frac{|r_i-r_{i,\mathrm{true}}|}{r_{i,\mathrm{true}}}\times100\%.
$$

For multiple defects, candidate matching and activation thresholds must be stated before comparing with references; reporting only the largest gate can hide missed defects. Without ground-truth geometry, fit metrics and candidate parameters are diagnostics, not localization accuracy.

The manuscript discusses radius/boundary-sharpness ambiguity (pp. 13, 16). Its attached-mass experiments support this equivalent density model within the tested setting; they do not establish performance for arbitrary cracks, voids, or structural geometries.
