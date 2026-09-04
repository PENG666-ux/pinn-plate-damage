# Method and assumptions

## Governing equation

For an isotropic Kirchhoff-Love plate without forcing or damping,

$$
D\left(w_{,XXXX}+2w_{,XXYY}+w_{,YYYY}\right)+\sigma(X,Y)w_{,\tau\tau}=0,
\qquad D=\frac{Eh^3}{12(1-\nu^2)}.
$$

Let $x=X/L_x$, $y=Y/L_y$, and $t=\tau/T$. After multiplication by
$L_x^4/D$, the implemented dimensionless residual is

$$
f=w_{,xxxx}+2a^2w_{,xxyy}+a^4w_{,yyyy}
+\frac{\sigma(x,y)L_x^4}{DT^2}w_{,tt},\qquad a=L_x/L_y.
$$

For numerical conditioning, the code divides the complete residual by the
healthy inertia coefficient. This multiplication by a non-zero constant does
not change the PDE solution set.

## Parameterized perturbation

The effective areal density is

$$
\sigma(x,y)=\rho h+\sum_{k=1}^{K}\alpha_k\Delta\sigma\,
S\!\left(\beta[r_k-d_k(x,y)]\right),
$$

where $d_k=\sqrt{(x-x_k)^2+(y-y_k)^2}$ and $S$ is the logistic function.
The signed-distance form makes $1/\beta$ an interpretable normalized transition
width. Parameters are mapped smoothly to bounded domains:
$\alpha_k\in(0,1)$, $x_k,y_k\in(0.04,0.96)$, and
$r_k\in(R_{\min},R_{\max})$.

The perturbation is additive mass per area. A positive `DELTA_SIGMA` cannot
represent stiffness loss, a void, or a crack unless the forward model is
changed and independently validated.

## Boundary conditions

The implementation assumes simply-supported edges. It penalizes $w=0$ and
zero normal bending moment. In normalized coordinates,

$$
x\text{-edges}:\quad w_{,xx}+\nu a^2 w_{,yy}=0,
\qquad
y\text{-edges}:\quad a^2w_{,yy}+\nu w_{,xx}=0.
$$

Data from a clamped or free plate are incompatible with these constraints.
Damping and external forcing are not currently modelled.

## Objective and stages

The objective is a weighted sum of standardized measurement MSE, PDE-residual
MSE, boundary residual MSE, an amplitude-weighted area penalty, and an L1-like
amplitude penalty. Stage 1 fits a healthy baseline; Stage 2 jointly updates the
field network and perturbation parameters; Stage 3 increases collocation
density near active estimates. Stage-1 L-BFGS uses one fixed collocation subset
for every closure, which is required for a consistent line-search objective.

## Identifiability

Inverse recovery is not guaranteed merely because the PINN loss is small.
Multiple mass fields may yield similar sparse vibration observations, and the
network may trade data fit against the physics penalty. Claims about recovered
locations or sizes therefore require synthetic recovery experiments, negative
controls, noise and sensor ablations, repeated initializations, and preferably
comparison with a conventional inverse or finite-element baseline.
