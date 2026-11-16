# Paper alignment and reproduction boundaries

## Reference and source precedence

The reference reviewed for this implementation is the user-supplied 24-page manuscript:

**U-PINN: Damage inference in thin-plate structures from sparse UGW sensor data using a two-stage physics-informed neural network**, by Lu Zhou, Zhi-Peng Wang, Wenying Duan, Zhenyu Sun, Yunke Luo, You Dong, and Yuxuan Rao.

No verified DOI, publication venue, or public version identifier was supplied. Page references below refer to physical pages of that PDF, not an inferred journal pagination. The manuscript is not distributed in this repository.

The accompanying “Research Observatory” HTML is a secondary explanatory artifact. Its visualizations are not numerical PINN predictions or evidence of a completed training run. Where it differs from the PDF, this implementation's documentation follows the PDF. For example, the HTML lists Stage II measurement count as 100,000, whereas PDF Table 1 lists 6,250.

## Equation and workflow map

| Paper source | Paper statement | Repository treatment |
|---|---|---|
| p. 3, Eq. (1) | Kirchhoff–Love plate dynamics | Full spatial biharmonic operator, including the factor 2 on the mixed derivative |
| pp. 3–4, Eqs. (5)–(7) | Local equivalent areal-density perturbations | Smooth circular candidate indicators multiply acceleration |
| p. 4, Eq. (6) | Sigmoid of signed **distance**, $\beta(r_i-d_i)$ | Uses Euclidean distance, rather than the earlier squared-distance indicator |
| p. 4, Eqs. (8)–(9) | Incident/scattered decomposition | Physical motivation; not substituted for the complete training residual |
| pp. 4–5, Sec. 3.1 | Five hidden layers, 50 neurons each, Tanh | Standard architecture; deliberately reduced in smoke mode |
| pp. 5–6, Sec. 3.2.1 | Healthy simulation pretraining, then experimental transfer | Optional simulation input; transfer trains only the last hidden layer and output layer |
| p. 5, Eqs. (11)–(14) | Unified measurement MSE and independent LHS collocation | No invented idealized BC/IC loss |
| pp. 7–8, Sec. 3.2.2 | Three consecutive inversion sub-stages | Freeze radii, then release them, then freeze the entire wavefield network |
| pp. 7–8, Eqs. (19)–(21) | Inactive-radius and gate penalties | $(1-\alpha)r^2$ and $\alpha(1-\alpha)$, respectively |
| p. 8 | Float64 training | Double-precision model and data |
| pp. 20–21, Appendix 4 | Coordinate and displacement normalization, PDE rescaling | Explicit separate $L_x,L_y,L_t$ derivative coefficients; consistent coordinates across datasets |

The gate term is zero at both $\alpha=0$ and $\alpha=1$. Calling it an $\ell_1$ sparsity penalty, a guarantee of fewer defects, or unconstrained defect-count estimation would overstate what the formula supports. Candidate count is bounded in advance by $K_{\max}$.

The prescribed density increment and boundary-sharpness value depend on the length convention. Current implementation parameters and optimizer/sampling choices are engineering choices wherever the source is incomplete. Matching a few architecture and loss settings is not equivalent to reproducing the experiments.

### Deliberate implementation choices

- The normalized [0,1] coordinates are internally mapped to [-1,1] before the network; automatic differentiation retains that mapping.
- Adam uses piecewise-constant rates per training sub-stage. L-BFGS additionally refines healthy direct fitting/pretraining. The implementation does not reproduce the paper's incompletely specified inverse-time LR decay.
- Collocation mixes global LHS samples, edge samples, and later damage-focused samples, with periodic refresh. The paper describes independent LHS collocation; these mixtures are engineering extensions, and edge PDE samples are not added idealized boundary conditions.
- The radius regularizer uses $(r_i/L_x)^2$. The manuscript does not specify its radius-scale convention sufficiently to claim numerical identity of the regularized objective.
- Centers are restricted to (0.04,0.96) in each normalized coordinate; radii use a softplus map with a 1 mm lower bound and no upper bound. Those bounds are implementation assumptions, not confirmed paper constraints.
- Checkpoints preserve physical metadata and support stage transfer. They do not preserve optimizer/RNG state or provide exact training resumption.

## Experimental conditions reported by the paper

- Aluminum plate: 300 × 300 × 1 mm; circular PZT-5H transducers: 5 mm radius and 1 mm thickness.
- One center actuator at (150,150) mm and one receiver moved between measurement locations.
- Attached masses emulate local damage; the paper explicitly notes they do not fully represent cracks or holes.
- Excitation: a five-cycle, 250 kHz tone burst, lasting 20 μs; 25 MHz acquisition, 16,000 points per waveform, three measurements averaged.
- Hardware pass band: 9 kHz–5 MHz; analysis window: 50–300 μs with a 0.04 μs sample interval; traces aligned to excitation onset.
- Healthy simulation uses COMSOL Multiphysics 6.2. The paper uses bonded contact and a fitted non-sinusoidal excitation.
- Table 1 reports 36,000 Stage I observations, 6,250 Stage II observations, and 50,000 PDE collocation points in each stage.
- Three initial candidates have centers (150,150), (90,210), (210,90) mm; radii 3, 6, 9 mm; and gates initialized at 0.5. The discussed boundary parameter is $\beta=100$.

Sources: PDF pp. 8–11, 14, 16. These are source statements, not available datasets or measured hardware characteristics of this repository.

## Reported results

All values in this table are transcribed from PDF Tables 2–3 (pp. 11, 14). **They are not outputs of the code in this repository.**

| Case/defect | Reference center (mm) | Reference radius (mm) | Predicted center (mm) | Predicted radius (mm) | Localization error (mm) | Radius error |
|---|---|---:|---|---:|---:|---:|
| I | (70,70) | 5 | (74.89,71.33) | 5.92 | 5.07 | 18.4% |
| II | (130,70) | 5 | (131.57,77.78) | 4.71 | 7.94 | 5.8% |
| III, small defect | (70,70) | 5 | (63.88,68.51) | 4.16 | 6.30 | 16.8% |
| III, large defect | (210,210) | 8 | (219.52,215.08) | 7.44 | 10.79 | 7.0% |

The mean over **four defect locations**, rather than three equally weighted cases, is approximately 7.52 mm. Rounded table entries yield 7.525 mm; the manuscript reports 7.52 mm. Reported wavefield RMSE values are 2.246e−3, 7.227e−3, and 7.895e−2 for Cases I, II, and III.

Appendix 1 (p. 18) reports 13.99 mm mean localization error for a supervised CNN and a 46.2% reduction with U-PINN. The CNN is trained using labeled damage scenarios; U-PINN performs per-case physics-informed inversion using that case's observations. This is not evidence that the two approaches used identical learning procedures or supervision.

Appendices 2–3 (pp. 18–20) report benefits from healthy simulation-to-experiment transfer and progressive inversion. These conclusions apply to the investigated comparisons, not universal convergence guarantees.

## Source inconsistencies and missing information

| Item | What the PDF says | How to handle it |
|---|---|---|
| Simulation count, p. 10 | 225 positions × 80 time steps, but 36,000 total samples | The product is 18,000; do not fabricate the missing samples |
| Simulation times, p. 10 | 20–100 μs at 0.5 μs spacing, also described as 80 time steps | The interval contains about 160 increments; the intended endpoint/count convention needs confirmation |
| Stage I initial LR, Table 4 p. 15 | Printed as $1.0\times10^3$ | Visually checked in PDF; treat as a likely typographic error requiring author confirmation. It is not a safe training default |
| Optimizer and duration, p. 14/Table 4 | Text says optimizer and epoch counts are summarized, but the table does not include them | Label repository optimizer choices and durations as engineering defaults |
| Learning-rate schedule, p. 5 | Inverse-time decay is mentioned without a complete schedule specification | Do not describe a chosen decay rate or schedule as an exact paper setting |
| Density increment, Eqs. (5)–(6) | Defines $\Delta\sigma_i$ but supplies no numeric value | Calibrate it from experimental information; record the assumption explicitly |
| Normalization coefficient, Appendix 4 | Compact $C_{\mathrm{pde}}$ notation and normalized biharmonic operator | Implement the fully expanded chain-rule coefficients; retain fourth powers of length and mixed scales |
| Sharpness and radius | $\beta=100$ is stated, while radius/sharpness equivalence is discussed | State the coordinate convention; an inferred radius is an equivalent model parameter |
| Reproduction inputs | No original signals, trained weights, seed, full epoch boundaries, or complete preprocessing provenance is included in the supplied materials | A method implementation can be checked, but experimental accuracy cannot be independently reproduced from these materials alone |

The repository also contains engineering assumptions for elastic properties, gate/position/radius bounds, candidate reporting thresholds, checkpoint layout, and data reduction. They should be recorded with each run and revisited for a new plate or sensor layout.

## What the synthetic run establishes

The generated fixture contains an analytic **healthy** modal solution written into both input folders. It supplies no true damaged-wavefield solution, mass location, or inversion target. A successful smoke run establishes execution of the pipeline and serialization of finite outputs. Mathematical regression checks separately test derivative scaling and freeze/update behavior. Neither constitutes a reproduction of the manuscript's damage-localization or radius-error values.

For a scientific reproduction, obtain the source data and missing settings, record physical coordinate/time/amplitude scales, use a documented receiver/scenario partition, train to convergence, and evaluate every reference defect with a stated matching and activation rule.
