# Quick start

## Installation

Use Python 3.10 or later and run commands from the repository root:

~~~bash
python -m venv .venv
~~~

Activate the environment using the command for your shell:

~~~powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
~~~

~~~bash
# Linux/macOS
source .venv/bin/activate
~~~

Install dependencies:

~~~bash
python -m pip install -r requirements.txt
~~~

CPU execution is supported. CUDA is optional; <code>--device auto</code> chooses an available CUDA device or falls back to CPU. Reported timings from the paper are not supplied as runtime promises for this implementation.

## Data contract

Each data folder contains one or more CSV files with headers:

~~~csv
t,x,y,u
0.000000,0.100000,0.200000,0.012000
0.500000,0.100000,0.200000,-0.008000
~~~

This is a schema illustration, not a training dataset.

| Column | Meaning | Required convention |
|---|---|---|
| t | Normalized time | $(t_{\mathrm{physical}}-t_{\min})/L_t$, within [0,1] |
| x | Normalized x coordinate | $(x_{\mathrm{physical}}-x_{\min})/L_x$, within [0,1] |
| y | Normalized y coordinate | $(y_{\mathrm{physical}}-y_{\min})/L_y$, within [0,1] |
| u | Normalized out-of-plane displacement | One common amplitude scale, $u_{\mathrm{physical}}/U$ |

Use **one shared coordinate origin, time origin, and set of scales** for simulation, healthy, and damaged folders. In particular:

- Do not independently min–max normalize each folder.
- Do not independently normalize each sensor waveform to unit amplitude; this discards relative amplitude information needed by the inverse problem.
- Fit any data-derived scaling using training observations and reuse it for validation and inversion.
- Preserve preprocessing records: units, coordinate and time origins, scale factors, alignment, filtering, and selected time window.
- Use only aligned, comparable wavefield observations. Do not mix seconds and microseconds or meters and millimeters in a column.
- Avoid distributing repeated measurements of the same damage configuration across training and test subsets when comparing localization performance.

Set the corresponding physical lengths and time span in <code>src/config.py</code>: <code>X_PHYSICAL</code>, <code>Y_PHYSICAL</code> in meters and <code>T_PHYSICAL</code> in seconds. The CSV loader does not infer the physical units or perform signal filtering/alignment. Save the physical origins and amplitude scale alongside your dataset so predictions can be converted back to measurement units.

Spatially sparse measurements do not need to form a complete Cartesian grid. The pipeline reduces long time series per actual receiver location rather than inventing missing measurements.

## Synthetic execution check

Generate the fixture and run the reduced CPU configuration:

~~~bash
python scripts/generate_demo.py --output data/demo
python -m src.main --healthy data/demo/healthy --damaged data/demo/damaged --output results/demo --device cpu --smoke
~~~

Both folders contain the **same healthy analytic mode**. “Damaged” is a folder-role label, not a claim of a synthetic defect. The smoke run uses a small network, few independent collocation points, and only a few optimizer steps. It checks that the complete pipeline runs and writes artifacts. It cannot establish optimization convergence, accurate damage inference, or false-positive rejection.

To exercise the optional transfer branch with the same execution fixture:

~~~bash
python -m src.main --pretrain data/demo/healthy --healthy data/demo/healthy --damaged data/demo/damaged --output results/demo-transfer --device cpu --smoke
~~~

Using identical data in both healthy roles checks the transfer code path only; it does not demonstrate simulation-to-experiment adaptation.

## Training with your observations

For a healthy baseline learned directly from measurements:

~~~bash
python -m src.main --healthy data/healthy --damaged data/damaged --output results/experiment --device auto
~~~

For the paper's simulation-to-experiment workflow:

~~~bash
python -m src.main --pretrain data/simulation --healthy data/healthy --damaged data/damaged --output results/transfer --device auto --plots
~~~

The optional <code>--pretrain</code> argument identifies healthy simulation CSV files. The subsequent healthy phase fine-tunes the network's final hidden and output layers. Without it, the healthy phase trains the full network directly and skips that transfer step.

Both invocations then run Stage II: gates/centers with fixed radii, gates/centers/radii, and finally damage-only refinement with the complete network frozen. Network learning rates are smaller than damage learning rates during the first two inversion sub-stages.

The script entry point <code>python src/main.py</code> accepts the same arguments. Use <code>--seed</code> to record a different random seed; matching a seed does not guarantee identical GPU behavior across environments. Inspect the current interface with:

~~~bash
python -m src.main --help
~~~

## Outputs

All run artifacts are placed beneath <code>--output</code>:

| File | Contents |
|---|---|
| config.json | Resolved run settings, including smoke overrides |
| metrics.json | Run diagnostics and inferred candidate parameters |
| pretrain_model.pth | Healthy simulation model, only when pretraining is requested |
| stage1_model.pth | Healthy model after direct fitting or transfer |
| stage2_model.pth | Model after the first two inversion sub-stages |
| stage3_final_model.pth | Final model after damage-only refinement |
| damage_params_final.npz | Final candidate parameters |
| stage1_loss.png, stage2_loss.png, stage3_loss.png | Loss curves, enabled with <code>--plots</code> |
| pretrain_loss.png | Pretraining loss curve, when pretraining and plotting are enabled |
| damage_estimate.png | Candidate estimate diagnostic, enabled with <code>--plots</code> |

The filenames retain earlier implementation stage numbering: “stage1” is scientific Stage I, “stage2” includes the first two scientific Stage II sub-stages, and “stage3” is Stage II's final network-frozen sub-stage.

In the NPZ output, <code>x_i</code> and <code>y_i</code> are normalized centers, <code>r_i</code> is physical radius in meters, and <code>r_i_norm</code> is radius divided by <code>X_PHYSICAL</code>. The JSON candidate entries additionally report centers and radii in millimeters. Center distances are measured from the normalized domain's physical origin; add any nonzero physical origin when mapping them into the original specimen coordinate system.

A data-fit RMSE from training observations must not be labeled a held-out score. Ground-truth localization errors are not computed merely by fitting waveforms. Record all active and inactive candidates, and use a predefined matching rule if comparing multiple inferred defects with known geometry.

## Configuration

Edit <code>src/config.py</code> before a non-smoke experiment:

- Material parameters: <code>E</code>, <code>NU</code>, <code>RHO0</code>, <code>H_PLATE</code>.
- Physical coordinate and time scales: <code>X_PHYSICAL</code>, <code>Y_PHYSICAL</code>, <code>T_PHYSICAL</code>.
- Density increment and boundary profile: <code>DELTA_SIGMA</code> in kg/m² and <code>BETA</code> applied to distances divided by <code>X_PHYSICAL</code>.
- Candidate count and initialization: <code>K_MAX</code>, <code>INIT_POSITIONS</code>, <code>INIT_RADII_MM</code>, <code>INIT_ALPHA</code>. Keep all candidate arrays the same length; initial centers are in normalized coordinates and radii are in millimeters.
- Network width/depth, data/collocation batch sizes, loss weights, learning rates, and training durations.

The original paper does not completely specify several of these settings. The configured density increment, optimizer choices, sampling mixtures, and durations should be treated as assumptions until calibrated. The implemented position mapping restricts centers to normalized coordinates (0.04,0.96); radii are greater than 1 mm without a finite upper bound. These bounds are fixed in the parameter mappings in <code>src/pinn.py</code>, rather than inferred from the supplied measurements. For the full list, see [paper_alignment.md](paper_alignment.md).

Checkpoints use format version 2 and retain network weights, layer sizes, damage parameters, physical constants, fixed input bounds, loss weights, and histories. They support stage transfer and weights-only loading; legacy checkpoints without physical metadata are rejected. They do **not** store optimizer state or random-generator state and do not provide exact training resumption. External physical origins, amplitude scaling, and signal-preprocessing provenance remain part of the dataset records, not inferred checkpoint metadata.

Changing the architecture or material constants and then loading an old checkpoint is not a new calibrated experiment. Re-run training when the intended physical setup changes.

## Verification

Run the mathematical and workflow checks:

~~~bash
python -m pip install pytest
python -m pytest -q
~~~

Then run the smoke commands above. Automated checks and finite smoke outputs establish software behavior. Experimental reproduction still requires the original signals, complete settings, and independently evaluated reference damage geometry.
