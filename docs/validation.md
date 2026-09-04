# Minimum validation protocol for academic use

Define the protocol before inspecting test results.

1. **Forward-operator check.** Run `pytest`; the manufactured mode must satisfy
   both the healthy PDE and the simply-supported edge operator.
2. **Synthetic recovery.** Generate cases spanning location, radius,
   perturbation amplitude, and number of sites. Report parameter error and
   field error for every case, including failures.
3. **Negative control.** Apply Stages 2-3 to held-out healthy data. Report the
   inferred total perturbation; a method that always predicts damage is not
   valid.
4. **Noise robustness.** Repeat at pre-defined signal-to-noise ratios using at
   least 20 independent noise realizations per level.
5. **Observation ablation.** Vary sensor density and temporal sampling without
   reusing test sensors during training.
6. **Initialization sensitivity.** Run multiple pre-registered random seeds and
   initial parameter sets. Report median, interquartile range, and all failure
   counts rather than the best run.
7. **Baseline comparison.** Compare against a non-PINN inverse method or a
   finite-element optimization baseline with the same observations.
8. **Experimental validation.** Calibrate material, thickness, boundary
   conditions, timing, and displacement units independently. Report mismatch
   and uncertainty; do not infer general structural damage from a mass-only
   surrogate.

Archive raw data, preprocessing code, split manifests, all run directories,
the exact Git commit, dependency lock information, and hardware/runtime details.
