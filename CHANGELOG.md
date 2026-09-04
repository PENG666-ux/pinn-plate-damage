# Changelog

## 1.1.0 - 2026-09-04

- Corrected the rectangular-plate nondimensional operator.
- Replaced the squared-distance damage gate with a signed-distance gate.
- Added explicit simply-supported displacement and moment residuals.
- Added train/validation/test partitioning by complete sensor location.
- Added standardized displacement training and scaled PDE residuals.
- Made sampling and L-BFGS closure objectives deterministic.
- Enforced truly bounded damage radii and removed the privileged-radius prior.
- Added versioned, configuration-checked, resumable checkpoints.
- Added manufactured-solution, sampling, leakage, and constraint tests plus CI.
- Replaced ground-truth-dependent figures with auditable inference plots.
- Documented model scope, limitations, and a minimum validation protocol.
