# Quick start

1. Install and verify the environment:

   ```bash
   pip install -e ".[test]"
   pytest
   ```

2. Put finite numeric `t,x,y,u` CSV data in `data/healthy` and `data/damaged`.

3. Confirm all physical constants and the actual edge condition in
   `src/config.py`. In particular, verify that `DELTA_SIGMA` has units of
   kg/m2 and is experimentally defensible.

4. Exercise the complete pipeline cheaply:

   ```bash
   python -m src.main --smoke-test --coordinate-mode physical
   ```

5. Run a recorded experiment:

   ```bash
   python -m src.main --coordinate-mode physical \
     --output-dir results/run-1234 --seed 1234
   ```

The output directory contains `data_split.json`, `metrics.json`, three
checkpoints, `damage_parameters.npz`, and figures. Metrics named `test` are
computed only after training; use `validation` for model selection.

For normalized input files, pass `--coordinate-mode normalized`. The program
rejects out-of-range coordinates rather than silently mixing physical and
dimensionless units.
