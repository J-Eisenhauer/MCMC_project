# Stripe phases: Metropolis Monte Carlo

Monte Carlo simulation of a 2D hard-core/soft-corona system (N = 250, σ1/σ0 = 2.5), following
Malescio & Pellicane, Nature Materials 2, 97–100 (2003).

- `mc_fixed_temperature.py`: core routines and density scan at T* = 0.1
- `mc_temperature_drop_parallel.py`: temperature drop at ρ* = 0.291, ten parallel chains
- `plot_results.py`: all figures of the report
- `figures and data/`: simulation output (CSV, snapshots) and figures

Requires `numpy`, `numba` and `matplotlib`. Run the scripts in the order above.
To plot from the uploaded data, place its two folders next to the scripts and rename
`var_temperature_results` to `temperature_drop_results_parallel`.
