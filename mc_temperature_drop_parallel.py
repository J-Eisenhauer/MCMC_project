"""
Parallel temperature-drop MC with 10 independent chains.

Runs 10 independent chains, all following the same temperature schedule
(only the random seed differs between chains).
Usage:
  python mc_temperature_drop_parallel.py

"""

from pathlib import Path
import time
import sys
from io import StringIO
from multiprocessing import Pool, Manager
from functools import partial
import numpy as np
import mc_fixed_temperature as mc


DENSITY = 0.291
MAX_DISPLACEMENT = 0.5
EQUILIBRATION_STEPS = 5_000_000
ENERGY_EVERY = 1_000
BASE_SEED = 2026
OUTPUT_DIR = Path("temperature_drop_results_parallel")

# Configuration snapshots (x, y) are saved at these temperatures
SNAPSHOT_TEMPERATURES = (0.25, 0.20, 0.19, 0.18, 0.15)

# Production steps per temperature window (T_high, T_low, steps)
PRODUCTION_STEPS_WINDOWS = [
    (0.250, 0.225, 50_000_000),
    (0.225, 0.200, 150_000_000),
    (0.200, 0.170, 300_000_000),
    (0.170, 0.150, 150_000_000)]


def base_temperature_schedule():
    """
    Returns numpy array of temperatures in descending order.

    Every chain uses this same schedule, only the seed differs between chains.
    """
    temps = []
    temps.extend(np.arange(250, 205, -10) / 1000)
    temps.extend(np.arange(205, 175, -2) / 1000)
    temps.extend(np.arange(175, 149, -10) / 1000)
    temps.append(0.150)
    return np.array(sorted(temps, reverse=True))


def production_steps_schedule(schedule):
    """Returns array of production steps for each temperature in the schedule."""
    steps = []
    for temperature in schedule:
        steps.append(next(n for t_high, t_low, n in PRODUCTION_STEPS_WINDOWS
                          if t_low <= temperature <= t_high))
    return np.array(steps)


def snapshot_indices(schedule, targets):
    """
    Returns the sorted schedule indices closest to each target temperature.
    """
    return sorted({int(np.argmin(np.abs(schedule - target))) for target in targets})


# CORE: Single-chain runner

def run_single_chain(chain_id, seed, temperatures, production_steps, box_length,
                     snapshot_steps, total_runs, progress_counter, progress_lock):
    """Runs a single chain of the temperature-drop simulation with a given seed."""

    rng = np.random.default_rng(seed)
    x, y = mc.random_nonoverlapping_positions(mc.N, box_length, rng)

    results = []
    maximum_displacement = MAX_DISPLACEMENT

    for index, temperature in enumerate(temperatures):
        n_production = int(production_steps[index])
        old_stdout = sys.stdout # save current stdout
        sys.stdout = StringIO() # suppress output for mc run (loads of print statements)

        # find max displacement for this temperature
        maximum_displacement = mc.tune_displacement(x, y, box_length, temperature, maximum_displacement, seed + 3 * index)

        # equilibration stage (evolves x, y in place)
        mc.run_stage(x, y, box_length, temperature, maximum_displacement,
            EQUILIBRATION_STEPS, ENERGY_EVERY, ENERGY_EVERY, False, seed + 3 * index + 1)

        # production stage
        production = mc.run_stage(
            x, y, box_length, temperature, maximum_displacement,
            n_production, ENERGY_EVERY, ENERGY_EVERY, False, seed + 3 * index + 2)

        sys.stdout = old_stdout # restore stdout

        # data unpacking
        production_energies = production[1].astype(float)
        accepted = production[4]
        unfavorable_accepted = production[5]
        tracked_energy = production[8]

        # Variance and heat capacity calculations
        mean_energy = np.mean(production_energies)
        mean_energy_squared = np.mean(production_energies**2)
        energy_fluctuation = mean_energy_squared - mean_energy**2
        heat_capacity_over_k = energy_fluctuation / (temperature**2)
        heat_capacity_per_particle = heat_capacity_over_k / mc.N
        acceptance_rate = accepted / n_production

        # total energy sanity check
        recalculated_energy = mc.total_energy(x, y, box_length)
        energy_discrepancy = abs(tracked_energy - recalculated_energy)

        # Only needed temperatures get a saved snapshot.
        if index in snapshot_steps:
            snapshot_filename = OUTPUT_DIR / f"chain_{chain_id}" / f"prod_T_{temperature:.3f}.npz"
            snapshot_filename.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(snapshot_filename, x=x, y=y)

        # store results for this temperature
        results.append([
            temperature, mean_energy, mean_energy / mc.N, energy_fluctuation,
            heat_capacity_over_k, heat_capacity_per_particle,
            acceptance_rate, unfavorable_accepted, energy_discrepancy,
            maximum_displacement, n_production, accepted,
            tracked_energy, recalculated_energy])

        # Update progress counter and print progress
        with progress_lock:
            progress_counter.value += 1
            print(f"{progress_counter.value}/{total_runs} MC runs done", flush=True)

    return np.asarray(results)



####################################################
# EXPORT + MAIN()
# everything is written to disk
####################################################

RESULT_COLUMNS = [
    "temperature", "mean_energy", "mean_energy_per_particle", "energy_fluctuation",
    "heat_capacity_over_k", "heat_capacity_per_particle", "acceptance_rate",
    "unfavorable_accepted", "energy_discrepancy", "max_displacement",
    "production_steps", "accepted_moves", "tracked_energy", "recalculated_energy",
]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 10 seeds
    num_seeds = 10
    schedule = base_temperature_schedule()
    production_steps = production_steps_schedule(schedule)
    snapshot_steps = snapshot_indices(schedule, SNAPSHOT_TEMPERATURES)
    box_length = np.sqrt(mc.N / DENSITY)
    seeds = [BASE_SEED + seed_idx * 1000 for seed_idx in range(num_seeds)]
    total_runs = num_seeds * len(schedule)

    manager = Manager() # for shared progress counter otherwise multiprocessing.Pool has no shared state
    # partial function to fix arguments for run_single_chain
    run = partial(run_single_chain, total_runs=total_runs,
                  progress_counter=manager.Value('i', 0), progress_lock=manager.Lock())

    # Create a list of tasks that will be executed in parallel. Each task is a tuple of arguments for run_single_chain.
    tasks = [(f"seed_{i}", seeds[i], schedule, production_steps, box_length, snapshot_steps)
             for i in range(num_seeds)]

    # Run the tasks in parallel using a pool of processes. The results from all chains are collected in all_results.
    start = time.time()
    with Pool(num_seeds) as pool:
        all_results = pool.starmap(run, tasks)
    total_time = time.time() - start

    # one table for all chains
    table = np.vstack([np.column_stack((np.full(len(r), i), r)) for i, r in enumerate(all_results)])
    np.savetxt(OUTPUT_DIR / "all_chains_results.csv", table, delimiter=",", fmt="%.17g",
               header="chain," + ",".join(RESULT_COLUMNS), comments="")

    print(f"Runtime: {total_time/3600:.2f}h")

if __name__ == "__main__":
    main()
