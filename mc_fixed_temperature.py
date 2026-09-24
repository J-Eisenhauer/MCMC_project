"""Fixed-temperature part of the stripe-phase Monte Carlo project.

Runs N particles at T* = 0.1 for the five densities and saves:
  - the equilibration energy trace,
  - the final configuration,
  - the production-averaged radial distribution function g(r),
  - a small text summary for numerical checks and acceptance ratios.

All plotting for this project lives in plot_results.py, so this script only
simulates and writes data to disk.
"""


from pathlib import Path
import time  
import numpy as np
from numba import njit


# -----------------------------  settings -----------------------------
N = 250                             # number of particles
DENSITIES = (0.100, 0.150, 0.227, 0.291, 0.380)  # rho* values
TEMPERATURE = 0.1                 # reduced temperature kT/epsilon
SIGMA_0 = 1.0                     # hard-core distance
SIGMA_1 = 2.5                     # shoulder distance
MAX_DISPLACEMENT = 0.5            # trial displacement in each direction
TUNING_BURN_IN_STEPS = 100_000    # discarded before acceptance is assessed
TUNING_STEPS = 10_000             # extra pre-equilibration moves per test
MIN_TUNING_ROUNDS = 3             # min number of tuning blocks for max_displacement 
MAX_TUNING_ROUNDS = 8              # max number of tuning blocks for max_displacement
MIN_ACCEPTANCE = 0.25              # desired lower acceptance-rate limit.
MAX_ACCEPTANCE = 0.35              # desired upper acceptance-rate limit.

EQUILIBRATION_STEPS = 5_000_000     # number of moves before production (for fixed T* runs)
PRODUCTION_STEPS = 100_000_000      # number of moves for production (for fixed T* runs)

ENERGY_EVERY = 1_000                # Energy is recorded every ENERGY_EVERY attempted moves
RDF_EVERY = 1_000                   # RDF = Radial Distribution Function

RDF_BINS = 1_000                  # Fine resolution between r=0 and L/2.
OUTPUT_DIR = Path("fixed_temperature_results")
BASE_SEED = 2026                 



# -------------------------- Monte Carlo functions ------------------------
@njit
def minimum_image(delta, box_length):
    """
    Return the shortest signed separation in a periodic square box.

    np.rint rounds to the nearest integer.  Subtracting this integer number of
    box lengths maps a displacement into approximately [-L/2, L/2].
    """
    return delta - box_length * np.rint(delta / box_length)


@njit
def particle_energy(x, y, particle, box_length):
    """
    Calculate how one particle interacts with every other particle.

    In reduced units, each soft overlap contributes epsilon = 1.
    Returning -1 is a fast signal that a forbidden hard-core overlap exists.

    """
    energy = 0
    sigma0_sq = SIGMA_0**2
    sigma1_sq = SIGMA_1**2

    for other in range(x.size): # Loop over all particles.
        if other == particle:
            continue  # A particle does not interact with itself.

        # Measure the distance using the closest periodic copy.
        dx = minimum_image(x[particle] - x[other], box_length)
        dy = minimum_image(y[particle] - y[other], box_length)
        distance_sq = dx * dx + dy * dy

        # Squared distances avoid an unnecessary square root in every test.
        if distance_sq < sigma0_sq:
            return -1
        if distance_sq < sigma1_sq:
            energy += 1               # epsilon = 1 in reduced units
    return energy


@njit
def total_energy(x, y, box_length):
    """
    Calculate the complete system energy for initialization and checks.

    Only pairs with j > i are visited, so every interaction is counted once.
    """

    energy = 0
    sigma0_sq = SIGMA_0**2
    sigma1_sq = SIGMA_1**2

    for i in range(x.size - 1):
        for j in range(i + 1, x.size):
            dx = minimum_image(x[i] - x[j], box_length)
            dy = minimum_image(y[i] - y[j], box_length)
            distance_sq = dx * dx + dy * dy

            if distance_sq < sigma0_sq:
                return -1

            if distance_sq < sigma1_sq:
                energy += 1

    return energy


def random_nonoverlapping_positions(n_particles, box_length, rng):
    """
    Construct a random starting configuration without hard-core overlaps.

    Particles are inserted one after another.  A candidate is discarded if it
    overlaps any particle that has already been placed.
    """

    # np.empty allocates the arrays without filling them with zeros.
    x = np.empty(n_particles)
    y = np.empty(n_particles)

    for i in range(n_particles):

        while i <= n_particles:
            # Draw two independent uniform coordinates.
            candidate_x, candidate_y = rng.uniform(0.0, box_length, 2)

            if i == 0:
                valid = True

            else:
                # x[:i] is a slice containing only particles placed so far.
                # Naive distance
                dx = candidate_x - x[:i]
                dy = candidate_y - y[:i]

                # Apply the minimum-image convention (periodic boundary conditions) to all distances at once.
                dx -= box_length * np.rint(dx / box_length)
                dy -= box_length * np.rint(dy / box_length)

                # np.all is True only if every pair satisfies the condition.
                valid = np.all(dx * dx + dy * dy >= SIGMA_0**2)

            if valid:
                x[i], y[i] = candidate_x, candidate_y
                break  # Stop searching after a valid position was found.

    return x, y


@njit
def accumulate_pair_histogram(x, y, box_length, histogram):
    """
    Applied for a chosen configuration: unordered pair distances are added to the RDF histogram.

    Distances larger than L/2 are excluded because circular shells are no
    longer represented without geometric complications in a square box.

    """

    maximum_r = box_length / 2.0
    bin_width = maximum_r / histogram.size

    for i in range(x.size - 1):
        for j in range(i + 1, x.size):

            dx = minimum_image(x[i] - x[j], box_length)
            dy = minimum_image(y[i] - y[j], box_length)
            r = np.sqrt(dx * dx + dy * dy)

            if r < maximum_r:
                # int truncates a positive number, selecting its histogram bin.
                bin_index = int(r / bin_width)
                histogram[bin_index] += 1


@njit
def run_stage(x, y, box_length, temperature, maximum_displacement,
              n_steps, energy_every, rdf_every, collect_rdf, seed):
    """
    Run one equilibration, tuning, or production stage.

    The x and y arrays are modified in place, so the returned final state is
    already stored in the input arrays.  RDF sampling is optional because it
    is needed only during production.
    """

    # Configs and allocations
    np.random.seed(seed)
    n_energy_samples = n_steps // energy_every

    energy_steps = np.empty(n_energy_samples, dtype=np.int64)
    energies = np.empty(n_energy_samples, dtype=np.int64)
    histogram = np.zeros(RDF_BINS, dtype=np.int64)

    accepted = 0
    unfavorable_accepted = 0
    hard_rejected = 0
    metropolis_rejected = 0
    energy_sample = 0
    rdf_frames = 0

    # Exact starting energy
    energy = total_energy(x, y, box_length)

    for step in range(1, n_steps + 1):

        # randint chooses one particle index uniformly from 0 to N-1.
        particle = np.random.randint(x.size)
        old_x, old_y = x[particle], y[particle]
        old_particle_energy = particle_energy(x, y, particle, box_length)

        # Propose displacement. The modulo wraps coordinates back into the box.
        x[particle] = (old_x + maximum_displacement*np.random.uniform(-1, 1)) % box_length
        y[particle] = (old_y + maximum_displacement*np.random.uniform(-1, 1)) % box_length

        new_particle_energy = particle_energy(x, y, particle, box_length)

        if new_particle_energy < 0:
            # Hard-core overlaps are forbidden and therefore always rejected.
            x[particle], y[particle] = old_x, old_y
            hard_rejected += 1

        else:
            delta_energy = new_particle_energy - old_particle_energy

            # Favorable/neutral moves are accepted.  Unfavorable moves are
            # accepted with the Metropolis probability exp(-Delta U / T*).
            if delta_energy <= 0 or np.random.random() < np.exp(-delta_energy / temperature):
                energy += delta_energy
                accepted += 1
                if delta_energy > 0:
                    unfavorable_accepted += 1

            else:
                # rejected proposal
                x[particle], y[particle] = old_x, old_y
                metropolis_rejected += 1

        # Energy at regular intervals
        if step % energy_every == 0:
            energy_steps[energy_sample] = step
            energies[energy_sample] = energy
            energy_sample += 1

        # Add the current complete configuration to g(r) at regular intervals
        if collect_rdf and step % rdf_every == 0:
            accumulate_pair_histogram(x, y, box_length, histogram)
            rdf_frames += 1

    return (energy_steps, energies, histogram, rdf_frames, accepted,
            unfavorable_accepted, hard_rejected, metropolis_rejected, energy)


def tune_displacement(x, y, box_length, temperature, initial_displacement, seed, tuning_steps=10_000):
    """
    Tune the proposal size until the acceptance rate lies in the target range.

    Its initial burn-in and all tuning blocks are discarded from measurements.
    The final displacement remains fixed throughout production.
    """

    displacement = initial_displacement
    print(f"  tuning burn-in: {TUNING_BURN_IN_STEPS} discarded moves")

    # First the system evolves without evaluating.
    run_stage(
        x, y, box_length, temperature, displacement,
        TUNING_BURN_IN_STEPS, TUNING_BURN_IN_STEPS,
        TUNING_BURN_IN_STEPS, False, seed)

    for tuning_round in range(MAX_TUNING_ROUNDS):

        test = run_stage(
            x, y, box_length, temperature, displacement,
            tuning_steps, tuning_steps, tuning_steps, False,
            seed + tuning_round + 1)

        # test[4] is the number of accepted moves returned by run_stage
        acceptance_rate = test[4] / tuning_steps
        print(f"  tuning {tuning_round + 1}: displacement={displacement:.4f}, acceptance={acceptance_rate:.2%}")

        in_target_range = MIN_ACCEPTANCE <= acceptance_rate <= MAX_ACCEPTANCE

        # several blocks so that one noisy block cannot end tuning early
        if in_target_range and tuning_round + 1 >= MIN_TUNING_ROUNDS:
            return displacement

        if tuning_round == MAX_TUNING_ROUNDS - 1:
            print("  warning: target acceptance range was not reached")
            return displacement

        # Smaller displacements increase acceptance, larger ones lower it.
        if acceptance_rate < MIN_ACCEPTANCE:
            displacement *= 0.8
        elif acceptance_rate > MAX_ACCEPTANCE:
            displacement *= 1.2

    return displacement


def normalize_rdf(histogram, frames, n_particles, box_length):
    """
    Convert all accumulated pair counts into the radial distribution g(r).

    Division by
    area,  (-> local number density in the shell)
    number density, (-> compares local density to average density)
    n_particles, (-> average over all particles)
    and frame count (-> average over all frames)
    makes g(r) dimensionless.
    The factor 2 converts unordered pairs (counted once) into
    neighbour relations (each pair contributes to both particles (count for ij contributes to particle i AND j)).
    """

    edges = np.linspace(0.0, box_length / 2.0, histogram.size + 1)

    # Each r value is placed at the center of its bin
    centers = 0.5 * (edges[:-1] + edges[1:])
    areas = np.pi * (edges[1:]**2 - edges[:-1]**2)
    number_density = n_particles / box_length**2
    g_r = 2.0 * histogram / (frames * n_particles * number_density * areas)
    return centers, g_r



############################################################################

# ----------------------------- Main Run  ----------------------------------

############################################################################

def simulate_density(density, case_number):
    """Run the complete workflow for one reduced density."""
    # From rho* = N sigma_0^2/L^2 and sigma_0 = 1, L = sqrt(N/rho*).
    box_length = np.sqrt(N / density)
    output_directory = OUTPUT_DIR / f"rho_{density:.3f}"
    # parents=True creates missing parent folders; exist_ok avoids an error on reruns.
    output_directory.mkdir(parents=True, exist_ok=True)

    # default_rng is the random-number generator.
    rng = np.random.default_rng(BASE_SEED + case_number)
    x, y = random_nonoverlapping_positions(N, box_length, rng)

    print(f"\nDensity {density:.3f}, L={box_length:.3f}")
    start_time = time.time()
    maximum_displacement = MAX_DISPLACEMENT

    # Equilibration is recorded for the diagnostic energy plot, but not for g(r).
    equilibration = run_stage(
        x, y, box_length, TEMPERATURE, maximum_displacement,
        EQUILIBRATION_STEPS, ENERGY_EVERY, RDF_EVERY, False,
        BASE_SEED + 1 + case_number)

    eq_steps, eq_energies = equilibration[0], equilibration[1]

    print("Displacement tuning:")
    maximum_displacement = tune_displacement(
        x, y, box_length, TEMPERATURE, maximum_displacement,
        BASE_SEED + case_number, TUNING_STEPS)

    # Production uses the fixed tuned displacement and accumulates the RDF.
    production = run_stage(
        x, y, box_length, TEMPERATURE, maximum_displacement,
        PRODUCTION_STEPS, ENERGY_EVERY, RDF_EVERY, True,
        BASE_SEED + 2 + case_number)

    (_, prod_energies, histogram, frames, accepted, unfavorable_accepted,
     hard_rejected, metropolis_rejected, tracked_energy) = production

    # recalculate U to detect errors in energy updates
    recalculated_energy = total_energy(x, y, box_length)
    energy_difference = abs(tracked_energy - recalculated_energy)

    acceptance_ratio = accepted / PRODUCTION_STEPS
    unfavorable_fraction = unfavorable_accepted / PRODUCTION_STEPS
    unfavorable_of_accepted = unfavorable_accepted / accepted

    r, g_r = normalize_rdf(histogram, frames, N, box_length)

    ###################################################################

    # Save all results to disk

    ####################################################################
    
    np.savetxt(output_directory / "equilibration_energy.csv",
               np.column_stack((eq_steps, eq_energies / N)), delimiter=",",
               header="attempted_moves,energy_per_particle", comments="")

    np.savetxt(output_directory / "rdf.csv", np.column_stack((r, g_r)),
               delimiter=",", header="r,g_r", comments="")

    np.savetxt(output_directory / "final_coordinates.csv",
               np.column_stack((x, y)), delimiter=",", header="x,y", comments="")

    summary = (
        f"N = {N}\nbox length = {box_length:.10g}\ndensity = {density}\n"
        f"T* = {TEMPERATURE}\nmaximum displacement = {maximum_displacement:.8g}\n"
        f"tuning burn-in attempts = {TUNING_BURN_IN_STEPS}\n"
        f"tuning attempts per round = {TUNING_STEPS}\n"
        f"equilibration attempts = {EQUILIBRATION_STEPS}\n"
        f"production attempts = {PRODUCTION_STEPS}\nrdf frames = {frames}\n"
        f"overall acceptance = {accepted}/{PRODUCTION_STEPS} ({acceptance_ratio:.6%})\n"
        f"unfavorable moves accepted = {unfavorable_accepted} "
        f"({unfavorable_fraction:.6%} of all attempts; "
        f"{unfavorable_of_accepted:.6%} of accepted moves)\n"
        f"hard-core rejected = {hard_rejected} ({hard_rejected / PRODUCTION_STEPS:.6%})\n"
        f"Metropolis rejected = {metropolis_rejected} "
        f"({metropolis_rejected / PRODUCTION_STEPS:.6%})\n"
        f"tracked final energy = {tracked_energy}\n"
        f"recalculated final energy = {recalculated_energy}\n"
        f"absolute energy discrepancy = {energy_difference}\n"
        f"mean production U/N = {np.mean(prod_energies / N):.8g}\n"
        f"runtime seconds = {time.time() - start_time:.2f}\n")

    (output_directory / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    return acceptance_ratio, energy_difference


def main():
    """Run all densities sequentially."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows = []
    for case_number, density in enumerate(DENSITIES):
        acceptance_ratio, energy_difference = simulate_density(density, case_number)
        rows.append([density, acceptance_ratio, energy_difference])

    rows = np.array(rows)
    np.savetxt(OUTPUT_DIR / "acceptance_and_energy_summary.csv", rows, delimiter=",",
               header="density,acceptance_ratio,energy_discrepancy", comments="")


# This guard runs main only when this file is executed directly.  It does not
# run when mc_temperature_drop_parallel.py imports the functions from this module.
if __name__ == "__main__":
    main()