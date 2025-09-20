# This module runs as main for Mika's CMAES evotorch functions

import numpy as np
import time
from Mika_evotorch_CMAES import run_evolution, run_simulation
import Fred_analysis


if __name__ == "__main__":
    runs = 5
    generations = 50
    steps = 20000
    vel_step = 20 # Every ... steps to record velocity (too small will not show)
    show_plots = True  # Toggle to show plots

    tanh_histories = []
    sigmoid_histories = []

    # Track total experiment time
    total_start = time.time()

    # Run tanh experiments
    tanh_durations = []
    tanh_start = time.time()
    best_tanh_genome = None 
    best_tanh_traj = None
    best_tanh_fitness = None

    for i in range(runs):
        run_start = time.time()
        print(f"[tanh] Run {i+1}/{runs}")
        genome, fitness_history = run_evolution(
            generations=generations,
            popsize=10,
            steps=steps,
            activation="tanh",
        )
        run_time = time.time() - run_start
        tanh_durations.append(run_time)
        tanh_histories.append(fitness_history)
        final_fitness = fitness_history[-1]

        # Check if better fitness for velocity run
        if best_tanh_fitness is None or final_fitness > best_tanh_fitness:
            best_tanh_fitness = final_fitness
            best_tanh_genome = genome

    tanh_total = time.time() - tanh_start

    # Select the best tanh run for velocity
    _, best_tanh_traj = run_simulation(
    genome=best_tanh_genome,
    steps=steps,
    activation="tanh")
    np.save("results/best_tanh_traj.npy", best_tanh_traj)

    # Run sigmoid experiments
    sigmoid_durations = []
    sigmoid_start = time.time()
    best_sigmoid_genome = None 
    best_sigmoid_traj = None
    best_sigmoid_fitness = None

    for i in range(runs):
        run_start = time.time()
        print(f"[sigmoid] Run {i+1}/{runs}")
        genome, fitness_history = run_evolution(
            generations=generations,
            popsize=10,
            steps=steps,
            activation="sigmoid",
        )
        run_time = time.time() - run_start
        sigmoid_durations.append(run_time)
        sigmoid_histories.append(fitness_history)
        final_fitness = fitness_history[-1]

        # Check if better fitness for velocity run
        if best_sigmoid_fitness is None or final_fitness > best_sigmoid_fitness:
            best_sigmoid_fitness = final_fitness
            best_sigmoid_genome = genome

    sigmoid_total = time.time() - sigmoid_start

    # Select the best sigmoid run for velocity
    _, best_sigmoid_traj = run_simulation(
    genome=best_sigmoid_genome,
    steps=steps,
    activation="sigmoid")
    
    np.save("results/best_sigmoid_traj.npy", best_sigmoid_traj)

    # Convert to arrays
    tanh_histories = np.array(tanh_histories)
    sigmoid_histories = np.array(sigmoid_histories)

    print("Tanh trajectory length:", len(best_tanh_traj))
    print("Sigmoid trajectory length:", len(best_sigmoid_traj))

    # Run analysis on sigmoid and tanh
    fig1 = Fred_analysis.plot_individual_runs(tanh_histories, "Tanh")
    fig2 = Fred_analysis.plot_individual_runs(sigmoid_histories, "Sigmoid")
    fig3 = Fred_analysis.plot_average_sig_tanh(tanh_histories, sigmoid_histories)
    fig4 = Fred_analysis.plot_velocity(best_tanh_traj, best_sigmoid_traj, vel_step)

    # Save plots
    fig1.savefig(f"results/tanh_individual_runs_gens{generations}_ste{steps}.png")
    fig2.savefig(f"results/sigmoid_individual_runs_gens{generations}_ste{steps}.png")
    fig3.savefig(f"results/tanh_vs_sigmoid_average_gens{generations}_ste{steps}.png")
    fig4.savefig(f"results/velocity_comparison_gens{generations}_steps{steps}.png")

    # Optional: Show plots
    if show_plots:
        import matplotlib.pyplot as plt
        plt.show()

    # Display the total time
    print(f"All {runs} tanh runs took {tanh_total:.2f} seconds "
          f"(avg run: {np.mean(tanh_durations):.2f} sec/run)")
    
    print(f"All {runs} sigmoid runs took {sigmoid_total:.2f} seconds "
          f"(avg run: {np.mean(sigmoid_durations):.2f} sec/run)")

    total_time = time.time() - total_start
    print(f"All experiments completed in {total_time:.2f} seconds")
