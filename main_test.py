# This module runs as main for Mika's CMAES evotorch functions

import numpy as np
import time
from Mika_evotorch_CMAES import run_evolution
import Fred_analysis


if __name__ == "__main__":
    runs = 5
    generations = 50
    steps = 2000
    show_plots = True  # Toggle to show plots

    tanh_histories = []
    sigmoid_histories = []

    # Track total experiment time
    total_start = time.time()

    # Run tanh experiments
    tanh_durations = []
    tanh_start = time.time()

    for i in range(runs):
        run_start = time.time()
        print(f"[tanh] Run {i+1}/{runs}")
        _, fitness_history = run_evolution(
            generations=generations,
            popsize=10,
            steps=steps,
            activation="tanh",
        )
        run_time = time.time() - run_start
        tanh_durations.append(run_time)
        tanh_histories.append(fitness_history)

    tanh_total = time.time() - tanh_start

    # Run sigmoid experiments
    sigmoid_durations = []
    sigmoid_start = time.time()

    for i in range(runs):
        run_start = time.time()
        print(f"[sigmoid] Run {i+1}/{runs}")
        _, fitness_history = run_evolution(
            generations=generations,
            popsize=10,
            steps=steps,
            activation="sigmoid",
        )
        run_time = time.time() - run_start
        sigmoid_durations.append(run_time)
        sigmoid_histories.append(fitness_history)

    sigmoid_total = time.time() - sigmoid_start

    # Convert to arrays
    tanh_histories = np.array(tanh_histories)
    sigmoid_histories = np.array(sigmoid_histories)

    # Run analysis on sigmoid and tanh
    fig1 = Fred_analysis.plot_individual_runs(tanh_histories, "Tanh")
    fig2 = Fred_analysis.plot_individual_runs(sigmoid_histories, "Sigmoid")
    fig3 = Fred_analysis.plot_average_sig_tanh(tanh_histories, sigmoid_histories)

    # Save plots
    fig1.savefig(f"results/tanh_individual_runs_gens{generations}_ste{steps}.png")
    fig2.savefig(f"results/sigmoid_individual_runs_gens{generations}_ste{steps}.png")
    fig3.savefig(f"results/tanh_vs_sigmoid_average_gens{generations}_ste{steps}.png")

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
