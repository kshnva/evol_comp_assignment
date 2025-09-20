# main_test.py
# This module runs as main for Mika's CMAES evotorch functions

import numpy as np
from Mika_evotorch_time import run_evolution
import Fred_analysis

if __name__ == "__main__":
    runs = 5
    generations = 30
    steps = 2000
    show_plots = False  # Toggle to show or only save plots

    tanh_histories = []
    sigmoid_histories = []

    # Run tanh experiments
    for i in range(runs):
        print(f"[tanh] Run {i+1}/{runs}")
        _, fitness_history = run_evolution(
            generations=generations,
            popsize=10,
            steps=steps,
            activation="tanh",
        )
        tanh_histories.append(fitness_history)

    # Run sigmoid experiments
    for i in range(runs):
        print(f"[sigmoid] Run {i+1}/{runs}")
        _, fitness_history = run_evolution(
            generations=generations,
            popsize=10,
            steps=steps,
            activation="sigmoid",
        )
        sigmoid_histories.append(fitness_history)

    # Convert to arrays
    tanh_histories = np.array(tanh_histories)
    sigmoid_histories = np.array(sigmoid_histories)

    # Run analysis in sigmoid and tanh
    fig1 = Fred_analysis.plot_individual_runs(tanh_histories, "Tanh")
    fig2 = Fred_analysis.plot_individual_runs(sigmoid_histories, "Sigmoid")
    fig3 = Fred_analysis.plot_average_sig_tanh(tanh_histories, sigmoid_histories)

    # Save plots
    fig1.savefig("results/tanh_individual_runs.png")
    fig2.savefig("results/sigmoid_individual_runs.png")
    fig3.savefig("results/tanh_vs_sigmoid_average.png")


    # Optionally show plots
    if show_plots:
        import matplotlib.pyplot as plt
        plt.show()
