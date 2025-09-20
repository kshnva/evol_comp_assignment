# main_experiment.py
import numpy as np
from Mika_evotorch_time import run_evolution
import Fred_analysis  # this imports your analysis functions

if __name__ == "__main__":
    runs = 5
    generations = 30
    steps = 2000

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

    # Save results
    np.save("tanh_histories.npy", tanh_histories)
    np.save("sigmoid_histories.npy", sigmoid_histories)
    print("Saved tanh_histories.npy and sigmoid_histories.npy")

    # Run analysis immediately
    #Fred_analysis.plot_individual_runs(tanh_histories, "Tanh")
    #Fred_analysis.plot_individual_runs(sigmoid_histories, "Sigmoid")
    Fred_analysis.plot_average(tanh_histories, sigmoid_histories)
