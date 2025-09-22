# This module runs as main for Mika's CMAES evotorch functions

import numpy as np
import time
import torch
import Fred_analysis
import matplotlib.pyplot as plt 
from Mika_evotorch_CMAES import run_evolution, run_simulation, run_best_genome
import Kush_additional_analysis
# Turn off evotorch spam messages
import logging
logging.getLogger("evotorch").setLevel(logging.WARNING)


if __name__ == "__main__":
    runs = 3
    generations = 10
    steps = 200
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

        # Select best genome
        if best_tanh_fitness is None or final_fitness > best_tanh_fitness:
            best_tanh_fitness = final_fitness
            best_tanh_genome = genome

    tanh_total = time.time() - tanh_start

    # Select the best tanh run for velocity
    if best_tanh_genome is None:
        raise RuntimeError("No valid tanh genome found")
    _, best_tanh_traj,z_pos_tanh, actuator_vel_tanh= run_simulation(
        genome=best_tanh_genome,
        steps=steps,
        activation="tanh")
    np.save("results/best_tanh_traj.npy", best_tanh_traj)
    # bouncy_metric_tanh, fig_bounce_tanh = Kush_additional_analysis.analyze_bounciness_fft(
    #     positions_z=z_pos_tanh,
    #     dt=0.01,
    #     high_freq_cutoff=1.0
    # )
    # print(f"Tanh bounciness metric: {bouncy_metric_tanh:.3f}")
    # fig_bounce_tanh.savefig("results/bounciness_tanh.png")


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

        # Select best genome
        if best_sigmoid_fitness is None or final_fitness > best_sigmoid_fitness:
            best_sigmoid_fitness = final_fitness
            best_sigmoid_genome = genome

    sigmoid_total = time.time() - sigmoid_start

    # Select the best sigmoid run for velocity
    if best_sigmoid_genome is None:
        raise RuntimeError("No valid sigmoid genome found")
    _, best_sigmoid_traj,z_pos_sigmoid, actuator_vel_sigmoid = run_simulation(
        genome=best_sigmoid_genome,
        steps=steps,
        activation="sigmoid")
    np.save("results/best_sigmoid_traj.npy", best_sigmoid_traj)

    positions_list = [z_pos_tanh, z_pos_sigmoid]
    labels = ["Tanh", "Sigmoid"]

    bouncy_metrics, fig = Kush_additional_analysis.analyze_bounciness_fft(
        positions_z_list=positions_list,
        dt=0.01,
        high_freq_cutoff=1.0,
        labels=labels
    )

    for label, metric in zip(labels, bouncy_metrics):
        print(f"{label} bounciness metric: {metric:.3f}")

    fig.savefig("results/bounciness_comparison.png")
    plt.show()
    actuator_labels = [f"Actuator {i+1}" for i in range(actuator_vel_tanh.shape[1])]
    vel_usage_tanh, fig_act_tanh = Kush_additional_analysis.analyze_actuator_usage(
        actuator_vel_history=actuator_vel_tanh,
        labels=actuator_labels
    )
    vel_usage_sigmoid, fig_act_sigmoid = Kush_additional_analysis.analyze_actuator_usage(
        actuator_vel_history=actuator_vel_sigmoid,
        labels=actuator_labels
    )

    print("Tanh actuator usage:", vel_usage_tanh)
    print("Sigmoid actuator usage:", vel_usage_sigmoid)
    total_usage_tanh = np.sum(vel_usage_tanh)
    total_usage_sigmoid = np.sum(vel_usage_sigmoid)

    print("Total actuator usage (Tanh):", total_usage_tanh)
    print("Total actuator usage (Sigmoid):", total_usage_sigmoid)


    fig_act_tanh.savefig("results/actuator_usage_tanh.png")
    fig_act_sigmoid.savefig("results/actuator_usage_sigmoid.png")
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

    # -----------------------
    # Visualization prompt
    # -----------------------
    choice = input("Visualize best genome? (tanh/sigmoid/none): ").strip().lower()

    if choice == "tanh":
        print("Launching MuJoCo viewer with best tanh genome...")
        run_best_genome(best_tanh_genome, activation="tanh")
    elif choice == "sigmoid":
        print("Launching MuJoCo viewer with best sigmoid genome...")
        run_best_genome(best_sigmoid_genome, activation="sigmoid")
    else:
        print("Skipping visualization.")
