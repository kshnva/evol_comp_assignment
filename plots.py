# (1) average fitness across generations, to measure convergence efficiency; 
# (2) velocity of the best individual over time, as an indicator of the quality of the locomotion that emerges
# (3) actuator effort, reflecting the energy required to generate movement. 

import numpy as np
import matplotlib.pyplot as plt 

# Plot average fitness over generations
def plot_algorithms_comparison(random_histories, cmaes_histories, cmaes_diag_histories):
    """Plot mean ± stdev curves for Random EA, CMA-ES, and Algorithm 3."""

    fig, ax = plt.subplots(figsize=(8, 5))
    gens = np.arange(random_histories.shape[1])

    def _plot(histories, label, color):
        mean_curve = histories.mean(axis=0)
        std_curve = histories.std(axis=0)
        ax.plot(gens, mean_curve, label=label, color=color)
        ax.fill_between(gens,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=color,
                        alpha=0.2)

    _plot(random_histories, "Random EA", "gray")
    _plot(cmaes_histories, "CMA-ES", "blue")
    _plot(cmaes_diag_histories, "CMA-ES (Diagonal)", "green")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (final y)")
    ax.set_title("Algorithm Comparison")
    ax.legend()
    ax.grid(True)
    return fig

# Plot velocity and effort over time for the best individual
def plot_velocity_and_effort(time, velocities, efforts):
    """Plot velocity and effort over time for the best individual."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(time, velocities, color='blue')
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Velocity Over Time")
    ax1.grid(True)

    ax2.plot(time, efforts, color='red')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Actuator Effort")
    ax2.set_title("Actuator Effort Over Time")
    ax2.grid(True)

    plt.tight_layout()
    return fig