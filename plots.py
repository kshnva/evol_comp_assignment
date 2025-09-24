# (1) average fitness across generations, to measure convergence efficiency; 
# (2) velocity of the best individual over time, as an indicator of the quality of the locomotion that emerges
# (3) actuator effort, reflecting the energy required to generate movement. 

import numpy as np
import matplotlib.pyplot as plt
import os 

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

# Load NPZ data
def load_detailed_data(filename):
    """Load detailed simulation data from NPZ file."""
    if not filename.endswith('.npz'):
        filename += '.npz'
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    
    data = np.load(filename)
    return {
        'time': data['time'],
        'positions': data['positions'],
        'velocities': data['velocities'],
        'actuator_efforts': data['actuator_efforts'],
        'final_fitness': data['final_fitness']
    }


def load_and_plot_comparison(cmaes_file, cmaes_diag_file):
    """Load and plot comparison of CMA-ES vs CMA-ES Diagonal."""
    # Load data
    cmaes_data = load_detailed_data(cmaes_file)
    cmaes_diag_data = load_detailed_data(cmaes_diag_file)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Velocity comparison
    ax1.plot(cmaes_data['time'], cmaes_data['velocities'], 
             color='blue', label=f"CMA-ES (fitness: {cmaes_data['final_fitness']:.3f})")
    ax1.plot(cmaes_diag_data['time'], cmaes_diag_data['velocities'], 
             color='green', label=f"CMA-ES Diag (fitness: {cmaes_diag_data['final_fitness']:.3f})")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Velocity Over Time")
    ax1.legend()
    ax1.grid(True)
    
    # Actuator effort comparison
    ax2.plot(cmaes_data['time'], cmaes_data['actuator_efforts'], 
             color='blue', label="CMA-ES")
    ax2.plot(cmaes_diag_data['time'], cmaes_diag_data['actuator_efforts'], 
             color='green', label="CMA-ES Diag")
    ax2.set_ylabel("Actuator Effort")
    ax2.set_title("Actuator Effort Over Time")
    ax2.legend()
    ax2.grid(True)
    
    # Position trajectory comparison
    ax3.plot(cmaes_data['time'], cmaes_data['positions'], 
             color='blue', label="CMA-ES")
    ax3.plot(cmaes_diag_data['time'], cmaes_diag_data['positions'], 
             color='green', label="CMA-ES Diag")
    ax3.set_ylabel("Y Position (m)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Position Trajectory")
    ax3.legend()
    ax3.grid(True)
    
    # Cumulative effort comparison
    ax4.plot(cmaes_data['time'], np.cumsum(cmaes_data['actuator_efforts']), 
             color='blue', label="CMA-ES")
    ax4.plot(cmaes_diag_data['time'], np.cumsum(cmaes_diag_data['actuator_efforts']), 
             color='green', label="CMA-ES Diag")
    ax4.set_ylabel("Cumulative Effort")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Cumulative Actuator Effort")
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
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


def plot_single_algorithm_data(filename, algorithm_name):
    """Plot detailed data for a single algorithm."""
    data = load_detailed_data(filename)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Velocity over time
    ax1.plot(data['time'], data['velocities'], color='blue')
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title(f"{algorithm_name} - Velocity Over Time")
    ax1.grid(True)
    
    # Actuator effort over time
    ax2.plot(data['time'], data['actuator_efforts'], color='red')
    ax2.set_ylabel("Actuator Effort")
    ax2.set_title(f"{algorithm_name} - Actuator Effort Over Time")
    ax2.grid(True)
    
    # Position trajectory
    ax3.plot(data['time'], data['positions'], color='green')
    ax3.set_ylabel("Y Position (m)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title(f"{algorithm_name} - Position Trajectory")
    ax3.grid(True)
    
    # Velocity histogram
    ax4.hist(data['velocities'], bins=50, alpha=0.7, color='purple')
    ax4.set_xlabel("Velocity (m/s)")
    ax4.set_ylabel("Frequency")
    ax4.set_title(f"{algorithm_name} - Velocity Distribution")
    ax4.grid(True)
    
    plt.suptitle(f"{algorithm_name} - Final Fitness: {data['final_fitness']:.3f}")
    plt.tight_layout()
    return fig


# Convenience function to load and plot all available data
def plot_all_available_data(results_dir="results"):
    """Load and plot all available NPZ files in the results directory."""
    npz_files = [f for f in os.listdir(results_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"No NPZ files found in {results_dir}")
        return
    
    print(f"Found NPZ files: {npz_files}")
    
    # If we have both CMA-ES files, plot comparison
    cmaes_file = os.path.join(results_dir, "best_cmaes_data.npz")
    cmaes_diag_file = os.path.join(results_dir, "best_cmaes_diag_data.npz")
    
    if os.path.exists(cmaes_file) and os.path.exists(cmaes_diag_file):
        fig_comparison = load_and_plot_comparison(cmaes_file, cmaes_diag_file)
        fig_comparison.savefig(os.path.join(results_dir, "velocity_effort_comparison.png"))
        plt.show()
    
    # Plot individual algorithm data
    for npz_file in npz_files:
        filepath = os.path.join(results_dir, npz_file)
        algorithm_name = npz_file.replace('.npz', '').replace('best_', '').replace('_data', '').upper()
        
        fig = plot_single_algorithm_data(filepath, algorithm_name)
        output_name = npz_file.replace('.npz', '_detailed_plot.png')
        fig.savefig(os.path.join(results_dir, output_name))
        plt.show()


if __name__ == "__main__":
    # Example usage
    plot_all_available_data()