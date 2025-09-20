# This module contains code to analyse gecko movement and fitness.


import numpy as np
import matplotlib.pyplot as plt


def plot_individual_runs(histories: np.ndarray, activation: str):
    """Plots fitness for individual runs of the same type as cumulative 
    distance."""
    plt.figure(figsize=(8, 5))
    for i, history in enumerate(histories):
        plt.plot(history, alpha=0.6, label=f"{activation} Run {i+1}")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (final y)")
    plt.title(f"{activation} Individual Runs")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_average(histories_tanh: np.ndarray, histories_sigmoid: np.ndarray):
    """This plots the average and deviations for a tanh vs sigmoid run in 
    cumulative distance."""
    def _plot_average(histories, label, color):
        mean_curve = histories.mean(axis=0)
        min_curve = histories.min(axis=0)
        max_curve = histories.max(axis=0)
        plt.plot(mean_curve, label=f"{label} mean", color=color)
        plt.fill_between(range(len(mean_curve)), min_curve, max_curve,
                         color=color, alpha=0.2)

    plt.figure(figsize=(8, 5))
    _plot_average(histories_tanh, "Tanh", "red")
    _plot_average(histories_sigmoid, "Sigmoid", "blue")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (final y)")
    plt.title("Average Fitness Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    tanh_histories = np.load("tanh_histories.npy")
    sigmoid_histories = np.load("sigmoid_histories.npy")

    plot_individual_runs(tanh_histories, "Tanh")
    plot_individual_runs(sigmoid_histories, "Sigmoid")
    plot_average(tanh_histories, sigmoid_histories)
