# Fred_analysis.py
# This module contains code to analyse gecko movement and fitness.

import numpy as np
import matplotlib.pyplot as plt


def plot_individual_runs(histories: np.ndarray, activation: str):
    """Return a Figure for individual runs of one activation type."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, history in enumerate(histories):
        ax.plot(history, alpha=0.6, label=f"{activation} Run {i+1}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (final y)")
    ax.set_title(f"{activation} Individual Runs")
    ax.legend()
    ax.grid(True)
    return fig


def plot_average_sig_tanh(histories_tanh: np.ndarray, histories_sigmoid: np.ndarray):
    """Return a Figure for average + deviation comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    def _plot_average(histories, label, color):
        mean_curve = histories.mean(axis=0)
        min_curve = histories.min(axis=0)
        max_curve = histories.max(axis=0)
        ax.plot(mean_curve, label=f"{label} mean", color=color)
        ax.fill_between(range(len(mean_curve)), min_curve, max_curve,
                        color=color, alpha=0.2)

    _plot_average(histories_tanh, "Tanh", "red")
    _plot_average(histories_sigmoid, "Sigmoid", "blue")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (final y)")
    ax.set_title("Average Fitness Comparison")
    ax.legend()
    ax.grid(True)
    return fig
