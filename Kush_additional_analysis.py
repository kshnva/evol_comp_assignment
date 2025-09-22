# Ana_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

def analyze_bounciness_fft(positions_z_list, dt: float = 0.01, high_freq_cutoff: float = 1.0, labels=None):
    """
    Analyze vertical bounciness of geko using FFT.
    """

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(positions_z_list))]

    bouncy_metrics = []
    fig, ax = plt.subplots(figsize=(8,5))

    for i, positions_z in enumerate(positions_z_list):
        vel_z = np.diff(positions_z) / dt
        N = len(vel_z)
        fft_mag = np.abs(rfft(vel_z))
        freqs = rfftfreq(N, dt)
        b_metric = np.sum(fft_mag[freqs > high_freq_cutoff])
        b_metric_norm = b_metric / N
        bouncy_metrics.append(b_metric_norm)
        ax.plot(freqs, fft_mag, alpha=0.7, label=f"{labels[i]} | Metric={b_metric_norm:.3f}")

    ax.axvline(high_freq_cutoff, color='red', linestyle='--', label=f'High freq cutoff = {high_freq_cutoff} Hz')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Bounciness FFT Spectrum")
    ax.legend()
    ax.grid(True)

    return bouncy_metrics, fig
def analyze_actuator_usage(actuator_vel_history, labels=None):
    """
    Compute total actuator usages.
    """
    actuator_usage = np.sum(np.abs(actuator_vel_history), axis=0)
    n_act = actuator_usage.size

    if labels is None:
        labels = [f"Actuator {i+1}" for i in range(n_act)]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(labels, actuator_usage, color='orange', alpha=0.7)
    ax.set_xlabel("Actuator")
    ax.set_ylabel("Total Usage (rad)")
    ax.set_title("Total Actuator Usage ")
    ax.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return actuator_usage, fig