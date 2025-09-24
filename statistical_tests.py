# This module runs statistical tests on presaved data
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu


def get_final_scores(histories):
    return histories[:, -1]


def run_stat_tests(full_cmaes_vals, sep_cmaes_vals, paired=False):
    """Wilcoxon (labeled paired) or Mann–Whitney U (not paired) tests"""
    if paired:
        print("Wilcoxon test")
        stat, p = wilcoxon(sep_cmaes_vals, full_cmaes_vals, alternative="greater")
    else:
        print("Mann–Whitney U test")
        stat, p = mannwhitneyu(sep_cmaes_vals, full_cmaes_vals, alternative="greater")
    
    print(f"Test statistic: {stat:.4f}, p-value: {p:.6f}")
    if p < 0.05:
        print("Result: Significant (reject null hypothesis, sep-CMA-ES > CMA-ES)")
    else:
        print("Result: Not significant (cannot reject null hypothesis)")


if __name__ == "__main__":
    # get the results
    cmaes_hist = np.load("results/cmaes_histories.npy")
    sep_cmaes_hist = sep_cmaes = np.load("results/cmaes_diag_histories.npy")

    # Get relevant results
    full_scores = cmaes_hist[:, -1]
    sep_scores = sep_cmaes_hist[:, -1]
    
    # set True if paired
    run_stat_tests(full_scores, sep_scores, paired=False)
