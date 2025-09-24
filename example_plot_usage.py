#!/usr/bin/env python3
"""
Example script showing how to load and plot the NPZ data files.
Run this after running gecko_main.py to visualize the results.
"""

from plots import (
    load_detailed_data, 
    load_and_plot_comparison, 
    plot_single_algorithm_data, 
    plot_all_available_data
)
import matplotlib.pyplot as plt
import os

def main():
    """Main function to demonstrate plotting capabilities."""
    
    # Method 1: Plot all available data automatically
    print("=== Plotting All Available Data ===")
    plot_all_available_data("results")
    
    # Method 2: Load and plot specific files
    try:
        print("\n=== Plotting Specific Algorithm Data ===")
        
        # Plot CMA-ES data
        if os.path.exists("results/best_cmaes_data.npz"):
            fig1 = plot_single_algorithm_data("results/best_cmaes_data.npz", "CMA-ES")
            plt.show()
        
        # Plot CMA-ES Diagonal data  
        if os.path.exists("results/best_cmaes_diag_data.npz"):
            fig2 = plot_single_algorithm_data("results/best_cmaes_diag_data.npz", "CMA-ES Diagonal")
            plt.show()
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure to run gecko_main.py first to generate the data files.")
    
    # Method 3: Load data manually and do custom plotting
    try:
        print("\n=== Custom Data Analysis ===")
        
        if os.path.exists("results/best_cmaes_data.npz"):
            data = load_detailed_data("results/best_cmaes_data.npz")
            
            # Print some statistics
            print(f"Final fitness: {data['final_fitness']}")
            print(f"Average velocity: {data['velocities'].mean():.4f} m/s")
            print(f"Max velocity: {data['velocities'].max():.4f} m/s")
            print(f"Total actuator effort: {data['actuator_efforts'].sum():.4f}")
            print(f"Average actuator effort: {data['actuator_efforts'].mean():.4f}")
            
    except Exception as e:
        print(f"Error in custom analysis: {e}")

if __name__ == "__main__":
    main()