#!/usr/bin/env python3
"""
Performance comparison script between original and Numba-optimized versions.
"""

import time
import numpy as np
import torch
from gecko_main import run_simulation as run_simulation_original
from gecko_main_numba import run_simulation_optimized as run_simulation_numba
from gecko_main import GENOME_SIZE, INITIAL_WEIGHT_RANGE, SIMULATION_STEPS

def benchmark_simulation_performance():
    """Compare performance between original and Numba versions."""
    
    # Generate test genomes
    num_tests = 10
    test_genomes = []
    for _ in range(num_tests):
        genome = torch.randn(GENOME_SIZE) * INITIAL_WEIGHT_RANGE
        test_genomes.append(genome)
    
    steps = 1000  # Shorter for benchmarking
    
    print("=== Performance Benchmark ===")
    print(f"Running {num_tests} simulations with {steps} steps each")
    
    # Benchmark original version
    print("\nTesting Original Version...")
    start_time = time.time()
    original_results = []
    
    for i, genome in enumerate(test_genomes):
        result = run_simulation_original(genome, steps=steps, act_func="tanh", collect_data=False)
        original_results.append(result[0])
        print(f"  Test {i+1}/{num_tests}: Fitness = {result[0]:.4f}")
    
    original_time = time.time() - start_time
    
    # Benchmark Numba version
    print("\nTesting Numba Optimized Version...")
    start_time = time.time()
    numba_results = []
    
    for i, genome in enumerate(test_genomes):
        result = run_simulation_numba(genome, steps=steps, act_func="tanh", collect_data=False)
        numba_results.append(result[0])
        print(f"  Test {i+1}/{num_tests}: Fitness = {result[0]:.4f}")
    
    numba_time = time.time() - start_time
    
    # Compare results
    print("\n=== Performance Results ===")
    print(f"Original Version Time: {original_time:.2f} seconds")
    print(f"Numba Version Time: {numba_time:.2f} seconds")
    
    if numba_time > 0:
        speedup = original_time / numba_time
        print(f"Speedup: {speedup:.2f}x")
    
    # Compare fitness results
    fitness_diff = np.array(original_results) - np.array(numba_results)
    print(f"\nFitness Comparison:")
    print(f"  Mean difference: {np.mean(fitness_diff):.6f}")
    print(f"  Max difference: {np.max(np.abs(fitness_diff)):.6f}")
    print(f"  Std difference: {np.std(fitness_diff):.6f}")
    
    if np.max(np.abs(fitness_diff)) < 1e-3:
        print("✓ Results are very similar!")
    elif np.max(np.abs(fitness_diff)) < 0.1:
        print("~ Results are reasonably close")
    else:
        print("⚠ Results differ significantly - may need tuning")


def benchmark_neural_network_performance():
    """Benchmark just the neural network computation performance."""
    
    from gecko_main_numba import (
        simulation_step_numba, decode_genome_numba,
        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, MAX_VELOCITY, MAX_ANGLE
    )
    
    # Generate test data
    num_iterations = 10000
    genome = np.random.randn(GENOME_SIZE) * 0.1
    W_in, W_rec, W_out = decode_genome_numba(genome)
    
    qpos = np.random.randn(15) * 0.1  # BASE_INPUT_SIZE
    h = np.zeros(HIDDEN_SIZE)
    ctrl_state = np.zeros(OUTPUT_SIZE)
    
    print("\n=== Neural Network Benchmark ===")
    print(f"Running {num_iterations} NN forward passes")
    
    # Warm up Numba compilation
    print("Warming up Numba compilation...")
    for _ in range(100):
        h, ctrl_state = simulation_step_numba(
            qpos, h, ctrl_state, W_in, W_rec, W_out,
            0, 1000, MAX_VELOCITY, MAX_ANGLE, True
        )
    
    # Benchmark Numba version
    print("Benchmarking Numba NN performance...")
    start_time = time.time()
    
    for t in range(num_iterations):
        h, ctrl_state = simulation_step_numba(
            qpos, h, ctrl_state, W_in, W_rec, W_out,
            t, num_iterations, MAX_VELOCITY, MAX_ANGLE, True
        )
    
    numba_nn_time = time.time() - start_time
    
    print(f"Numba NN Time: {numba_nn_time:.4f} seconds")
    print(f"Time per forward pass: {(numba_nn_time / num_iterations) * 1000:.4f} ms")
    print(f"Forward passes per second: {num_iterations / numba_nn_time:.0f}")


def memory_usage_test():
    """Test memory usage of both versions."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        print("\n=== Memory Usage Test ===")
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Baseline memory: {baseline_memory:.1f} MB")
        
        # Test original version
        genome = torch.randn(GENOME_SIZE) * INITIAL_WEIGHT_RANGE
        result = run_simulation_original(genome, steps=500, act_func="tanh")
        
        original_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"After original simulation: {original_memory:.1f} MB (+{original_memory - baseline_memory:.1f})")
        
        # Test Numba version
        result = run_simulation_numba(genome, steps=500, act_func="tanh")
        
        numba_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"After Numba simulation: {numba_memory:.1f} MB (+{numba_memory - baseline_memory:.1f})")
        
    except ImportError:
        print("psutil not available for memory testing")


if __name__ == "__main__":
    print("Gecko Performance Benchmark Tool")
    print("="*50)
    
    # Run benchmarks
    benchmark_simulation_performance()
    benchmark_neural_network_performance()
    memory_usage_test()
    
    print("\n" + "="*50)
    print("Benchmark complete!")
    
    print("\nTo run the Numba-optimized version:")
    print("  python gecko_main_numba.py")
    
    print("\nTo run the original version:")
    print("  python gecko_main.py")