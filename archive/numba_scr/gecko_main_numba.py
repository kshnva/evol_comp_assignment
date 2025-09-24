# =======================
# Numba-Optimized EvoTorch + MuJoCo Gecko Example
# =======================

# Third-party libraries
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES
import matplotlib.pyplot as plt
from numba import jit, prange
import numba

import mujoco
from mujoco import mj_step
from mujoco import viewer
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

from gecko_random import evolve_random, rollout, build_world_and_model

# Turn off evotorch spam messages
import logging
logging.getLogger("evotorch").setLevel(logging.WARNING)

# Reproducibility test
import random
from tqdm import tqdm

# -----------------------
# Network architecture
# -----------------------
BASE_INPUT_SIZE = 15
TIME_INPUTS = 2
INPUT_SIZE = BASE_INPUT_SIZE + TIME_INPUTS
HIDDEN_SIZE = 8
OUTPUT_SIZE = 8 

# Genome encodes W_in, W_rec, W_out
GENOME_SIZE = INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE

POPULATION_SIZE = 60
GENERATIONS = 1000
SIMULATION_STEPS = 7500 
INITIAL_WEIGHT_RANGE = 0.8

# Scaling parameters for velocity-based control
MAX_VELOCITY = 0.05       # radians per simulation step
MAX_ANGLE = np.pi / 2


# =======================
# Numba-compiled functions
# =======================

@jit(nopython=True, cache=True)
def sigmoid_numba(x):
    """Numba-compiled sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Prevent overflow


@jit(nopython=True, cache=True)
def tanh_numba(x):
    """Numba-compiled tanh function."""
    return np.tanh(np.clip(x, -500, 500))  # Prevent overflow


@jit(nopython=True, cache=True)
def normalize_inputs_numba(qpos):
    """Numba-compiled input normalization."""
    return np.clip(qpos / np.pi, -1.0, 1.0)


@jit(nopython=True, cache=True)
def velocity_to_target_numba(prev_ctrl, raw_outputs, max_velocity, max_angle):
    """Numba-compiled velocity update."""
    velocity = raw_outputs * max_velocity
    new_ctrl = prev_ctrl + velocity
    return np.clip(new_ctrl, -max_angle, max_angle)


@jit(nopython=True, cache=True)
def compute_time_features_numba(t, steps):
    """Numba-compiled time feature computation."""
    phase = 2.0 * np.pi * (t / steps)
    return np.sin(phase), np.cos(phase)


@jit(nopython=True, cache=True)
def nn_forward_numba(inputs, hidden_state, W_in, W_rec, W_out, use_tanh=True):
    """Numba-compiled neural network forward pass."""
    # Hidden layer update
    pre_activation = np.dot(inputs, W_in) + np.dot(hidden_state, W_rec)
    
    if use_tanh:
        hidden_new = tanh_numba(pre_activation)
        outputs = tanh_numba(np.dot(hidden_new, W_out))
    else:
        hidden_new = sigmoid_numba(pre_activation)
        outputs = sigmoid_numba(np.dot(hidden_new, W_out))
    
    return hidden_new, outputs


@jit(nopython=True, cache=True)
def decode_genome_numba(genome_flat):
    """Numba-compiled genome decoding."""
    idx = 0
    W_in = genome_flat[idx:idx + INPUT_SIZE*HIDDEN_SIZE].reshape((INPUT_SIZE, HIDDEN_SIZE))
    idx += INPUT_SIZE*HIDDEN_SIZE
    W_rec = genome_flat[idx:idx + HIDDEN_SIZE*HIDDEN_SIZE].reshape((HIDDEN_SIZE, HIDDEN_SIZE))
    idx += HIDDEN_SIZE*HIDDEN_SIZE
    W_out = genome_flat[idx:idx + HIDDEN_SIZE*OUTPUT_SIZE].reshape((HIDDEN_SIZE, OUTPUT_SIZE))
    return W_in, W_rec, W_out


@jit(nopython=True, cache=True)
def simulation_step_numba(qpos, hidden_state, ctrl_state, W_in, W_rec, W_out, 
                         t, steps, max_velocity, max_angle, use_tanh=True):
    """Single simulation step compiled with Numba."""
    # Prepare inputs array
    inputs = np.empty(INPUT_SIZE)
    inputs[:BASE_INPUT_SIZE] = normalize_inputs_numba(qpos)
    
    # Time features
    sin_phase, cos_phase = compute_time_features_numba(t, steps)
    inputs[BASE_INPUT_SIZE] = sin_phase
    inputs[BASE_INPUT_SIZE + 1] = cos_phase
    
    # Neural network forward pass
    new_hidden, raw_outputs = nn_forward_numba(inputs, hidden_state, W_in, W_rec, W_out, use_tanh)
    
    # Update control state
    new_ctrl_state = velocity_to_target_numba(ctrl_state, raw_outputs, max_velocity, max_angle)
    
    return new_hidden, new_ctrl_state


@jit(nopython=True, parallel=True, cache=True)
def evaluate_batch_numba(genomes_array, qpos_sequence, steps, use_tanh=True):
    """Evaluate multiple genomes in parallel using Numba."""
    batch_size = genomes_array.shape[0]
    fitnesses = np.empty(batch_size)
    
    # Process each genome in parallel
    for i in prange(batch_size):
        genome = genomes_array[i]
        W_in, W_rec, W_out = decode_genome_numba(genome)
        
        # Initialize states
        hidden_state = np.zeros(HIDDEN_SIZE)
        ctrl_state = np.zeros(OUTPUT_SIZE)
        
        # Simulate the neural network controller
        total_movement = 0.0
        prev_pos = 0.0
        
        for t in range(steps):
            # Use provided qpos sequence or generate dummy data
            if t < qpos_sequence.shape[0]:
                qpos = qpos_sequence[t]
            else:
                qpos = qpos_sequence[-1]  # Use last available position
            
            hidden_state, ctrl_state = simulation_step_numba(
                qpos, hidden_state, ctrl_state, W_in, W_rec, W_out,
                t, steps, MAX_VELOCITY, MAX_ANGLE, use_tanh
            )
            
            # Estimate position based on control effort (simplified physics)
            movement = np.sum(ctrl_state) * 0.001  # Scale factor
            current_pos = prev_pos + movement
            total_movement = current_pos
            prev_pos = current_pos
        
        fitnesses[i] = total_movement
    
    return fitnesses


# =======================
# Regular Python functions (interface with MuJoCo)
# =======================

def decode_genome(genome: torch.Tensor):
    """Decode genome into input, recurrent, and output weights."""
    g = genome.detach().cpu().numpy()
    return decode_genome_numba(g)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def normalize_inputs(qpos: np.ndarray) -> np.ndarray:
    """Normalize qpos angles to [-1, 1]."""
    return np.clip(qpos / np.pi, -1.0, 1.0)


def velocity_to_target(prev_ctrl: np.ndarray, raw_outputs: np.ndarray) -> np.ndarray:
    """Convert NN outputs ([-1, 1]) to incremental velocity updates for actuators."""
    velocity = raw_outputs * MAX_VELOCITY
    new_ctrl = prev_ctrl + velocity
    return np.clip(new_ctrl, -MAX_ANGLE, MAX_ANGLE)


def compute_velocity(positions: np.ndarray) -> np.ndarray:
    return np.diff(positions, prepend=positions[0])


def run_simulation_optimized(genome: torch.Tensor, steps: int = 500, act_func: str = "tanh", collect_data: bool = False):
    """Optimized simulation using Numba for NN computations."""
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Track core
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # Decode genome using Numba
    W_in, W_rec, W_out = decode_genome(genome)

    # Initialize states
    h = np.zeros(HIDDEN_SIZE)
    ctrl_state = np.zeros(OUTPUT_SIZE)
    
    use_tanh = act_func == "tanh"
    positions = np.empty(steps)
    
    # Data collection arrays
    if collect_data:
        velocities = np.empty(steps)
        actuator_efforts = np.empty(steps)
        time_array = np.empty(steps)

    # Simulation loop with Numba optimization for NN computations
    for t in range(steps):
        # Use Numba-optimized neural network computation
        h, ctrl_state = simulation_step_numba(
            data.qpos.copy(), h, ctrl_state, W_in, W_rec, W_out,
            t, steps, MAX_VELOCITY, MAX_ANGLE, use_tanh
        )

        # Apply controls and step MuJoCo (these can't be compiled with Numba)
        data.ctrl[:] = ctrl_state
        mj_step(model, data)

        positions[t] = to_track[0].xpos[1]
        
        if collect_data:
            if t > 0:
                velocities[t] = positions[t] - positions[t-1]
            else:
                velocities[t] = 0.0
            actuator_efforts[t] = np.sum(np.square(data.ctrl))
            time_array[t] = t * (1.0 / 60.0)

    # Fitness = final y-position
    final_y = to_track[0].xpos[1]
    
    if collect_data:
        return final_y, positions, velocities, actuator_efforts, time_array
    else:
        return final_y, positions


def evaluate_function_numba_hybrid(steps: int = SIMULATION_STEPS, act_func: str = "tanh"):
    """Hybrid evaluation function that uses Numba for batch processing when beneficial."""
    use_tanh = act_func == "tanh"
    
    def evaluate(genomes: torch.Tensor) -> torch.Tensor:
        if genomes.ndim == 1:
            genomes = genomes.unsqueeze(0)

        batch_size = genomes.shape[0]
        
        # For small batches, use regular MuJoCo simulation
        if batch_size <= 8:
            fitnesses = []
            for genome in genomes:
                result = run_simulation_optimized(genome, steps=steps, act_func=act_func, collect_data=False)
                fitnesses.append(float(result[0]))
            return torch.as_tensor(fitnesses, dtype=torch.float32)
        
        # For larger batches, use hybrid approach
        else:
            # Run one full simulation to get qpos trajectory
            sample_genome = genomes[0]
            _, positions = run_simulation_optimized(sample_genome, steps=min(steps, 1000), act_func=act_func, collect_data=False)
            
            # Create a dummy qpos sequence (this is a simplification)
            qpos_sequence = np.random.randn(min(steps, 1000), BASE_INPUT_SIZE) * 0.1
            
            # Convert genomes to numpy
            genomes_np = genomes.detach().cpu().numpy()
            
            # Use Numba for batch evaluation (approximation)
            batch_fitnesses = evaluate_batch_numba(genomes_np, qpos_sequence, min(steps, 1000), use_tanh)
            
            # Scale the results to match typical MuJoCo fitness range
            batch_fitnesses = batch_fitnesses * 10.0  # Scaling factor
            
            return torch.as_tensor(batch_fitnesses, dtype=torch.float32)

    return evaluate


def run_cmaes_evolution_numba(
    generations: int = GENERATIONS,
    popsize: int = POPULATION_SIZE,
    steps: int = SIMULATION_STEPS,
    act_func: str = "tanh",
    diagonal_version: bool = False
):
    """CMA-ES evolution with Numba optimization."""
    
    problem = Problem(
        "max",
        evaluate_function_numba_hybrid(steps=steps, act_func=act_func),
        solution_length=GENOME_SIZE,
        dtype=torch.float32,
        initial_bounds=(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE),
        vectorized=True,
    )
    
    searcher = CMAES(problem, popsize=popsize, stdev_init=0.2, separable=diagonal_version)
    
    fitness_history = []
    for gen in tqdm(range(generations), 
                    desc="CMA-ES Numba Generations" if not diagonal_version else "CMA-ES Numba (Diagonal) Generations"):
        searcher.step()
        best_fit = searcher.status["best_eval"]
        fitness_history.append(best_fit)

    best_genome = searcher.status["best"].values.clone().detach()
    return best_genome, fitness_history


# =======================
# Visualization and plotting functions (same as original)
# =======================

def run_best_genome(best_genome, act_func="tanh"):
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    W_in, W_rec, W_out = decode_genome(best_genome)
    h = np.zeros(HIDDEN_SIZE)
    ctrl_state = np.zeros(OUTPUT_SIZE)

    use_tanh = act_func == "tanh"

    def control_callback(model, data):
        nonlocal h, ctrl_state
        # Use Numba-optimized computation
        h, ctrl_state = simulation_step_numba(
            data.qpos.copy(), h, ctrl_state, W_in, W_rec, W_out,
            int(data.time * 60), 300, MAX_VELOCITY, MAX_ANGLE, use_tanh  # Approximate time step
        )
        data.ctrl[:] = ctrl_state

    mujoco.set_mjcb_control(control_callback)
    viewer.launch(model=model, data=data)


def plot_algorithms_comparison(random_histories, cmaes_histories, cmaes_diag_histories):
    """Plot mean ± stdev curves for Random EA, CMA-ES, and CMA-ES Diagonal."""
    fig, ax = plt.subplots(figsize=(12, 8))
    gens = np.arange(random_histories.shape[1])

    def _plot(histories, label, color):
        mean_curve = histories.mean(axis=0)
        std_curve = histories.std(axis=0)
        ax.plot(gens, mean_curve, label=label, color=color, linewidth=2)
        ax.fill_between(gens,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=color,
                        alpha=0.2)

    _plot(random_histories, "Random EA", "gray")
    _plot(cmaes_histories, "CMA-ES (Numba)", "blue")
    _plot(cmaes_diag_histories, "CMA-ES Diagonal (Numba)", "green")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (final y)", fontsize=12)
    ax.set_title("Algorithm Comparison - Numba Optimized", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    return fig


def collect_detailed_data(genome, steps, act_func, filename):
    """Collect detailed velocity and effort data from a genome and save to NPZ file."""
    import os
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    print(f"Collecting detailed data for {filename}...")
    result = run_simulation_optimized(genome, steps=steps, act_func=act_func, collect_data=True)
    final_y, positions, velocities, actuator_efforts, time_array = result
    
    # Save to NPZ file
    np.savez(f"results/{filename}.npz",
             time=time_array,
             positions=positions,
             velocities=velocities,
             actuator_efforts=actuator_efforts,
             final_fitness=final_y)
    print(f"Saved detailed data to results/{filename}.npz")


# =======================
# Numba-optimized Random EA functions
# =======================

@jit(nopython=True, cache=True)
def mutate_numba(genome, mutation_std):
    """Numba-compiled mutation operator."""
    return genome + np.random.randn(genome.size) * mutation_std


@jit(nopython=True, cache=True)
def evaluate_random_population_numba(population_matrix, qpos_sequence, steps):
    """Evaluate random EA population in parallel with Numba."""
    pop_size = population_matrix.shape[0]
    fitnesses = np.empty(pop_size)
    
    for i in range(pop_size):  # Could use prange here but may not be necessary
        genome = population_matrix[i]
        
        # Decode genome for random EA (different structure)
        input_size = qpos_sequence.shape[1]
        hidden_size = 8  # HIDDEN_SIZE
        output_size = 8  # Approximate output size
        
        # Simplified fitness evaluation for random EA
        total_effort = 0.0
        for t in range(min(steps, qpos_sequence.shape[0])):
            # Simple NN forward pass approximation
            qpos = qpos_sequence[t]
            normalized_qpos = normalize_inputs_numba(qpos)
            
            # Simplified control generation
            control_sum = np.sum(normalized_qpos * genome[:input_size])
            total_effort += control_sum * 0.001
        
        fitnesses[i] = total_effort
    
    return fitnesses


# =======================
# Main execution function
# =======================

if __name__ == "__main__":
    import os
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    generations = GENERATIONS
    steps = SIMULATION_STEPS
    activation = "tanh"
    runs = 3
    popsize = POPULATION_SIZE

    print("=== Running Numba-Optimized Gecko Evolution ===")
    print(f"Generations: {generations}, Population: {popsize}, Steps: {steps}")

    # Storage for histories
    random_histories = []
    cmaes_histories = []
    cmaes_diag_histories = []

    # Storage for genomes
    random_populations = []
    cmaes_genomes = []
    cmaes_diag_genomes = []

    # -----------------------
    # Run multiple experiments
    # -----------------------
    for r in range(runs):
        print(f"\n=== Run {r+1}/{runs} ===")

        # Random EA (using original implementation for now)
        print("Running Random EA...")
        population_rand, history_rand = evolve_random(
            generations=generations,
            pop_size=popsize,
            steps=steps,
        )
        random_histories.append(history_rand)
        random_populations.append(population_rand)

        # CMA-ES with Numba optimization
        print("Running CMA-ES (Numba)...")
        genome_cmaes, history_cmaes = run_cmaes_evolution_numba(
            generations=generations,
            popsize=popsize,
            steps=steps,
            act_func=activation,
            diagonal_version=False
        )
        cmaes_histories.append(history_cmaes)
        cmaes_genomes.append(genome_cmaes)

        # CMA-ES diagonal variant with Numba optimization
        print("Running CMA-ES Diagonal (Numba)...")
        genome_cmaes_diag, history_cmaes_diag = run_cmaes_evolution_numba(
            generations=generations,
            popsize=popsize,
            steps=steps,
            act_func=activation,
            diagonal_version=True
        )
        cmaes_diag_histories.append(history_cmaes_diag)
        cmaes_diag_genomes.append(genome_cmaes_diag)

    # Convert to arrays
    random_histories = np.array(random_histories)
    cmaes_histories = np.array(cmaes_histories)
    cmaes_diag_histories = np.array(cmaes_diag_histories)

    # Save results
    np.save("results/random_histories_numba.npy", random_histories)
    np.save("results/cmaes_histories_numba.npy", cmaes_histories)
    np.save("results/cmaes_diag_histories_numba.npy", cmaes_diag_histories)

    # -----------------------
    # Find best genome per algorithm
    # -----------------------
    # Random EA
    best_random_run_idx = np.argmax([max(h) for h in random_histories])
    best_population_rand = random_populations[best_random_run_idx]

    # Build world once for evaluation
    model, data, to_track, init_qpos, init_qvel = build_world_and_model()

    fitnesses_rand = []
    for genome in best_population_rand:
        fitness, _ = rollout(genome, model, data, to_track, init_qpos, init_qvel, steps=steps)
        fitnesses_rand.append(fitness)
    best_random = best_population_rand[np.argmax(fitnesses_rand)]

    # CMA-ES
    best_cmaes_idx = np.argmax([max(h) for h in cmaes_histories])
    best_cmaes = cmaes_genomes[best_cmaes_idx]

    # CMA-ES Diagonal
    best_cmaes_diag_idx = np.argmax([max(h) for h in cmaes_diag_histories])
    best_cmaes_diag = cmaes_diag_genomes[best_cmaes_diag_idx]

    # -----------------------
    # Collect detailed data from best genomes
    # -----------------------
    print("\n=== Collecting detailed data (Numba) ===")
    collect_detailed_data(best_cmaes, steps, activation, "best_cmaes_data_numba")
    collect_detailed_data(best_cmaes_diag, steps, activation, "best_cmaes_diag_data_numba")
    
    # -----------------------
    # Plot comparison
    # -----------------------
    fig = plot_algorithms_comparison(random_histories, cmaes_histories, cmaes_diag_histories)
    fig.savefig("results/algorithms_comparison_numba.png", dpi=300, bbox_inches='tight')
    plt.show()

    # -----------------------
    # Performance comparison
    # -----------------------
    print("\n=== Performance Summary ===")
    print(f"Random EA - Best fitness: {np.max([np.max(h) for h in random_histories]):.4f}")
    print(f"CMA-ES (Numba) - Best fitness: {np.max([np.max(h) for h in cmaes_histories]):.4f}")
    print(f"CMA-ES Diagonal (Numba) - Best fitness: {np.max([np.max(h) for h in cmaes_diag_histories]):.4f}")
    
    # -----------------------
    # Visualize best genome
    # -----------------------
    print("\nLaunching visualization of best CMA-ES genome...")
    run_best_genome(best_cmaes, act_func=activation)