# =======================
# EvoTorch + MuJoCo Gecko Example (with time + recurrence + velocity integration)
# =======================

# Third-party libraries
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES
import matplotlib.pyplot as plt

import mujoco
from mujoco import mj_step
from mujoco import viewer
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Turn off evotorch spam messages
import logging
logging.getLogger("evotorch").setLevel(logging.WARNING)

# Reproducibility
import random
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# -----------------------
# Network architecture
# -----------------------
BASE_INPUT_SIZE = 15      # qpos values
TIME_INPUTS = 2           # sin/cos of time
INPUT_SIZE = BASE_INPUT_SIZE + TIME_INPUTS
HIDDEN_SIZE = 8
OUTPUT_SIZE = 8           # number of actuators (controls)

# Genome encodes W_in, W_rec, W_out
GENOME_SIZE = INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE

POPULATION_SIZE = 10
GENERATIONS = 10
SIMULATION_STEPS = 100   # shorter for testing (can increase)
INITIAL_WEIGHT_RANGE = 1.0

# Scaling parameters for velocity-based control
MAX_VELOCITY = 0.05       # radians per simulation step
MAX_ANGLE = np.pi / 2     # actuator limits


def decode_genome(genome: torch.Tensor):
    """Decode genome into input, recurrent, and output weights."""
    # call numpy earlier to avoid 3x. Put into CPU for handling
    g = genome.detach().cpu().numpy()
    
    idx = 0
    W_in = g[idx: idx + INPUT_SIZE*HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += INPUT_SIZE*HIDDEN_SIZE
    W_rec = g[idx: idx + HIDDEN_SIZE*HIDDEN_SIZE].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
    idx += HIDDEN_SIZE*HIDDEN_SIZE
    W_out = g[idx: idx + HIDDEN_SIZE*OUTPUT_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    return W_in, W_rec, W_out


# -----------------------
# Activation functions
# -----------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


# -----------------------
# Normalization helpers
# -----------------------
def normalize_inputs(qpos: np.ndarray) -> np.ndarray:
    """Normalize qpos angles to [-1, 1]."""
    return np.clip(qpos / np.pi, -1.0, 1.0)


def velocity_to_target(prev_ctrl: np.ndarray, raw_outputs: np.ndarray) -> np.ndarray:
    """Convert NN outputs ([-1, 1]) to incremental velocity updates for actuators."""
    velocity = raw_outputs * MAX_VELOCITY
    new_ctrl = prev_ctrl + velocity
    return np.clip(new_ctrl, -MAX_ANGLE, MAX_ANGLE)


# -----------------------
# Simulation function
# -----------------------
def run_simulation(genome: torch.Tensor, steps: int = 500, act_func: str = "tanh") -> float:
    """Run simulation with recurrent NN controller (velocity-based)."""
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Track core
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # Decode genome
    W_in, W_rec, W_out = decode_genome(genome)

    # Hidden state initialized to zero
    h = np.zeros(HIDDEN_SIZE)

    # Control state (actuator angles), initialized to zero
    ctrl_state = np.zeros(OUTPUT_SIZE)

    activation = tanh if act_func == "tanh" else sigmoid
    positions = np.empty(steps)
    inputs = np.empty(INPUT_SIZE)

    # Simulation loop
    for t in range(steps):
        # Normalize qpos inputs
        inputs[:BASE_INPUT_SIZE] = normalize_inputs(data.qpos)

        # Add time features
        phase = 2 * np.pi * (t / steps)   # normalized time
        inputs[BASE_INPUT_SIZE] = np.sin(phase)
        inputs[BASE_INPUT_SIZE + 1] = np.cos(phase)

        # Recurrent NN forward pass
        h = activation(np.dot(inputs, W_in) + np.dot(h, W_rec))  # recurrent update
        raw_outputs = activation(np.dot(h, W_out))

        # Update actuator states incrementally
        ctrl_state = velocity_to_target(ctrl_state, raw_outputs)

        # Apply controls
        data.ctrl[:] = ctrl_state
        mj_step(model, data)

        positions[t] = to_track[0].xpos[1]

    # Fitness = final y-position
    final_y = to_track[0].xpos[1]
    return final_y, positions


# -----------------------
# Fitness Function
# -----------------------
def evaluate_function(steps: int = SIMULATION_STEPS, act_func: str = "tanh"):
    """Return an evaluation function configured with activation type."""
    def evaluate(genomes: torch.Tensor) -> torch.Tensor:
        if genomes.ndim == 1:
            genomes = genomes.unsqueeze(0)

        fitnesses = []
        for genome in genomes:
            final_y, _ = run_simulation(
                genome, steps=steps, act_func=act_func)
            fitnesses.append(float(final_y))

        return torch.as_tensor(fitnesses, dtype=torch.float32)

    return evaluate


# -----------------------
# EvoTorch runner
# -----------------------
def run_evolution(
    generations: int = GENERATIONS,
    popsize: int = POPULATION_SIZE,
    steps: int = SIMULATION_STEPS,
    act_func: str = "tanh"
):
    problem = Problem(
        "max",
        evaluate_function(steps=steps, act_func=act_func),
        solution_length=GENOME_SIZE,
        dtype=torch.float32,
        initial_bounds=(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE),
        vectorized=True,
    )
    
    searcher = CMAES(
        problem, 
        popsize=popsize, 
        stdev_init=0.5
        )
    
    fitness_history = []
    for _ in range(generations):
        searcher.step()
        best_fit = searcher.status["best_eval"]
        fitness_history.append(best_fit)

    best_genome = searcher.status["best"].values
    return best_genome, fitness_history


# -----------------------
# Visualize Best Genome
# -----------------------
def run_best_genome(
        best_genome,
        act_func = "tanh"
        ):
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    W_in, W_rec, W_out = decode_genome(best_genome)
    h = np.zeros(HIDDEN_SIZE)
    ctrl_state = np.zeros(OUTPUT_SIZE)

    activation = tanh if act_func == "tanh" else sigmoid

    def control_callback(model, data):
        nonlocal h, ctrl_state
        # Inputs
        q_inputs = normalize_inputs(data.qpos)

        t = data.time

        phase = 2 * np.pi * (t / 5.0)  # slow oscillator
        time_inputs = np.array([np.sin(phase), np.cos(phase)])
        inputs = np.concatenate([q_inputs, time_inputs])

        # Forward recurrent update
        h = activation(np.dot(inputs, W_in) + np.dot(h, W_rec))
        raw_outputs = activation(np.dot(h, W_out))

        # Incremental velocity update
        ctrl_state = velocity_to_target(ctrl_state, raw_outputs)

        data.ctrl[:] = ctrl_state

    mujoco.set_mjcb_control(control_callback)
    viewer.launch(model=model, data=data)


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


def plot_individual_run(fitness_history: list[float], activation: str):
    """Plot fitness history for a single evolutionary run."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(fitness_history, marker="o", linestyle="-", color="b", label="Best Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (final y)")
    ax.set_title(f"{activation} Single Run")
    ax.legend()
    ax.grid(True)

    return fig


def tanh_vs_sigmoid():
    """Runs a tanh vs sigmoid evaluation over several runs."""
    return


if __name__ == "__main__":
    generations = GENERATIONS
    steps = SIMULATION_STEPS
    activation = "tanh" # options: "tanh", "sigmoid"

    # Run the evolutionary algorithm
    genome, fitness_history = run_evolution(
            generations=generations,
            popsize=10,
            steps=steps,
            act_func=activation
        )
    
    final_fitness = fitness_history[-1]

    # Simulate the velocity
    _, trajectory_vel = run_simulation(
            genome=genome,
            steps=steps,
            act_func=activation
        )
    np.save(f"results/best_{activation}_traj.npy", trajectory_vel)

    # Change later for multiple runs
    histories = fitness_history

    # convert to array
    histories = np.array(histories)

    # Run and save analysis
    fig1 = plot_individual_run(fitness_history, activation)
    fig1.savefig(f"results/{activation}_individual_runs_gens{generations}_steps{steps}.png")

    # Show plots
    plt.show()

    # Show genome
    run_best_genome(genome, act_func=activation)



