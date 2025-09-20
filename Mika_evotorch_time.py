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
GENERATIONS = 50
SIMULATION_STEPS = 5000   # shorter for testing (can increase)
INITIAL_WEIGHT_RANGE = 1.0

# Scaling parameters for velocity-based control
MAX_VELOCITY = 0.05       # radians per simulation step
MAX_ANGLE = np.pi / 2     # actuator limits


def decode_genome(genome: torch.Tensor):
    """Decode genome into input, recurrent, and output weights."""
    idx = 0
    W_in = genome[idx: idx + INPUT_SIZE*HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += INPUT_SIZE*HIDDEN_SIZE
    W_rec = genome[idx: idx + HIDDEN_SIZE*HIDDEN_SIZE].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
    idx += HIDDEN_SIZE*HIDDEN_SIZE
    W_out = genome[idx: idx + HIDDEN_SIZE*OUTPUT_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    return W_in.numpy(), W_rec.numpy(), W_out.numpy()


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
def run_simulation(genome: torch.Tensor, steps: int = 500, activation: str = "tanh") -> float:
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

    # Choose activation function
    act_fn = tanh if activation == "tanh" else sigmoid

    # Simulation loop
    for t in range(steps):
        # Normalize qpos inputs
        q_inputs = normalize_inputs(data.qpos)

        # Add time features
        phase = 2 * np.pi * (t / steps)   # normalized time
        time_inputs = np.array([np.sin(phase), np.cos(phase)])

        inputs = np.concatenate([q_inputs, time_inputs])

        # Recurrent NN forward pass
        h = act_fn(np.dot(inputs, W_in) + np.dot(h, W_rec))  # recurrent update
        raw_outputs = act_fn(np.dot(h, W_out))  # outputs

        # Update actuator states incrementally
        ctrl_state = velocity_to_target(ctrl_state, raw_outputs)

        # Apply controls
        data.ctrl[:] = ctrl_state
        mj_step(model, data)

    # Fitness = final y-position
    final_y = to_track[0].xpos[1]
    return final_y


# -----------------------
# Fitness Function
# -----------------------
def evaluate_factory(steps: int = SIMULATION_STEPS, activation: str = "tanh"):
    """Return an evaluation function configured with activation type."""
    def evaluate(genome: torch.Tensor) -> float:
        return run_simulation(genome, steps=steps, activation=activation)
    return evaluate


# -----------------------
# EvoTorch Runner
# -----------------------
def run_evolution(
    generations: int = GENERATIONS,
    popsize: int = POPULATION_SIZE,
    steps: int = SIMULATION_STEPS,
    activation: str = "tanh",
):
    problem = Problem(
        "max",
        evaluate_factory(steps=steps, activation=activation),
        solution_length=GENOME_SIZE,
        dtype=torch.float32,
        initial_bounds=(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE),
    )

    searcher = CMAES(
        problem,
        popsize=popsize,
        stdev_init=0.5,
    )

    fitness_history = []
    for gen in range(generations):
        searcher.step()
        best_fit = searcher.status["best_eval"]
        fitness_history.append(best_fit)
        #print(f"Gen {gen} | Best Y: {best_fit:.4f}")

    best_genome = searcher.status["best"].values
    return best_genome, fitness_history


# -----------------------
# Visualize Best Genome
# -----------------------
def run_best_genome(best_genome, activation: str = "tanh"):
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    W_in, W_rec, W_out = decode_genome(best_genome)
    h = np.zeros(HIDDEN_SIZE)
    ctrl_state = np.zeros(OUTPUT_SIZE)

    act_fn = tanh if activation == "tanh" else sigmoid

    def control_callback(model, data):
        nonlocal h, ctrl_state
        # Inputs
        q_inputs = normalize_inputs(data.qpos)
        t = data.time
        phase = 2 * np.pi * (t / 5.0)  # slow oscillator
        time_inputs = np.array([np.sin(phase), np.cos(phase)])
        inputs = np.concatenate([q_inputs, time_inputs])

        # Forward recurrent update
        h = act_fn(np.dot(inputs, W_in) + np.dot(h, W_rec))
        raw_outputs = act_fn(np.dot(h, W_out))

        # Incremental velocity update
        ctrl_state = velocity_to_target(ctrl_state, raw_outputs)

        data.ctrl[:] = ctrl_state

    mujoco.set_mjcb_control(control_callback)
    viewer.launch(model=model, data=data)
