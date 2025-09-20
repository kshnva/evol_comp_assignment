# =======================
# EvoTorch + MuJoCo Gecko Example (with time + recurrence)
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


def scale_outputs(raw_outputs: np.ndarray) -> np.ndarray:
    """Map tanh outputs [-1, 1] into actuator range."""
    max_angle = np.pi / 2
    return np.clip(raw_outputs * max_angle, -max_angle, max_angle)


# -----------------------
# Simulation function
# -----------------------
def run_simulation(genome: torch.Tensor, steps: int = 500) -> float:
    """Run simulation with recurrent NN controller."""
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

    # Simulation loop
    for t in range(steps):
        # Normalize qpos inputs
        q_inputs = normalize_inputs(data.qpos)

        # Add time features
        phase = 2 * np.pi * (t / steps)   # normalized time
        time_inputs = np.array([np.sin(phase), np.cos(phase)])

        inputs = np.concatenate([q_inputs, time_inputs])

        # Recurrent NN forward pass
        h = tanh(np.dot(inputs, W_in) + np.dot(h, W_rec))  # recurrent update
        raw_outputs = tanh(np.dot(h, W_out))

        # Scale to actuator range
        controls = scale_outputs(raw_outputs)

        # Apply controls
        data.ctrl[:] = controls
        mj_step(model, data)

    # Fitness = final y-position
    final_y = to_track[0].xpos[1]
    return final_y


# -----------------------
# Fitness Function
# -----------------------
def evaluate(genome: torch.Tensor) -> float:
    return run_simulation(genome, steps=SIMULATION_STEPS)


# -----------------------
# EvoTorch Setup
# -----------------------
problem = Problem(
    "max",
    evaluate,
    solution_length=GENOME_SIZE,
    dtype=torch.float32,
    initial_bounds=(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE),
)

searcher = CMAES(
    problem,
    popsize=POPULATION_SIZE,
    stdev_init=0.5,
)

fitness_history = []

for gen in range(GENERATIONS):
    searcher.step()
    best_fit = searcher.status["best_eval"]
    fitness_history.append(best_fit)
    print(f"Gen {gen} | Best Y: {best_fit:.4f}")

best_genome = searcher.status["best"].values
print("\nBest genome found:", best_genome)


# -----------------------
# Plot Fitness Progress
# -----------------------
plt.figure(figsize=(8, 5))
plt.plot(fitness_history, marker="o", linestyle="-", color="b")
plt.xlabel("Generation")
plt.ylabel("Best Fitness (final y)")
plt.title("Evolution Progress (CMA-ES)")
plt.grid(True)
plt.show()


# -----------------------
# Visualize Best Genome
# -----------------------
def run_best_genome(best_genome):
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    W_in, W_rec, W_out = decode_genome(best_genome)
    h = np.zeros(HIDDEN_SIZE)

    def control_callback(model, data):
        nonlocal h
        # Inputs
        q_inputs = normalize_inputs(data.qpos)
        t = data.time
        phase = 2 * np.pi * (t / 5.0)  # slow oscillator
        time_inputs = np.array([np.sin(phase), np.cos(phase)])
        inputs = np.concatenate([q_inputs, time_inputs])

        # Forward recurrent update
        h = tanh(np.dot(inputs, W_in) + np.dot(h, W_rec))
        raw_outputs = tanh(np.dot(h, W_out))
        controls = scale_outputs(raw_outputs)

        data.ctrl[:] = controls

    mujoco.set_mjcb_control(control_callback)
    viewer.launch(model=model, data=data)


if __name__ == "__main__":
    run_best_genome(best_genome)
