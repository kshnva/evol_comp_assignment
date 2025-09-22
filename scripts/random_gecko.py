# evo_gecko.py
# Evolutionary algorithm for gecko NN controller (headless rollouts + final viewer)
# Requirements: numpy, mujoco, matplotlib, ariel libs as in your original script

import os
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# ============================================================
# Globals / Hyperparameters
# ============================================================
HISTORY = []

# NN architecture
HIDDEN_SIZE = 8

# EA hyperparameters
POP_SIZE = 30
GENERATIONS = 5
ELITE_COUNT = 6
MUTATION_STD = 0.1
ROLLOUT_STEPS = 400     # number of simulation steps per rollout
DELTA = 0.05             # smoother control updates
HINGE_RANGE = np.pi / 2


# ============================================================
# Neural Network Setup
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def flatten_weights(W1, W2, W3):
    return np.concatenate([W1.ravel(), W2.ravel(), W3.ravel()])

def decode(genotype, input_size, hidden_size, output_size):
    """Decode flattened genotype into W1, W2, W3 matrices."""
    idx = 0
    s1 = input_size * hidden_size
    W1 = genotype[idx:idx+s1].reshape(input_size, hidden_size)
    idx += s1

    s2 = hidden_size * hidden_size
    W2 = genotype[idx:idx+s2].reshape(hidden_size, hidden_size)
    idx += s2

    s3 = hidden_size * output_size
    W3 = genotype[idx:idx+s3].reshape(hidden_size, output_size)

    return W1, W2, W3

def nn_forward(qpos, W1, W2, W3):
    """Simple 3-layer NN forward pass. qpos is 1D array."""
    l1 = sigmoid(np.dot(qpos, W1))
    l2 = sigmoid(np.dot(l1, W2))
    out = sigmoid(np.dot(l2, W3))   # in (0,1)
    return out


# ============================================================
# Simulation / Rollouts
# ============================================================
def build_world_and_model():
    """Create world, spawn gecko, compile model and return model, data, tracking info."""
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # find gecko "core" geom to track
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # save initial qpos/qvel
    init_qpos = data.qpos.copy()
    init_qvel = data.qvel.copy()

    return world, model, data, to_track, init_qpos, init_qvel

def reset_sim(data, init_qpos, init_qvel):
    """Reset data to initial state (qpos, qvel, ctrl)."""
    data.qpos[:] = init_qpos
    data.qvel[:] = init_qvel
    data.ctrl[:] = 0.0
    mujoco.mj_forward(data.model, data)

def rollout(genotype, model, data, to_track, init_qpos, init_qvel,
            steps=ROLLOUT_STEPS, hidden_size=HIDDEN_SIZE, delta=DELTA):
    """
    Run a headless rollout using the provided genotype.
    Returns fitness (final x) and HISTORY of positions.
    """
    global HISTORY
    HISTORY = []

    input_size = len(data.qpos)
    output_size = model.nu

    W1, W2, W3 = decode(genotype, input_size, hidden_size, output_size)
    reset_sim(data, init_qpos, init_qvel)

    for _ in range(steps):
        q = data.qpos.copy()
        raw_out = nn_forward(q, W1, W2, W3)
        scaled = (raw_out * 2.0 - 1.0) * HINGE_RANGE

        data.ctrl += scaled * delta
        data.ctrl = np.clip(data.ctrl, -HINGE_RANGE, HINGE_RANGE)

        if len(to_track) > 0:
            HISTORY.append(to_track[0].xpos.copy())

        mujoco.mj_step(model, data)

    if len(HISTORY) == 0:
        return -np.inf, np.array([])
    final_x = HISTORY[-1][0]
    return float(final_x), np.array(HISTORY)


# ============================================================
# EA: Genetic Operators
# ============================================================
def initialize_population(genome_size, pop_size=POP_SIZE):
    return [np.random.randn(genome_size) * 0.1 for _ in range(pop_size)]

def mutate(genome, mutation_std=MUTATION_STD):
    return genome + np.random.randn(genome.size) * mutation_std

def reproduce(population, fitnesses, elite_count=ELITE_COUNT):
    """Elitism + mutation only (no crossover)."""
    order = np.argsort(fitnesses)[::-1]
    sorted_pop = [population[i] for i in order]
    elites = [g.copy() for g in sorted_pop[:elite_count]]
    new_pop = elites[:]
    while len(new_pop) < len(population):
        parent = elites[np.random.randint(elite_count)]
        child = mutate(parent)
        new_pop.append(child)
    return new_pop, sorted_pop[0], fitnesses[order][0]


# ============================================================
# EA Loop
# ============================================================
def evolve(model, data, to_track, init_qpos, init_qvel):
    input_size = len(init_qpos)
    output_size = model.nu
    genome_size = input_size*HIDDEN_SIZE + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*output_size

    population = initialize_population(genome_size, POP_SIZE)
    best_genotype, best_score = None, -np.inf
    best_over_gens = []

    for gen in range(GENERATIONS):
        fitnesses = [rollout(ind, model, data, to_track, init_qpos, init_qvel)[0]
                     for ind in population]

        population, gen_best, gen_best_score = reproduce(population, np.array(fitnesses))

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_genotype = gen_best.copy()

        best_over_gens.append(gen_best_score)
        print(f"Gen {gen+1}/{GENERATIONS}  best={gen_best_score:.4f}  mean={np.mean(fitnesses):.4f}")

    return best_genotype, best_score, best_over_gens


# ============================================================
# Main
# ============================================================
def main():
    world, model, data, to_track, init_qpos, init_qvel = build_world_and_model()
    print("Model built. Input size:", len(init_qpos), "Output size (nu):", model.nu)

    best_genotype, best_score, best_over_gens = evolve(model, data, to_track, init_qpos, init_qvel)

    # Save outputs in Downloads
    downloads_dir = os.path.expanduser("~/Downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    np.save(os.path.join(downloads_dir, "best_genotype.npy"), best_genotype)
    print(f"Saved best genotype (fitness={best_score:.4f}) to {downloads_dir}/best_genotype.npy")

    # # Save fitness curve
    # plt.figure(figsize=(8,5))
    # plt.plot(best_over_gens, marker='o')
    # plt.xlabel("Generation"); plt.ylabel("Best Fitness")
    # plt.title("Best Fitness over Generations")
    # plt.grid(True)
    # plot_path = os.path.join(downloads_dir, "fitness_curve.png")
    # plt.savefig(plot_path, dpi=150); plt.close()
    # print(f"Saved fitness curve to {plot_path}")

    # # Final demo with viewer
    # print("Launching final rollout in viewer...")
    # reset_sim(data, init_qpos, init_qvel)
    # W1, W2, W3 = decode(best_genotype, len(init_qpos), HIDDEN_SIZE, model.nu)

    # def best_ctrl_cb(m, d):
    #     q = d.qpos.copy()
    #     raw_out = nn_forward(q, W1, W2, W3)
    #     scaled = (raw_out * 2.0 - 1.0) * HINGE_RANGE
    #     d.ctrl += scaled * DELTA
    #     d.ctrl[:] = np.clip(d.ctrl, -HINGE_RANGE, HINGE_RANGE)
    #     if len(to_track) > 0:
    #         HISTORY.append(to_track[0].xpos.copy())

