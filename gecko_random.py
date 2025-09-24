# Random evolutionary algorithm for Gecko NN controller

import numpy as np
import mujoco
from tqdm import tqdm

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# ============================================================
# Globals / Hyperparameters
# ============================================================
HIDDEN_SIZE = 8
POP_SIZE = 30
GENERATIONS = 5
ELITE_COUNT = 6
MUTATION_STD = 0.1
ROLLOUT_STEPS = 400
DELTA = 0.05
HINGE_RANGE = np.pi / 2


# ============================================================
# Neural Network Setup
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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

    init_qpos = data.qpos.copy()
    init_qvel = data.qvel.copy()

    return model, data, to_track, init_qpos, init_qvel


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
    Returns fitness (final y-position) and trajectory of positions.
    """
    input_size = len(data.qpos)
    output_size = model.nu
    W1, W2, W3 = decode(genotype, input_size, hidden_size, output_size)

    reset_sim(data, init_qpos, init_qvel)
    positions = []

    for _ in range(steps):
        q = data.qpos.copy()
        raw_out = nn_forward(q, W1, W2, W3)
        scaled = (raw_out * 2.0 - 1.0) * HINGE_RANGE

        data.ctrl += scaled * delta
        data.ctrl = np.clip(data.ctrl, -HINGE_RANGE, HINGE_RANGE)

        if to_track:
            positions.append(to_track[0].xpos.copy())

        mujoco.mj_step(model, data)

    if not positions:
        return -np.inf, np.array([])

    # Fitness = final y-position (same as CMA-ES)
    final_y = positions[-1][1]
    fitness = final_y
    return float(fitness), np.array(positions)


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
def evolve_random(generations=GENERATIONS, pop_size=POP_SIZE, steps=ROLLOUT_STEPS):
    """
    Run the simple random evolutionary algorithm (elitism + mutation)."""
    model, data, to_track, init_qpos, init_qvel = build_world_and_model()
    genome_size = len(init_qpos)*HIDDEN_SIZE + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*model.nu

    population = initialize_population(genome_size, pop_size)
    fitness_history = []

    for gen in tqdm(range(generations), desc="Random EA Generations"):
        results = [rollout(ind, model, data, to_track, init_qpos, init_qvel, steps=steps)
                   for ind in population]
        fitnesses = [r[0] for r in results]

        population, gen_best, gen_best_score = reproduce(population, np.array(fitnesses))
        fitness_history.append(gen_best_score)
        # print(f"[Random EA] Gen {gen+1}/{generations} | Best Y: {gen_best_score:.4f} | Mean: {np.mean(fitnesses):.4f}")

    return population, fitness_history

