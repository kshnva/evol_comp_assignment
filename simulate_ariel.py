# Simulate a saved best genome in the Ariel + MuJoCo viewer

import numpy as np
import torch
import mujoco
from mujoco import viewer
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

from Mika_evotorch_CMAES import decode_genome, normalize_inputs, velocity_to_target, tanh, HIDDEN_SIZE, OUTPUT_SIZE


def run_best_genome(best_genome, activation: str = "tanh"):
    mujoco.set_mjcb_control(None)

    # Build world
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    W_in, W_rec, W_out = decode_genome(best_genome)
    h = np.zeros(HIDDEN_SIZE)
    ctrl_state = np.zeros(OUTPUT_SIZE)

    act_fn = tanh if activation == "tanh" else lambda x: 1.0 / (1.0 + np.exp(-x))

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


if __name__ == "__main__":
    import sys
    activation = sys.argv[1] if len(sys.argv) > 1 else "tanh"

    if activation == "tanh":
        genome = torch.load("results/best_tanh_genome.pt")
    elif activation == "sigmoid":
        genome = torch.load("results/best_sigmoid_genome.pt")
    else:
        raise ValueError("Activation must be 'tanh' or 'sigmoid'")

    run_best_genome(genome, activation=activation)
