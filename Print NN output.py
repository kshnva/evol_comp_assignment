# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history
HISTORY = []

# Controller hyperparams
GENOME_DIM = 32         # length of genome vector
SIGMA_INIT = 0.1        # initial mutation step size
HIDDEN = 128            # hidden layer size
DELTA = 0.05            # delta multiplier for smoother physics
HINGE_LIMIT = np.pi/2   # actuator limit
DEVICE = torch.device("cpu")


class Controller(nn.Module):
    """NN that maps (joint angles + genome) -> actuator outputs."""
    def __init__(self, obs_dim: int, genome_dim: int, out_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + genome_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)

        nn.init.kaiming_uniform_(self.fc1.weight, a=0.1)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.1)
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, genome: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, genome], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = torch.tanh(self.fc_out(x))  # [-1,1]
        return out


def build_observation(model: "mujoco.MjModel", data: "mujoco.MjData") -> np.ndarray:
    """Return joint angles (qpos) as observation."""
    qpos = np.array(data.qpos).ravel()
    return qpos.astype(np.float32)


def controller_step_factory(model: "mujoco.MjModel",
                            data: "mujoco.MjData",
                            *,
                            genome: np.ndarray,
                            controller_net: Controller,
                            use_delta: bool = True,
                            to_track):
    """Return a MuJoCo control callback using qpos + genome."""
    genome_torch = torch.from_numpy(genome.astype(np.float32)).unsqueeze(0).to(DEVICE)

    def step(m: "mujoco.MjModel", d: "mujoco.MjData"):
        obs = build_observation(m, d)
        obs_torch = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)

        # Run NN
        with torch.no_grad():
            action_t = controller_net(obs_torch, genome_torch)
        action = action_t.squeeze(0).cpu().numpy()

        # ---- PRINT NN output and joint angles every 10 timesteps ----
        print_interval = 1000  # steps
        step_num = int(d.time / m.opt.timestep)
        if step_num % print_interval == 0:
            print(f"Step {step_num}, Time {d.time:.2f}s")
            print(f"Joint angles: {obs}")
            print(f"NN output: {action}")
            print("-" * 40)

        # Scale and apply controls
        delta_ctrl = action * DELTA * HINGE_LIMIT
        if use_delta:
            d.ctrl[:] = d.ctrl + delta_ctrl
        else:
            d.ctrl[:] = action * HINGE_LIMIT

        d.ctrl[:] = np.clip(d.ctrl, -HINGE_LIMIT, HINGE_LIMIT)

        # Track core position history (optional)
        HISTORY.append(to_track[0].xpos.copy())

    return step


def show_qpos_history(history: list):
    pos_data = np.array(history)
    plt.figure(figsize=(10, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    plt.show()


def main():
    mujoco.set_mjcb_control(None)

    # Initialise world and gecko
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # Build controller
    obs_dim = len(data.qpos)
    nu = model.nu
    controller = Controller(obs_dim=obs_dim, genome_dim=GENOME_DIM, out_dim=nu, hidden=HIDDEN).to(DEVICE)
    controller.eval()

    # Example genome (later EA will supply this)
    genome = np.random.uniform(-1, 1, size=GENOME_DIM).astype(np.float32)

    # Set callback
    step_callback = controller_step_factory(
        model, data,
        genome=genome,
        controller_net=controller,
        use_delta=True,
        to_track=to_track
    )
    mujoco.set_mjcb_control(lambda m, d: step_callback(m, d))

    # Launch simulation viewer
    viewer.launch(model=model, data=data)

    # Optionally plot history
    show_qpos_history(HISTORY)


if __name__ == "__main__":
    main()
