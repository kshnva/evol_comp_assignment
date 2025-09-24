# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history
HISTORY = []  # list of np.ndarray of shape (3,) representing core body's XYZ at each timestep

# Controller hyperparams
GENOME_DIM = 32         # number of genome components used by NN
SIGMA_INIT = 0.1        # initial mutation rate (1 component)
HIDDEN = 128            # hidden layer size of NN
DELTA = 0.05            # delta multiplier for smoother physics
HINGE_LIMIT = np.pi/2   # actuator limit (radians)
DEVICE = torch.device("cpu")

#-----------------------------------------------------------------------------------------
####### NN CONTROLLER CLASS & FACTORY FUNCTION ########
#-----------------------------------------------------------------------------------------
class Controller(nn.Module):
    """NN that maps (joint angles + genome) -> actuator outputs."""
    def __init__(self, obs_dim: int, genome_dim: int, out_dim: int, hidden: int = HIDDEN):
        """
        obs_dim: int, number of input observations (len of data.qpos)
        genome_dim: int, number of genome components fed to NN
        out_dim: int, number of actuators (model.nu)
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + genome_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)

        # Weight initialization
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.1)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.1)
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, genome: torch.Tensor) -> torch.Tensor:
        """
        obs: torch.Tensor, shape (1, obs_dim)
        genome: torch.Tensor, shape (1, genome_dim)
        returns: torch.Tensor, shape (1, out_dim)
        """
        x = torch.cat([obs, genome], dim=-1)  # shape (1, obs_dim + genome_dim)
        x = F.relu(self.fc1(x))                # shape (1, hidden)
        x = F.relu(self.fc2(x))                # shape (1, hidden)
        out = torch.tanh(self.fc_out(x))       # shape (1, out_dim), values in [-1,1]
        return out


def build_observation(model: "mujoco.MjModel", data: "mujoco.MjData") -> np.ndarray:
    """
    Returns joint angles (qpos) as observation vector.
    Output: np.ndarray of shape (len(data.qpos),)
    """
    qpos = np.array(data.qpos).ravel()  # flatten to 1D
    return qpos.astype(np.float32)


def controller_step_factory(model: "mujoco.MjModel",
                            data: "mujoco.MjData",
                            *,
                            genome: tuple,
                            controller_net: Controller,
                            use_delta: bool = True,
                            to_track):
    """
    Creates a MuJoCo control callback.
    
    genome: tuple (nn_genome, sigma)
        - nn_genome: np.ndarray, shape (GENOME_DIM,), fed into NN
        - sigma: np.ndarray, shape (1,), used for mutation (not NN)
    to_track: list of bound geom objects to track positions
    """
    nn_genome = genome[0]  # extract only NN part
    genome_torch = torch.from_numpy(nn_genome.astype(np.float32)).unsqueeze(0).to(DEVICE)  
    # shape: (1, GENOME_DIM)

    def step(m: "mujoco.MjModel", d: "mujoco.MjData"):
        obs = build_observation(m, d)              # shape: (obs_dim,)
        obs_torch = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)  # shape: (1, obs_dim)

        # Run NN
        with torch.no_grad():
            action_t = controller_net(obs_torch, genome_torch)  # shape: (1, out_dim)
        action = action_t.squeeze(0).cpu().numpy()             # shape: (out_dim,)

        # Scale for smoothness
        delta_ctrl = action * DELTA * HINGE_LIMIT            # shape: (out_dim,)
        expected_nu = m.nu                                   # expected number of actuators

        # Handle mismatch between NN output and actuators
        if delta_ctrl.shape[0] != expected_nu:
            if delta_ctrl.shape[0] > expected_nu:
                delta_ctrl = delta_ctrl[:expected_nu]
            else:
                pad = np.zeros(expected_nu - delta_ctrl.shape[0], dtype=delta_ctrl.dtype)
                delta_ctrl = np.concatenate([delta_ctrl, pad])

        # Apply controls
        if use_delta:
            d.ctrl[:] = d.ctrl + delta_ctrl   # shape: (nu,)
        else:
            d.ctrl[:] = action * HINGE_LIMIT
        d.ctrl[:] = np.clip(d.ctrl, -HINGE_LIMIT, HINGE_LIMIT)

        # Track core position history
        # d.xpos[core_id]: shape (3,) -> XYZ in world frame
        HISTORY.append(to_track[0].xpos.copy())

    return step


# ------------------------ EA Components ------------------------ #



#-----------------------------------------------------------------------------------------
####### Parent Selection FUNCTION ########
#-----------------------------------------------------------------------------------------
def parent_selection_tournament(population: "Population", tournament_size: int = 3) -> "Population":
    """
    Tournament selection for parent tagging based on fitness.
    Each tournament selects the individual with the best fitness to be tagged for crossover.
    
    population: list of Individuals
    tournament_size: number of individuals per tournament
    """

    # Reset all parent selection tags
    for ind in population:
        ind.tags["ps"] = False

    # Run tournaments across the population
    num_individuals = len(population)
    for idx in range(0, num_individuals, tournament_size):
        # Grab a tournament slice (wrap-around if needed)
        tournament_inds = population[idx: idx + tournament_size]
        if len(tournament_inds) < tournament_size:
            # If last tournament is smaller, wrap around
            tournament_inds += population[:tournament_size - len(tournament_inds)]

        # Choose winner based on fitness
        if config.is_maximisation:
            winner = max(tournament_inds, key=lambda ind: ind.fitness)  # highest fitness wins
        else:
            winner = min(tournament_inds, key=lambda ind: ind.fitness)  # lowest fitness wins

        # Tag winner as selected parent
        winner.tags["ps"] = True

    return population



#-----------------------------------------------------------------------------------------
####### RECOMBINATION FUNCTION ########
#-----------------------------------------------------------------------------------------
def discrete_crossover(population: "Population") -> "Population": 
    """
    Performs discrete (coin-flip) crossover on a population of individuals.
    Assumes each individual has `genotype` as a tuple: (nn_genome: np.ndarray, sigma: np.ndarray)
    """

    # Select parents marked from parent_selection
    parents = [ind for ind in population if ind.tags.get("ps", False)]

    for idx in range(0, len(parents) - 1, 2):
        parent_i = parents[idx]
        parent_j = parents[idx + 1]

        # Unpack genomes
        nn_i, sigma_i = parent_i.genotype  # nn_i: shape (GENOME_DIM,), sigma_i: shape (1,)
        nn_j, sigma_j = parent_j.genotype

        # Generate random coin flips for each gene
        mask = np.random.randint(0, 2, size=nn_i.shape).astype(bool)  # True -> take from parent_i

        # Child NN genomes
        child_nn_1 = np.where(mask, nn_i, nn_j)
        child_nn_2 = np.where(mask, nn_j, nn_i)

        # Child sigma: coin flip for each child individually
        if random.random() < 0.5:
            child_sigma_1 = sigma_i.copy()
        else:
            child_sigma_1 = sigma_j.copy()

        if random.random() < 0.5:
            child_sigma_2 = sigma_i.copy()
        else:
            child_sigma_2 = sigma_j.copy()

        # Create child individuals
        child_1 = Individual()
        child_1.genotype = (child_nn_1, child_sigma_1)
        child_1.tags = {"mut": True}
        child_1.requires_eval = True

        child_2 = Individual()
        child_2.genotype = (child_nn_2, child_sigma_2)
        child_2.tags = {"mut": True}
        child_2.requires_eval = True

        # Add children to population
        population.extend([child_1, child_2])

    return population



def show_qpos_history(history: list):
    """
    history: list of np.ndarray, each shape (3,)
    Plots XY path of the tracked body.
    """
    pos_data = np.array(history)  # shape: (timesteps, 3)
    plt.figure(figsize=(10, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')  # XY
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
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world and gecko
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Find geoms to track
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]  # list of bound geom objects

    # Build controller
    obs_dim = len(data.qpos)  # number of joint positions
    nu = model.nu              # number of actuators
    controller = Controller(obs_dim=obs_dim, genome_dim=GENOME_DIM, out_dim=nu, hidden=HIDDEN).to(DEVICE)
    controller.eval()

    # Genome tuple: (nn_genome, sigma)
    nn_genome = np.random.uniform(-1, 1, size=GENOME_DIM).astype(np.float32)  # shape: (GENOME_DIM,)
    sigma = np.array([SIGMA_INIT], dtype=np.float32)                           # shape: (1,)
    genome = (nn_genome, sigma)

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
