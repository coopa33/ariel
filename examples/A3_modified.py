"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Tuple

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import os
from deap import base, tools
import random
from functools import partial
# import multiprocessing
from scoop import futures
from time import time
import pickle
from datetime import datetime
from dataclasses import field, dataclass, replace
from typing import List

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner, continue_simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from scripts_A3.creator_toolbox_prep import ensure_deap_types, register_factories
from scripts_A3.eval_tools import compute_brain_genome_size, nn_controller, decode_brain_genotype, decode_body_genotype, find_in_out_size
from scripts_A3.statistics_output import print_statistics
# from scripts_A3.file_management import save_generation, load_population_from_generation, load_generation_data, get_next_run_id

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

@dataclass
class EAConfig:
    """
        General configuration dataclass
    """
    # Seed
    rng_seed: int = 42
    # Simulation durations for checkpoints
    duration_flat: int = 15
    duration_rugged: int = 30
    duration_elevated: int = 55
    # Checkpoints
    checkpoint_rugged: list[float] = field(default_factory=lambda: [0.6, 0, 0.1])
    checkpoint_elevated: list[float] = field(default_factory=lambda: [2.4, 0, 0.1])
    # Possible starting positions
    start_normal: list[float] = field(default_factory=lambda: [-0.8, 0, 0.1])
    start_rugged: list[float] = field(default_factory=lambda: [1.6, 0, 0.1])
    # Simulation positions
    spawn_position: list[float] = field(init=False)
    target_position: list[float] = field(default_factory=lambda: [5, 0, 0.5])
    # Robot parameters
    num_of_modules: int = 30
    data: Path = field(init=False)
    # NDE of best body
    nde = None
    
    def __post_init__(self):
        self.spawn_position = self.start_normal.copy()
        # Data setup
        script_name = __file__.split("/")[-1][:-3]
        cwd = Path.cwd()
        self.data = cwd / "__data__" / script_name
        self.data.mkdir(parents = True, exist_ok=True)
        
    def create_reproducible_rng(self, context_seed: int = 0) -> np.random.Generator:
        """Create reproducible RNG with context-specific seed"""
        # Combine base seed with context for reproducible but varied sequences
        combined_seed = (self.rng_seed + context_seed) % (2**32)
        return np.random.default_rng(combined_seed)

@dataclass
class EABrainConfig:
    """Configuration for Brain Evolutionary Algorithm"""
    runs_brain: int = 1
    ngen_brain: int = 50
    pop_size_brain: int = 100
    cxpb_brain: float = 0.5
    mutpb_brain: float = 0.5
    elites_brain: int = 1
    # Adaptive parameters
    use_adaptive_params: bool = True
    adaptive_strategy: str = "generation_based"  # "generation_based" or "fitness_based"

@dataclass
class EABodyConfig:
    """Configuration for Body Evolutionary Algorithm"""
    runs_body: int = 1
    ngen_body: int = 2
    pop_size_body: int = 10
    cxpb_body: float = 0.5
    mutpb_body: float = 0.5
    elites_body: int = 1
    # Adaptive parameters
    use_adaptive_params: bool = True
    adaptive_strategy: str = "generation_based"  # "generation_based" or "fitness_based"


# --- RANDOM GENERATOR SETUP --- #
# SEED = 2
# RNG = np.random.default_rng(SEED)

# # --- DATA SETUP ---
# SCRIPT_NAME = __file__.split("/")[-1][:-3]
# CWD = Path.cwd()
# DATA = CWD / "__data__" / SCRIPT_NAME
# DATA.mkdir(exist_ok=True)

# # Global variables
# DURATION_FLAT = 15
# DURATION_RUGGED = 30
# DURATION_ELEVATED = 55
# START_NORMAL = [-0.8, 0, 0.1]
# START_RUGGED = [1.6, 0, 0.1]
# CHECKPOINT_RUGGED = [0.6, 0, 0.1] 
# CHECKPOINT_ELEVATED = [2.4, 0, 0.1]
# SPAWN_POS = START_NORMAL

# NUM_OF_MODULES = 30
# TARGET_POSITION = [5, 0, 0.5]

def save_generation(generation, pop_body_genotype, best_body, best_brain, run_id = 0, sim_config = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_dir = sim_config.data / f"run_{run_id}" / f"generation_{generation:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    with open(gen_dir / "body_population.pkl", "wb") as f:
        pickle.dump(pop_body_genotype, f)
    
    best_data = {
        "generation": generation,
        "timestamp": timestamp,
        "best_body_genotype": list(best_body),  # Convert to list for serialization
        "best_brain_genotype": list(best_brain) if best_brain is not None else None,
        "body_fitness": best_body.fitness.values[0] if best_body.fitness.valid else None,
        "brain_fitness": best_brain.fitness.values[0] if hasattr(best_brain, 'fitness') and best_brain.fitness.valid else None,
        "nde": sim_config.nde
    }
    
    with open(gen_dir / "best_performers.pkl", "wb") as f:
        pickle.dump(best_data, f)
    
    print(f"Saved generation {generation} data to {gen_dir}")
    
def load_generation_data(generation, run_id, sim_config = EAConfig()):
    """Load saved generation data"""
    gen_dir = sim_config.data / f"run_{run_id}" / f"generation_{generation:03d}"
    
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generation {generation} data not found at {gen_dir}")
    
    with open(gen_dir / "body_population.pkl", "rb") as f:
        population = pickle.load(f)
    
    with open(gen_dir / "best_performers.pkl", "rb") as f:
        best_data = pickle.load(f)
    
    return population, best_data
def load_population_from_generation(sim_config, generation, run_id = 0):
    """Load population from a specific generation"""
    gen_dir = sim_config.data / f"run_{run_id}" / f"generation_{generation:03d}"
    
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generation {generation} data not found at {gen_dir}")
    
    # Load population
    with open(gen_dir / "body_population.pkl", "rb") as f:
        population = pickle.load(f)
    
    # Load best data
    with open(gen_dir / "best_performers.pkl", "rb") as f:
        best_data = pickle.load(f)
    
    # Restore NDE to config
    if "nde" in best_data and best_data["nde"] is not None:
        sim_config.nde = best_data["nde"]
    
    print(f"Resumed from generation {generation} with {len(population)} individuals")
    print(f"Best fitness from that generation: {best_data.get('body_fitness', 'Unknown')}")
    
    return population, best_data

def get_next_run_id(sim_config: EAConfig) -> int:
    """Find the next available run ID to avoid overwriting"""
    run_id = 0
    while (sim_config.data / f"run_{run_id}").exists():
        run_id += 1
    return run_id

def find_latest_generation(sim_config, run_id):
    run_dir = sim_config.data / f"run_{run_id}"
    if not run_dir.exists():
        return -1
    latest_gen = -1
    
    # Look for generation directories
    for gen_dir in run_dir.iterdir():
        if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
            try:
                gen_num = int(gen_dir.name.split("_")[1])
                if (gen_dir / "body_population.pkl").exists() and (gen_dir / "best_performers.pkl").exists():
                    latest_gen = max(latest_gen, gen_num)
            except (ValueError, IndexError):
                continue
    return latest_gen

def save_unique_png(fig, path = "__data__/", ext = ".png"):
    """Function to save plt figures with unique filenames. To prevent
       overwriting existing plots.

    Args:
        fig (matplotlib.figure): Figure to save
        path (str, optional): The path of the directory where the figure 
                              is to be saved. Defaults to "__data__/".

    Returns:
        _type_: The path of the saved file
    """
    i = 0
    filename = f"{path}position{ext}"
    while os.path.exists(filename):
        i += 1
        filename = f"{path}position_{i}{ext}"
    fig.savefig(filename)
    return filename

def fitness_function(history: list[float], sim_config) -> float:
    """A fitness function that maximizes the inverse of distance to target"""
    xt, yt, zt = sim_config.target_position
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return - cartesian_distance

def fitness_function_without_landing(history: list[float], sim_config) -> float:
    """Same as the above fitness function, but ignores the distance traveled (4 seconds)"""
    xt, yt, zt = sim_config.target_position
    
    # Determine positions after "landing" and at simulation end
    landing_frame = 4
    xi, yi, zi = history[landing_frame]
    xf, yf, zf = history[-1]
    
    initial_distance_to_target = np.sqrt((xt - xi) ** 2 + (yt - yi) ** 2 + (zt - zi) ** 2)
    final_distance_to_target = np.sqrt((xt - xf) ** 2 + (yt - yf) ** 2 + (zt - zf) ** 2)

    # Calculate distance traveled towards target, which is to be maximized
    distance_traveled_towards_target = initial_distance_to_target - final_distance_to_target
    return distance_traveled_towards_target

def diff_distance(history):
    """Calculate the difference in distance to target between two frames(2 seconds until end)"""
    xc, yc = history[2][:2]
    xt, yt = history[-1][:2]
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2
    )
    return cartesian_distance

def passed_checkpoint(checkpoint, history):
    """Check if the robot has passed a certain checkpoint"""
    xc = history[-1][0]
    xt = checkpoint[0]
    return (xc >= xt)
    
    


def show_xpos_history(history: list[float], sim_config) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(sim_config.data / "background.png")
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, sim_config.spawn_position[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    save_unique_png(fig)
    

    
def rw_controller(
    model: mj.MjModel,
    data: mj.MjData,
    matrices = None, 
    sim_config = None
) -> npt.NDArray[np.float64]:
    """Random walk controller"""
    
    # Ensure seed is safe for parallelization
    if sim_config is None:
        sim_config = EAConfig()
    process_id = os.getpid()
    context_seed = int(time()) % 1000 + process_id
    rng = sim_config.create_reproducible_rng(context_seed)
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = rng.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = rng.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = rng.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi

    
def experiment(
    robot: Any,
    controller: Controller,
    matrices,
    sim_config,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation"""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=sim_config.spawn_position.copy())

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = [matrices, sim_config]  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )

            # Continue simulation for robots starting from normal spawn, if they passed checkpoints
            if sim_config.spawn_position == sim_config.start_normal and passed_checkpoint(sim_config.checkpoint_rugged, controller.tracker.history["xpos"][0]):
                console.log("Passed checkpoint, continue simulation")
                continue_simple_runner(
                    model,
                    data,
                    duration = sim_config.duration_rugged
                )

            if sim_config.spawn_position == sim_config.start_normal and passed_checkpoint(sim_config.checkpoint_rugged, controller.tracker.history["xpos"][0]):
                console.log("Passed checkpoint, continue simulation")
                continue_simple_runner(
                    model,
                    data,
                    duration = sim_config.duration_elevated
                )

        case "frame":
            # Render a single frame (for debugging)
            save_path = str(sim_config.data / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(sim_config.data / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #
    
def create_robot_graph(body_genotype, sim_config, nde = None):
    """Create a robot graph from a body genotype using NDE and HPD.
       If an NDE instance is provided, it will be reused."""
    if nde is None:
        body_genotype = decode_body_genotype(genotype = body_genotype, genotype_size=64)
        # Input body genotype
        nde = NeuralDevelopmentalEncoding(number_of_modules=sim_config.num_of_modules)
        p_matrices = nde.forward(body_genotype)

        # Decode the high-probability graph
        hpd = HighProbabilityDecoder(sim_config.num_of_modules)
        robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1],
            p_matrices[2],
        )
        
        # ? ------------------------------------------------------------------ #
        # Save the graph to a file
        save_graph_as_json(
            robot_graph,
            sim_config.data / "robot_graph.json",
        )
    else:
        body_genotype = decode_body_genotype(genotype = body_genotype, genotype_size=64)
        p_matrices = nde.forward(body_genotype)

        # Decode the high-probability graph
        hpd = HighProbabilityDecoder(sim_config.num_of_modules)
        robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1],
            p_matrices[2],
        )
        
        # ? ------------------------------------------------------------------ #
        # Save the graph to a file
        save_graph_as_json(
            robot_graph,
            sim_config.data / "robot_graph.json",
        )
    return robot_graph

def is_viable_body(body_genotype, sim_config, gate_time = 6.0, delta_min = 0.05, mode = "simple", max_fps = None):
    robot_graph = create_robot_graph(body_genotype=body_genotype, sim_config = sim_config)
    robot_spec = construct_mjspec_from_graph(robot_graph)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(
        controller_callback_function=rw_controller,
        tracker = tracker
    )
    experiment(robot_spec, controller=ctrl, matrices = None, sim_config = sim_config, duration = gate_time, mode = mode)
    xpos = tracker.history["xpos"][0]
    disp = diff_distance(xpos)
    vel_est = disp/max(gate_time, 1e-6)
    
    return (disp >= delta_min) or (vel_est >= delta_min / gate_time)

def make_viable_body(base_body_generator, sim_config, delta = 0.1, mode = "simple"):
    g = base_body_generator()
    while not is_viable_body(g, sim_config = sim_config, gate_time= 5.0, delta_min=delta, mode=mode):
        g = base_body_generator()
    return g

def evaluate_robot(
    brain_genotype,
    robot_graph,
    controller_func,
    network_specs,
    sim_config,
    experiment_mode = "simple",
    initial_duration = 15
    ) -> Tuple[float, ] :
    """
    Evaluate a single robot fitness based on its performance in the experiment.

    Args:
        brain_genotype (list): Individual brain genotype
        robot_graph (Any): The robot specs (body phenotype)
        controller_func (Callable): The controller function to use
        network_specs (dict[str, int]): A dictionary including the following keys - 'input_size' 'output_size' 'hidden_size' 'no_hidden_layers'
        experiment_mode (str): Rendering/simulation mode options. Defaults to "simple".

    Returns:
        Tuple[float, ]: Fitness score
    """
    # Construct robot specs, tracker, and controller
    robot_spec = construct_mjspec_from_graph(robot_graph)
    
    # Decode genotype to weight matrices
    w_matrices = decode_brain_genotype(
        brain_genotype = brain_genotype,
        network_specs= network_specs)

    tracker = Tracker(
        mujoco_obj_to_find =    mj.mjtObj.mjOBJ_GEOM,
        name_to_bind =          "core"
)
    # Run experiment
    ctrl = Controller(
        controller_callback_function=   controller_func,
        tracker =                       tracker
    )
    experiment(
        robot = robot_spec,
        controller = ctrl,
        matrices = w_matrices,
        sim_config = sim_config,
        duration = initial_duration,
        mode = experiment_mode
    )
    # Add this debug print in evaluate_robot after experiment runs:
    final_pos = tracker.history["xpos"][0][-1]
    initial_pos = tracker.history["xpos"][0][0] 
    distance_moved = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
    print(f"Robot moved: {distance_moved:.4f} units")
    # show_xpos_history(tracker.history["xpos"][0], sim_config = sim_config)
    # In evaluate_robot, before returning fitness:
    print(f"Final position: {tracker.history['xpos'][0][-1]}")
    # Return fitness
    return (fitness_function(tracker.history["xpos"][0], sim_config = sim_config), )

def whole_arithmetic_recomb(ind1, ind2, alpha):
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        cross_value_1 = alpha * x2 + (1 - alpha) * x1
        cross_value_2 = (1-alpha) * x2 + alpha * x1
        ind1[i] = cross_value_1
        ind2[i] = cross_value_2

    return ind1, ind2    

def calculate_adaptive_brain_params(current_gen, max_gen, config):
    """
    Calculate adaptive crossover and mutation probabilities for brain evolution.
    
    Strategy: Start with higher mutation for exploration, gradually increase crossover
    for exploitation as neural networks benefit from combining successful patterns.
    """
    if max_gen <= 1:
        return config.cxpb_brain, config.mutpb_brain
    
    # Progress ratio from 0 to 1
    progress = current_gen / (max_gen - 1)
    
    # Adaptive crossover: increase from base to higher values
    adaptive_cxpb = config.cxpb_brain + (0.8 - config.cxpb_brain) * progress
    
    # Adaptive mutation: decrease from base to lower values  
    adaptive_mutpb = config.mutpb_brain * (1 - 0.6 * progress)
    
    # Ensure values stay within reasonable bounds
    adaptive_cxpb = max(0.1, min(0.9, adaptive_cxpb))
    adaptive_mutpb = max(0.1, min(0.7, adaptive_mutpb))
    
    return adaptive_cxpb, adaptive_mutpb

def calculate_adaptive_body_params(current_gen, max_gen, config):
    """
    Calculate adaptive crossover and mutation probabilities for body evolution.
    
    Strategy: Body morphology needs balanced exploration/exploitation throughout,
    but slightly favor mutation early for discovering novel structures.
    """
    if max_gen <= 1:
        return config.cxpb_body, config.mutpb_body
    
    # Progress ratio from 0 to 1
    progress = current_gen / (max_gen - 1)
    
    # Adaptive crossover: gradual increase but less aggressive than brain
    adaptive_cxpb = config.cxpb_body + (0.7 - config.cxpb_body) * progress
    
    # Adaptive mutation: gradual decrease but maintain some exploration
    adaptive_mutpb = config.mutpb_body * (1 - 0.4 * progress)
    
    # Ensure values stay within reasonable bounds
    adaptive_cxpb = max(0.2, min(0.8, adaptive_cxpb))
    adaptive_mutpb = max(0.2, min(0.6, adaptive_mutpb))
    
    return adaptive_cxpb, adaptive_mutpb

def calculate_fitness_based_adaptive_params(population, base_cxpb, base_mutpb):
    """
    Alternative adaptive strategy based on population diversity and fitness variance.
    High diversity = lower crossover, higher mutation
    Low diversity = higher crossover, lower mutation
    """
    if len(population) < 2:
        return base_cxpb, base_mutpb
    
    # Calculate fitness variance as diversity measure
    fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
    if len(fitnesses) < 2:
        return base_cxpb, base_mutpb
    
    fitness_variance = np.var(fitnesses)
    fitness_mean = np.mean(fitnesses)
    
    # Normalize variance by mean to get relative diversity
    if fitness_mean > 0:
        diversity_ratio = fitness_variance / abs(fitness_mean)
    else:
        diversity_ratio = 1.0
    
    # High diversity (>0.1) = increase mutation, decrease crossover
    # Low diversity (<0.05) = increase crossover, decrease mutation
    if diversity_ratio > 0.1:
        adaptive_cxpb = base_cxpb * 0.8
        adaptive_mutpb = base_mutpb * 1.3
    elif diversity_ratio < 0.05:
        adaptive_cxpb = base_cxpb * 1.2
        adaptive_mutpb = base_mutpb * 0.7
    else:
        adaptive_cxpb = base_cxpb
        adaptive_mutpb = base_mutpb
    
    # Ensure bounds
    adaptive_cxpb = max(0.1, min(0.9, adaptive_cxpb))
    adaptive_mutpb = max(0.1, min(0.8, adaptive_mutpb))
    
    return adaptive_cxpb, adaptive_mutpb

    return len(identical_pairs)

def demo_adaptive_parameters():
    """Demonstrate how adaptive parameters change over generations"""
    print("=== Adaptive Parameter Demonstration ===")
    
    # Create sample configs
    brain_config = EABrainConfig(
        ngen_brain=5,
        cxpb_brain=0.5,
        mutpb_brain=0.5,
        use_adaptive_params=True
    )
    
    body_config = EABodyConfig(
        ngen_body=4,
        cxpb_body=0.5,
        mutpb_body=0.5,
        use_adaptive_params=True
    )
    
    print("\nBrain Evolution Adaptive Parameters:")
    print("Gen | Crossover | Mutation | Strategy")
    print("----|-----------|----------|----------")
    for g in range(brain_config.ngen_brain):
        cxpb, mutpb = calculate_adaptive_brain_params(g, brain_config.ngen_brain, brain_config)
        print(f" {g:2d} |    {cxpb:.3f}  |   {mutpb:.3f}  | Exploration→Exploitation")
    
    print("\nBody Evolution Adaptive Parameters:")
    print("Gen | Crossover | Mutation | Strategy")
    print("----|-----------|----------|----------")
    for g in range(body_config.ngen_body):
        cxpb, mutpb = calculate_adaptive_body_params(g, body_config.ngen_body, body_config)
        print(f" {g:2d} |    {cxpb:.3f}  |   {mutpb:.3f}  | Balanced exploration")
    
    print("\nKey Benefits:")
    print("1. Brain: Starts with exploration (high mutation) → ends with exploitation (high crossover)")
    print("2. Body: More conservative adaptation to preserve viable morphologies")
    print("3. Automatic parameter tuning based on generation progress")
    print("4. Can switch to fitness-based adaptation for population diversity control")

def debug_population_diversity(population):
    """Debug function to check for truly identical individuals"""
    identical_pairs = []
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            ind1, ind2 = population[i], population[j]
            
            # Check if truly identical
            if len(ind1) == len(ind2):
                # Element-wise comparison with very small tolerance
                differences = [abs(x1 - x2) for x1, x2 in zip(ind1, ind2)]
                max_diff = max(differences) if differences else 0
                
                if max_diff < 1e-10:  # Truly identical
                    identical_pairs.append((i, j, max_diff))
                elif max_diff < 1e-6:  # Very similar
                    print(f"Very similar individuals {i} and {j}: max_diff = {max_diff}")
    
    if identical_pairs:
        print(f"Found {len(identical_pairs)} truly identical pairs: {identical_pairs}")
    else:
        print("No truly identical individuals found in initial population")
    
    return len(identical_pairs)







def EA_brain(body_genotype, ea_brain_config, sim_config, ind_type, mode):
    """Entry point."""
    # ? ------------------------------------------------------------------ #
    # Create brain toolbox
    toolbox_brain = base.Toolbox()

    # Input body genotype
    nde = NeuralDevelopmentalEncoding(number_of_modules=sim_config.num_of_modules)
    p_matrices = nde.forward(body_genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(sim_config.num_of_modules)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )
    
    
    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    save_graph_as_json(
        robot_graph,
        sim_config.data / "robot_graph.json",
    )

    # ? ------------------------------------------------------------------ #
    ### === Preparation for EA Brain ===
    
    # Initialization distribution
    func_gauss = partial(random.uniform, -1, 1)
    
    # Define the network specifications
    input_size, output_size = find_in_out_size(robot_graph, sim_config.spawn_position.copy())
    network_specs = {
        "input_size" :          input_size,
        "output_size" :         output_size,
        "hidden_size" :         128,
        "no_hidden_layers" :    3
    }    
    # Calculate the size of a brain genotype, based on network specs
    ind_size = compute_brain_genome_size(network_specs)
    
    
    # Register in toolbox 
    register_factories(
        t=              toolbox_brain,
        ind_type=       ind_type,
        init_func=      func_gauss,
        t_attr_name=    "attr_float",
        t_ind_name=     "create_brain_genome",
        t_pop_name=     "create_brain_genome_pop",
        no_alleles=     ind_size,
    )
    toolbox_brain.register(
        "EvaluateRobot",
        evaluate_robot,
        robot_graph =       robot_graph, # This is the phenotyp expression of the body genotype.
        controller_func =   nn_controller,
        experiment_mode =   mode,
        sim_config =        sim_config,
        network_specs =     network_specs,
        initial_duration =   15
    )
    toolbox_brain.register(
        "ParentSelectBrain",
        tools.selTournament,
        tournsize = 3
    )
    toolbox_brain.register(
        "SurvivalSelectBrain",
        tools.selBest,
        k = ea_brain_config.pop_size_brain
    )
    toolbox_brain.register(
        "MateBrain",
        whole_arithmetic_recomb,
        alpha = 0.4
    )
    toolbox_brain.register(
        "MutateBrain",
        tools.mutGaussian,
        mu = 0.0,
        sigma = 0.3,
        indpb = 0.3
    )
    
    # ? ------------------------------------------------------------------ #
    ### === Evolutionary Algorithm for Brain ===
    
    champions = []
    for r in range(ea_brain_config.runs_brain):
        # Create population
        pop_brain_genotype = toolbox_brain.create_brain_genome_pop(n = ea_brain_config.pop_size_brain)
        # debug_population_diversity(pop_brain_genotype)
        # Assign each individual a fitness value
        f_brain_genotype = list(map(toolbox_brain.EvaluateRobot, pop_brain_genotype))
        for ind, f in zip(pop_brain_genotype, f_brain_genotype):
            ind.fitness.values = f
        print("First gen stats")
        print_statistics(pop_brain_genotype)
        
        # Go through generations
        for g in range(ea_brain_config.ngen_brain):
            # Calculate adaptive parameters for this generation
            if ea_brain_config.use_adaptive_params:
                if ea_brain_config.adaptive_strategy == "generation_based":
                    adaptive_cxpb, adaptive_mutpb = calculate_adaptive_brain_params(
                        current_gen=g, 
                        max_gen=ea_brain_config.ngen_brain, 
                        config=ea_brain_config
                    )
                elif ea_brain_config.adaptive_strategy == "fitness_based":
                    adaptive_cxpb, adaptive_mutpb = calculate_fitness_based_adaptive_params(
                        population=pop_brain_genotype,
                        base_cxpb=ea_brain_config.cxpb_brain,
                        base_mutpb=ea_brain_config.mutpb_brain
                    )
                else:
                    adaptive_cxpb, adaptive_mutpb = ea_brain_config.cxpb_brain, ea_brain_config.mutpb_brain
            else:
                adaptive_cxpb, adaptive_mutpb = ea_brain_config.cxpb_brain, ea_brain_config.mutpb_brain
            
            print(f"Brain Gen {g}: adaptive_cxpb={adaptive_cxpb:.3f}, adaptive_mutpb={adaptive_mutpb:.3f}")
            
            offspring = toolbox_brain.ParentSelectBrain(pop_brain_genotype, k = ea_brain_config.pop_size_brain)
            offspring = list(map(toolbox_brain.clone, offspring))
            random.shuffle(offspring)
            
            # Apply variation operators with adaptive probabilities
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < adaptive_cxpb:
                    toolbox_brain.MateBrain(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < adaptive_mutpb:
                    toolbox_brain.MutateBrain(mutant)
                    del mutant.fitness.values
                            
            # Evaluate offspring fitnesses of individuals which had genotypes changed by mating and mutating
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_fitnesses = list(toolbox_brain.map(toolbox_brain.EvaluateRobot, invalid_ind))
            for ind, fit in zip(invalid_ind, invalid_fitnesses):
                ind.fitness.values = fit
            # Survival selection + Elitism
            pop_brain_genotype[:] = toolbox_brain.SurvivalSelectBrain(offspring + tools.selBest(pop_brain_genotype, k = ea_brain_config.elites_brain))
            print("Brain EA")
            print_statistics(pop_brain_genotype)
            
        champions.append(tools.selBest(pop_brain_genotype, k = 1)[0])
        
    best_brain = tools.selBest(champions, k = 1)[0]
    return best_brain.fitness.values, best_brain, nde
    

def EA_body(
    ea_brain_config,
    ea_body_config,
    sim_config,
    resume_from_generation = -1,
    resume_run_id = 0
    ):

    
    """
    TO-DO:
    - Make body genotype not exceed allowed bounds for probabilites DONE
    - Refine viable body selector to check for movement after "landing" DONE
    - Create a function that can determine the positions (x, y) of the terrain boundaries DONE
    - Create a function that can check whether the robot has passed the terrain boundary in the simulation time DONE
    - Implement that the experiment continues if the checkpoint is passed, and otherwise stops with the fitness being the distance to target at the end of experiment. DONE
    """
    # Define genotype size for body
    body_genotype_size = 3*64
    
    # Ensure the deap types are in creator
    _, ind_type = ensure_deap_types()
    
    # Create body toolbox
    toolbox_body = base.Toolbox()
    
    # Multiprocessing for body EA
    # pool = multiprocessing.Pool(processes=os.cpu_count() - 2) 
    toolbox_body.register("map", futures.map) 
    toolbox_body.register(
        "attr_float", 
        random.random
    )
    toolbox_body.register(
        "individual", tools.initRepeat, ind_type, 
        toolbox_body.attr_float, n = body_genotype_size
    )
    toolbox_body.register(
        "make_viable_body",
        make_viable_body,
        sim_config = sim_config,
        base_body_generator = toolbox_body.individual,
        delta = 0.2
    )
    toolbox_body.register(
        "population",
        tools.initRepeat,
        list,
        toolbox_body.make_viable_body
    )

    toolbox_body.register(
        "EvaluateRobotBody",
        EA_brain,
        ea_brain_config = ea_brain_config,
        sim_config = sim_config,
        ind_type = ind_type,
        mode = "simple"
    )
    toolbox_body.register(
        "ParentSelectBody",
        tools.selTournament,
        tournsize = 3
    )
    toolbox_body.register(
        "SurvivalSelectBody",
        tools.selBest,
        k = ea_body_config.pop_size_body
    )
    toolbox_body.register(
        "MateBody",
        whole_arithmetic_recomb,
        alpha = 0.4
    )
    toolbox_body.register(
        "MutateBody",
        tools.mutGaussian,
        mu = 0.0,
        sigma = 0.15,
        indpb = 0.15
    )
    ### --- EA Body ---
    champions = []
    for r in range(ea_body_config.runs_body):
        
        if resume_from_generation >= 0:
            try:
                pop_body_genotype, best_data = load_population_from_generation(
                    sim_config = sim_config,
                    generation = resume_from_generation,
                    run_id = resume_run_id
                )
                start_generation = resume_from_generation + 1
                print(f"Resuming run {resume_run_id} from generation {start_generation}")
                current_run_id = resume_run_id
            except FileNotFoundError as e:
                print(f"Reums failed: {e}")
                print("Starting new run instead")
                pop_body_genotype = toolbox_body.population(n = ea_body_config.pop_size_body)
                start_generation = 0
                current_run_id = r
                
                # Evaluate initial population
                f_body_genotype = list(toolbox_body.map(toolbox_body.EvaluateRobotBody, [decode_body_genotype(ind, 64) for ind in pop_body_genotype]))
                for ind, (f, best_brain, nde) in zip(pop_body_genotype, f_body_genotype):
                    ind.fitness.values = f
                    ind.best_brain = best_brain
                    ind.nde = nde
        else:
            # New run
            pop_body_genotype = toolbox_body.population(n = ea_body_config.pop_size_body)
            start_generation = 0
            current_run_id = r
    
            f_body_genotype = list(toolbox_body.map(toolbox_body.EvaluateRobotBody, [decode_body_genotype(ind, 64) for ind in pop_body_genotype]))
            for ind, (f, best_brain, nde) in zip(pop_body_genotype, f_body_genotype):
                ind.fitness.values = f
                ind.best_brain = best_brain
                ind.nde = nde
            

        # Go through generations
        end_generation = start_generation + ea_body_config.ngen_body
        for g in range(start_generation, end_generation):
            # Calculate adaptive parameters for this generation
            generation_within_run = g - start_generation
            if ea_body_config.use_adaptive_params:
                if ea_body_config.adaptive_strategy == "generation_based":
                    adaptive_cxpb, adaptive_mutpb = calculate_adaptive_body_params(
                        current_gen=generation_within_run,
                        max_gen=ea_body_config.ngen_body,
                        config=ea_body_config
                    )
                elif ea_body_config.adaptive_strategy == "fitness_based":
                    adaptive_cxpb, adaptive_mutpb = calculate_fitness_based_adaptive_params(
                        population=pop_body_genotype,
                        base_cxpb=ea_body_config.cxpb_body,
                        base_mutpb=ea_body_config.mutpb_body
                    )
                else:
                    adaptive_cxpb, adaptive_mutpb = ea_body_config.cxpb_body, ea_body_config.mutpb_body
            else:
                adaptive_cxpb, adaptive_mutpb = ea_body_config.cxpb_body, ea_body_config.mutpb_body
            
            print(f"Body Gen {g}: adaptive_cxpb={adaptive_cxpb:.3f}, adaptive_mutpb={adaptive_mutpb:.3f}")
            
            offspring = toolbox_body.ParentSelectBody(pop_body_genotype, k = ea_body_config.pop_size_body)
            offspring = list(toolbox_body.map(toolbox_body.clone, offspring))
            random.shuffle(offspring)
            
            # Apply variation operators with adaptive probabilities
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < adaptive_cxpb:
                    toolbox_body.MateBody(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    # Delete non-deap attribute safely
                    if hasattr(child1, 'best_brain'):
                        del child1.best_brain
                    if hasattr(child2, 'best_brain'):
                        del child2.best_brain
                    if hasattr(child1, 'nde'):
                        del child1.nde
                    if hasattr(child2, 'nde'):
                        del child2.nde
                        
            for mutant in offspring:
                if random.random() < adaptive_mutpb:
                    toolbox_body.MutateBody(mutant)
                    del mutant.fitness.values
                    if hasattr(mutant, 'best_brain'):
                        del mutant.best_brain
                    if hasattr(mutant, 'nde'):
                        del mutant.nde

            # Evaluate offspring fitnesses of individuals which had genotypes changed by mating and mutating
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                for i in range(len(ind)):
                    if ind[i] < -1.0:
                        ind[i] = -1.0
                    elif ind[i] > 1.0:
                        ind[i] = 1.0         
            invalid_fitnesses = list(toolbox_body.map(toolbox_body.EvaluateRobotBody, [decode_body_genotype(ind, 64) for ind in invalid_ind]))

            for ind, (f, best_brain, nde) in zip(invalid_ind, invalid_fitnesses):
                ind.fitness.values = f
                ind.best_brain = best_brain
                ind.nde = nde
                
            # Survival selection + Elitism
            pop_body_genotype[:] = toolbox_body.SurvivalSelectBody(offspring + tools.selBest(pop_body_genotype, k = ea_body_config.elites_body))
            
            # Save generation data
            best_body = tools.selBest(pop_body_genotype, k = 1)[0]
            best_brain = best_body.best_brain
            sim_config.nde = best_body.nde
            save_generation(
                g, 
                pop_body_genotype, 
                best_body, 
                best_brain, 
                run_id = str(current_run_id), 
                sim_config= sim_config)
            
            # Print generation statistics
            print("Body EA")
            print_statistics(pop_body_genotype)
            
        # Only resume from saved generation for the first run
        resume_from_generation = -1
        champions.append(tools.selBest(pop_body_genotype, k = 1)[0])
    
    return tools.selBest(champions, k = 1)[0]


def main(
    sim_config,
    run_ea = True, 
    read_pop = False, 
    generation = 0, 
    run_id = 0,
    auto_resume = True,
    force_new_run = False):
        

    sim_config = replace(sim_config)
    ea_brain_config = EABrainConfig(
        runs_brain = 3,
        ngen_brain = 3,
        pop_size_brain = 10,
        cxpb_brain = 0.5,
        mutpb_brain = 0.5,
        elites_brain = 1,
        use_adaptive_params = True,
        adaptive_strategy = "generation_based"
    )
    ea_body_config = EABodyConfig(
        runs_body=2,
        ngen_body=2,
        pop_size_body=10,
        cxpb_body=0.5,
        mutpb_body=0.5,
        elites_body=1,
        use_adaptive_params = True,
        adaptive_strategy = "generation_based"
    )
    
    # Set seeds for reproducibility
    random.seed(sim_config.rng_seed)
    np.random.seed(sim_config.rng_seed)
    
    if run_ea:
        resume_gen = -1
        resume_run = 0
        
        if auto_resume and not force_new_run:
            latest_gen = find_latest_generation(sim_config = sim_config, run_id = 0)
            print(latest_gen)
            if latest_gen >= 0:
                print(f"Found previous run with data up to generation {latest_gen}")
                response = input(f"Resume from generation {latest_gen}? (y/n): ")
                if response == "y":
                    resume_gen = latest_gen
                    resume_run = 0
                else:
                    new_run_id = get_next_run_id(sim_config)
                    print(f"Starting new run with ID: {new_run_id}")
                    resume_run = new_run_id
            else:
                print("No previous run found, starting new run")                
        start = time()
        best_robot = EA_body(
            ea_brain_config=ea_brain_config,
            ea_body_config=ea_body_config,
            sim_config=sim_config,
            resume_from_generation=resume_gen,
            resume_run_id=resume_run
        )
        end = time()
        print(f"Elapsed time: {end - start:.2f} seconds")
        print(f"Best fitness: {best_robot.fitness.values[0]:.6f}")
        
    if read_pop:
        pop_body_genotype, best_data = load_generation_data(generation=1, run_id=0, sim_config=sim_config)
        best_robot = best_data["best_body_genotype"]
        best_brain = best_data["best_brain_genotype"]
        best_nde = best_data["nde"]
        
        print(f"Loaded best fitness: {best_data['body_fitness']:.6f}")
        robot_graph = create_robot_graph(best_robot, sim_config = sim_config, nde= best_nde)
        input_size, output_size = find_in_out_size(robot_graph, sim_config.spawn_position.copy())
        evaluate_robot(
            brain_genotype = best_brain,
            robot_graph = robot_graph,
            controller_func = nn_controller,
            network_specs = {
                "input_size" :          input_size,
                "output_size" :         output_size,
                "hidden_size" :         128,
                "no_hidden_layers" :    3
            },
            sim_config = sim_config,
            experiment_mode = "launcher",
            initial_duration = 60
        )
         
    # ? ------------------------------------------------------------------ #
    
if __name__ == "__main__":
    # Demonstrate adaptive parameters
    demo_adaptive_parameters()
    
    # To automatically resume from last checkpoint
    sim_config = EAConfig(rng_seed=42)
    SIMULATE = False
    if SIMULATE:
        main(sim_config = sim_config, run_ea = True, read_pop = False, auto_resume=True, run_id=0)

        #To force a new run (won't overwrite existing runs)
        # main(run_ea = True, read_pop = False, auto_resume=False, force_new_run=True, sim_config=sim_config)
    
        # To read and render a specific generation
        # main(run_ea = False, read_pop = True, generation = 9, run_id = 0)
    
    # To load a specific generation to analyse:
    RENDER_GEN = True
    if RENDER_GEN:
        pop, best_data = load_population_from_generation(
            sim_config = sim_config,
            generation = 59,
            run_id = 0
        )
        best_robot = best_data["best_body_genotype"]
        best_brain = best_data["best_brain_genotype"]
        best_nde = best_data["nde"]
        robot_graph = create_robot_graph(best_robot, sim_config = sim_config, nde= best_nde)
        input_size, output_size = find_in_out_size(robot_graph, sim_config.spawn_position.copy())
        print_statistics(pop)
        evaluate_robot(
            brain_genotype = best_brain,
            robot_graph = robot_graph,
            controller_func = nn_controller,
            network_specs = {
                "input_size" :          input_size,
                "output_size" :         output_size,
                "hidden_size" :         128,
                "no_hidden_layers" :    3
            },
            sim_config = sim_config,
            experiment_mode = "launcher",
            initial_duration = 120
        )
    
    
    
    
    
        
        


    
    
