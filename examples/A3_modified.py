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
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from scripts_A3.creator_toolbox_prep import ensure_deap_types, register_factories
from scripts_A3.eval_tools import compute_brain_genome_size, nn_controller, decode_brain_genotype, find_in_out_size

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

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

def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def show_xpos_history(history: list[float]) -> None:
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
    save_path = str(DATA / "background.png")
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
    ym0, ymc = 0, SPAWN_POS[0]

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
    
def experiment(
    robot: Any,
    controller: Controller,
    matrices,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

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
    args: list[Any] = [matrices]  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
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
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
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
                show_left_ui=False, # for Mia laptop
                show_right_ui=False # for Mia laptop
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )
    # ==================================================================== #

def evaluate_robot(
    brain_genotype,
    robot_graph,
    controller_func,
    network_specs,
    experiment_mode = "simple"
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
    tracker = Tracker(
        mujoco_obj_to_find =    mj.mjtObj.mjOBJ_GEOM,
        name_to_bind =          "core"
    )
    ctrl = Controller(
        controller_callback_function=   controller_func,
        tracker =                       tracker
    )
    
    # Decode genotype to weight matrices
    w_matrices = decode_brain_genotype(
        brain_genotype = brain_genotype,
        network_specs= network_specs)
    
    # Run experiment
    experiment(
        robot = robot_spec,
        controller = ctrl,
        matrices = w_matrices,
        duration = 15,
        mode = experiment_mode
    )
    
    # show_xpos_history(tracker.history["xpos"][0])
    
    # Return fitness
    return (fitness_function(tracker.history["xpos"][0]), )

def main() -> None:
    """Entry point."""
    # ? ------------------------------------------------------------------ #
    genotype_size = 64
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    body_genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(body_genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    save_graph_as_json(
        robot_graph,
        DATA / "robot_graph.json",
    )

    # ? ------------------------------------------------------------------ #
    ### === Preparation for EA Brain ===
    
    # DEAP preps
    _, ind_type = ensure_deap_types()
    
    # Initialization distribution
    func_gauss = partial(random.gauss, 0, 1)
    
    # Define the network specifications
    input_size, output_size = find_in_out_size(robot_graph, SPAWN_POS)
    network_specs = {
        "input_size" :          input_size,
        "output_size" :         output_size,
        "hidden_size" :         8,
        "no_hidden_layers" :    2
    }
    
    # Calculate the size of a brain genotype, based on network specs
    ind_size = compute_brain_genome_size(network_specs)


    # create whole arithmatic brain crossover function
    # DO CROSSOVER ALLELE WISE 
    def crossover_brain(parent1, parent2, alpha=0.4):
        """Applies whole artihmatic crossover to pairs of brain genotypes (parents)
        to produce two new brain genotypes (offspring)

        Parameters
        ----------
        parent1 : list
            genotype parent 1
        parent2 : list
            genotype parent 2
        alpha : float, optional
            weighting factor, by default 0.4

        Returns
        -------
        tuple
            returns a tuple with the genotype of the offspring
        """

        for i in range(len(parent1)):
            # arithmetic crossover per allele
            of1 = alpha * parent1[i] + (1 - alpha) * parent2[i]
            of2 = (1 - alpha) * parent1[i] + alpha * parent2[i]

            # modify in-place
            parent1[i] = of1
            parent2[i] = of2
        return parent1,parent2

  
    # Register factories and evaluation function
    toolbox = base.Toolbox()
    register_factories(
        t=              toolbox,
        ind_type=       ind_type,
        init_func=      func_gauss,
        t_attr_name=    "attr_float",
        t_ind_name=     "create_brain_genome",
        t_pop_name=     "create_brain_genome_pop",
        no_alleles=     ind_size,
    )
    toolbox.register(
        "EvaluateRobot",
        evaluate_robot,
        robot_graph =       robot_graph, # This is the phenotyp expression of the body genotype.
        controller_func =   nn_controller,
        experiment_mode =   "simple", 
        network_specs =     network_specs
    )
    toolbox.register(
        "ParentSelectBrain",
        tools.selTournament,
        tournsize = 3
    )
    toolbox.register(
        "GaussianMutation",
        tools.mutGaussian,
        mu=0, 
        sigma=0.2,
        indpb=0.5
    )
    toolbox.register(
        "CrossoverBrain",
        crossover_brain
    )
    toolbox.register(
        "SurvivorSelection",
        tools.selBest,
        k= 100 # THIS SHOULD BE EQUAL TO POP SIZE
    )
    # ? ------------------------------------------------------------------ #
    ### === Evolutionary Algorithm for Brain ===
    def run_EA_brain(NGEN, pop_size, MUTPB, CXPB):

        # Initialise the population
        pop_brain_genotype = toolbox.create_brain_genome_pop(n = pop_size)
        
        # Assign each individual a fitness value
        f_brain_genotype = toolbox.map(toolbox.EvaluateRobot, pop_brain_genotype)
        for ind, f in zip(pop_brain_genotype, f_brain_genotype):
            ind.fitness.values = f
            # print("individual",ind)
            # print("fitness", f)
            #print("initial pop",ind.fitness.values)

        for i in range(NGEN):
            # Select parents
            parents = toolbox.ParentSelectBrain(pop_brain_genotype, k=pop_size)
        
            # crossover parents
            offspring = [toolbox.clone(ind) for ind in parents]

            # apply crossover to cloned parents
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random()<CXPB:
                    toolbox.CrossoverBrain(child1, child2, alpha=0.5)
                    del child1.fitness.values
                    del child2.fitness.values

            # apply mutation to children
            for mutant in offspring:
                if MUTPB > random.random():
                    toolbox.GaussianMutation(mutant)
                    del mutant.fitness.values

            # calclulate fitness offspring
            for ind in offspring:
                # do not evaluate robot brains if they already have a fitness
                if len(ind.fitness.values) == 0:
                    ind.fitness.values = toolbox.EvaluateRobot(ind)
            
            # survival selection
            combined = pop_brain_genotype + offspring
            selected=toolbox.SurvivorSelection(combined)

            # visualize the second and last generation fitnesses
            for ind in selected:
                if i==NGEN-1 or i==0:
                    print(f"Gen {i} fitness",ind.fitness.values)

            # replace the population with the selected individuals
            pop_brain_genotype = selected

        best_ind = tools.selBest(pop_brain_genotype, k=1)[0]
        return pop_brain_genotype, best_ind
    

    # Run the EA
    last_pop, best_ind=run_EA_brain(5,100,0.5,1)


    # Visualize the best individual in launcher mode
    print("\nBest fitness:", best_ind.fitness.values)
    evaluate_robot(
        brain_genotype=best_ind,
        robot_graph=robot_graph,
        controller_func=nn_controller,
        network_specs=network_specs,
        experiment_mode="launcher" 
    )

    

    



         
    # ? ------------------------------------------------------------------ #
    
if __name__ == "__main__":
    main()
    
