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
NUM_OF_MODULES = 6
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

from deap import tools
import pickle

def main() -> None:
    """Entry point."""

    # === Step 1: Generate a random body genotype ===
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

    # Decode the high-probability graph (body phenotype)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # Save body structure to file
    save_graph_as_json(robot_graph, DATA / "robot_graph.json")

    # === Step 2: Evolutionary setup for the brain ===
    _, ind_type = ensure_deap_types()
    func_gauss = partial(random.gauss, 0, 1)

    input_size, output_size = find_in_out_size(robot_graph, SPAWN_POS)
    network_specs = {
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": 8,
        "no_hidden_layers": 2,
    }

    ind_size = compute_brain_genome_size(network_specs)

    toolbox = base.Toolbox()
    register_factories(
        t=toolbox,
        ind_type=ind_type,
        init_func=func_gauss,
        t_attr_name="attr_float",
        t_ind_name="create_brain_genome",
        t_pop_name="create_brain_genome_pop",
        no_alleles=ind_size,
    )
    toolbox.register(
        "EvaluateRobot",
        evaluate_robot,
        robot_graph=robot_graph,
        controller_func=nn_controller,
        experiment_mode="simple",   # FAST (no graphics)
        network_specs=network_specs,
    )

    # === Step 3: Run a single generation (baseline experiment) ===
    pop_size = 50
    pop_brain_genotype = toolbox.create_brain_genome_pop(n=pop_size)

    f_brain_genotype = toolbox.map(toolbox.EvaluateRobot, pop_brain_genotype)
    for ind, f in zip(pop_brain_genotype, f_brain_genotype):
        ind.fitness.values = f

    # Get best individual
    best_ind = tools.selBest(pop_brain_genotype, 1)[0]
    print("Best fitness:", best_ind.fitness.values[0])

    # Save best brain to disk
    with open(DATA / "best_brain.pkl", "wb") as f:
        pickle.dump(best_ind, f)

    # === Step 4: Replay best robot with visualization ===
    print("Replaying best robot with visualization...")
    evaluate_robot(
        brain_genotype=best_ind,
        robot_graph=robot_graph,
        controller_func=nn_controller,
        network_specs=network_specs,
        experiment_mode="launcher",  # change to "video" to save video
    )

if __name__ == "__main__":
        # === Body EA helper code ===
    from deap import base, creator, tools
    import math
    import copy

    # Parameters for the body genotype representation
    GENE_LEN = 64  # same as in main() for each of type/conn/rot
    BODY_VECTOR_LEN = 3 * GENE_LEN  # flattened [type, conn, rot]

    # EA settings (tune these)
    BODY_POP = 30
    BODY_NGEN = 5
    CX_PROB = 0.6
    MUT_PROB = 0.3
    MUT_SIGMA = 0.2  # gaussian sigma for float mutation

    # Brain settings used to *evaluate* each body (cheap inner loop)
    # You can increase inner_pop/inner_gen for better evaluation (more compute)
    INNER_BRAIN_POP = 30
    INNER_BRAIN_GEN = 2

    # Setup DEAP types for body individuals
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("BodyIndividual", list, fitness=creator.FitnessMax)

    toolbox_body = base.Toolbox()
    # Initialiser: uniform floats in [0,1]
    toolbox_body.register("attr_float", RNG.random)
    toolbox_body.register(
        "individual",
        tools.initRepeat,
        creator.BodyIndividual,
        toolbox_body.attr_float,
        BODY_VECTOR_LEN,
    )
    toolbox_body.register("population", tools.initRepeat, list, toolbox_body.individual)

    # basic GA operators
    toolbox_body.register("mate", tools.cxTwoPoint)
    toolbox_body.register("mutate", tools.mutGaussian, mu=0.0, sigma=MUT_SIGMA, indpb=0.05)
    toolbox_body.register("select", tools.selTournament, tournsize=3)

    # Helper: unpack flat body genotype into three arrays expected by NDE
    def unpack_body_genotype(individual):
        # individual: flat list length 3*GENE_LEN
        a = np.array(individual[:GENE_LEN], dtype=np.float32)
        b = np.array(individual[GENE_LEN:2 * GENE_LEN], dtype=np.float32)
        c = np.array(individual[2 * GENE_LEN:], dtype=np.float32)
        return [a, b, c]

    # Helper: connectivity check (penalize disconnected graphs)
    def is_graph_connected(robot_graph):
        try:
            # robot_graph is a networkx.DiGraph; convert to undirected and check connectivity
            und = robot_graph.to_undirected()
            import networkx as nx
            return nx.is_connected(und)
        except Exception:
            return False

    # Option A (cheap): Evaluate a body using a single RANDOM brain (Baseline-like)
    def evaluate_body_with_random_brain(individual):
        body_genotype = unpack_body_genotype(individual)
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        p_matrices = nde.forward(body_genotype)
        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        robot_graph = hpd.probability_matrices_to_graph(*p_matrices)
        # Save graph for debugging
        save_graph_as_json(robot_graph, DATA / f"body_graph_{hash(tuple(individual))%100000}.json")

        # connectivity check
        if not is_graph_connected(robot_graph):
            # Very bad body
            return (-100.0,)

        # Build robot_spec
        robot_spec = construct_mjspec_from_graph(robot_graph)

        # Prepare a *single random brain* (baseline-like), using network_specs inferred
        input_size, output_size = find_in_out_size(robot_graph, SPAWN_POS)
        network_specs = {
            "input_size": input_size,
            "output_size": output_size,
            "hidden_size": 8,
            "no_hidden_layers": 2,
        }
        brain_size = compute_brain_genome_size(network_specs)
        # Random gaussian brain
        brain_genotype = [random.gauss(0, 1) for _ in range(brain_size)]

        # Evaluate robot (this uses evaluate_robot wrapper but with brain provided)
        # If your evaluate_robot expects a particular format, adapt accordingly.
        fitness = evaluate_robot(
            brain_genotype=brain_genotype,
            robot_graph=robot_graph,
            controller_func=nn_controller,
            network_specs=network_specs,
            experiment_mode="simple",  # fast, headless evaluation
        )
        return fitness

    # Option B (recommended for quality): Evaluate body by doing a small inner EA on the brain
    def evaluate_body_with_inner_brain_ea(individual):
        # same decode steps
        body_genotype = unpack_body_genotype(individual)
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        p_matrices = nde.forward(body_genotype)
        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        robot_graph = hpd.probability_matrices_to_graph(*p_matrices)

        save_graph_as_json(robot_graph, DATA / f"body_graph_{hash(tuple(individual))%100000}.json")
        if not is_graph_connected(robot_graph):
            return (-100.0,)

        # network specs
        input_size, output_size = find_in_out_size(robot_graph, SPAWN_POS)
        network_specs = {
            "input_size": input_size,
            "output_size": output_size,
            "hidden_size": 8,
            "no_hidden_layers": 2,
        }
        brain_size = compute_brain_genome_size(network_specs)

        # Inner EA to find decent brain for this body (cheap: small pop, few gens)
        # Setup DEAP for brain only (local toolbox to avoid names collision)
        creator.create("FitnessMaxBrain", base.Fitness, weights=(1.0,))
        creator.create("BrainIndividual", list, fitness=creator.FitnessMaxBrain)
        tb = base.Toolbox()
        tb.register("attr_f", random.gauss, 0, 1)
        tb.register("individual", tools.initRepeat, creator.BrainIndividual, tb.attr_f, brain_size)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("mate", tools.cxTwoPoint)
        tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.05)
        tb.register("select", tools.selTournament, tournsize=3)

        # Evaluate brain by running robot simulation (calls evaluate_robot)
        def evaluate_brain_ind(brain_ind):
            return evaluate_robot(
                brain_genotype=list(brain_ind),
                robot_graph=robot_graph,
                controller_func=nn_controller,
                network_specs=network_specs,
                experiment_mode="simple",
            )

        tb.register("evaluate", evaluate_brain_ind)

        # small inner run
        bpop = tb.population(n=INNER_BRAIN_POP)
        # evaluate initial
        for ind in bpop:
            ind.fitness.values = tb.evaluate(ind)
        for g in range(INNER_BRAIN_GEN):
            offspring = tb.select(bpop, len(bpop))
            offspring = list(map(tb.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    tb.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < 0.2:
                    tb.mutate(mutant)
                    del mutant.fitness.values
            # evaluate invalid
            invalid = [ind for ind in offspring if not hasattr(ind, "fitness") or not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = tb.evaluate(ind)
            bpop[:] = offspring

        # best brain fitness for this body:
        best = max(bpop, key=lambda x: x.fitness.values[0])
        best_fit = best.fitness.values[0]
        return (best_fit,)

    # Choose evaluator (pick Option B for quality; Option A for speed)
    BODY_EVALUATOR = evaluate_body_with_inner_brain_ea  # OR evaluate_body_with_random_brain

    toolbox_body.register("evaluate", BODY_EVALUATOR)

    # === Run Body EA ===
    def run_body_ea():
        pop = toolbox_body.population(n=BODY_POP)
        # evaluate initial population
        for ind in pop:
            ind.fitness.values = toolbox_body.evaluate(ind)

        for gen in range(BODY_NGEN):
            offspring = toolbox_body.select(pop, len(pop))
            offspring = list(map(toolbox_body.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CX_PROB:
                    toolbox_body.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < MUT_PROB:
                    toolbox_body.mutate(mutant)
                    # keep genes in [0,1]
                    for i in range(len(mutant)):
                        if mutant[i] < 0.0:
                            mutant[i] = 0.0
                        elif mutant[i] > 1.0:
                            mutant[i] = 1.0
                    del mutant.fitness.values

            # evaluate newly invalid individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox_body.evaluate(ind)

            # replace population
            pop[:] = offspring

            # Logging
            fits = [ind.fitness.values[0] for ind in pop]
            print(f"Gen {gen}, Best {max(fits):.4f}, Avg {sum(fits)/len(fits):.4f}")

        # return best individual
        best = max(pop, key=lambda i: i.fitness.values[0])
        return best

    # Run it
    best_body = run_body_ea()
    print("Best body fitness:", best_body.fitness.values)
    # Save best body graph
    best_body_genotype = unpack_body_genotype(best_body)
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward(best_body_genotype)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    best_graph = hpd.probability_matrices_to_graph(*p_mats)
    save_graph_as_json(best_graph, DATA / "best_body_graph.json")
    main()
    
