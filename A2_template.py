# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# import fitness functions
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # don’t probe CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       # hide INFO
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # avoid oneDNN init noise

import itertools
import random
from deap import base, creator, tools
import torch
import torch.nn as nn
import multiprocessing as mp
import time 

G_MODEL = None
G_DATA = None
G_TO_TRACK = None
G_NN = None
G_GOAL = None 

def worker_init(goal_xy, hidden_dim, n_layers, seed=None):
    """Runs once per worker: build world+robot, compile model, allocate data, bind geoms, build NN."""
    import os, numpy as np, torch, mujoco
    from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
    from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

    global G_MODEL, G_DATA, G_TO_TRACK, G_NN, G_GOAL

    # per-worker seeds (optional)
    if seed is not None:
        pid_seed = (seed ^ (os.getpid() & 0xFFFFFFFF))
        np.random.seed(pid_seed)
        torch.manual_seed(pid_seed)

    # build world + robot, then compile ONCE in this worker
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    G_MODEL = world.spec.compile()
    G_DATA = mujoco.MjData(G_MODEL)

    # bind tracked geom once 
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    G_TO_TRACK = [G_DATA.bind(g) for g in geoms if "core" in g.name]

    # build NN 
    input_dim = len(G_DATA.qvel) + len(G_DATA.qpos)
    output_dim = G_MODEL.nu
    G_NN = NeuralNet(input_dim=input_dim, output_dim=output_dim,
                     n_layers=n_layers, hidden_dim=hidden_dim)
    G_NN.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    # store goal
    G_GOAL = np.asarray(goal_xy, dtype=float)

class NeuralNet(nn.Module):
    """
    Neural Network class specification
    """
    def __init__(self, 
        input_dim:      int, 
        output_dim:     int, 
        n_layers:       int, 
        hidden_dim:     int, 
        activation_output: nn.Module    =   nn.Tanh(),
        activation: nn.Module           =   nn.Tanh()
        ):
        super().__init__()
        layers = []
        x_dim = input_dim
        
        for _ in range(n_layers):
            layers.append(nn.Linear(x_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation)
            x_dim = hidden_dim # Continue with hidden sized inputs
        layers.append(nn.Linear(x_dim, output_dim))
        layers.append(activation_output)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def size_network_weights(model: nn.Module) -> int:
    """ Return the number of weights in the entire network """
    return sum(p.numel() for p in model.parameters()) 

@torch.no_grad()
def assign_weights(
    model:      nn.Module,
    w_np:       np.ndarray
    ):
    """ Assign weights from individual onto the network """
    w_np = np.asarray(w_np, dtype = np.float32).ravel(order="C") # Make sure w_np is the right format
    w_t_cpu = torch.from_numpy(w_np)
    idx = 0
    for p in model.parameters():
        n = p.numel()
        if idx + n > w_t_cpu.numel():
            raise ValueError("Individual size is smaller than the number of weights in the network")
        slice_view = w_t_cpu[idx:idx + n].reshape(p.shape).to(device=p.device, dtype=p.dtype)
        p.copy_(slice_view)
        idx += n

@torch.no_grad()
def controller(model, data, to_track, NN, history):  # noqa: ANN001, N803
    """Controls robot movement based on neural net output."""  # noqa: D401
    # Get inputs, make sure that input type and device match neural net
    p = next(NN.parameters())
    dtype = p.dtype
    device = p.device
    in_mujoco = np.concatenate([data.qpos.copy(), data.qvel.copy()], axis = 0)
    inputs = torch.from_numpy(in_mujoco).to(device = device, dtype = dtype).unsqueeze(0)
    # Get outputs from neural net
    outputs = NN(inputs)
    outputs = outputs.squeeze().detach().cpu().numpy()
    
    # Adjust robot hinges gradually
    delta = 0.05
    data.ctrl = np.clip((outputs * delta) + data.ctrl, -np.pi/2, np.pi/2) 

    history.append(to_track[0].xpos.copy())
    
def random_move(model, data, to_track, history) -> None:
    """Generate random movements for the robot's joints.
    
    The mujoco.set_mjcb_control() function will always give 
    model and data as inputs to the function. Even if you don't use them,
    you need to have them as inputs.

    Parameters
    ----------

    model : mujoco.MjModel
        The MuJoCo model of the robot.
    data : mujoco.MjData
        The MuJoCo data of the robot.

    Returns
    -------
    None
        This function modifies the data.ctrl in place.
    """

    # Get the number of joints
    num_joints = model.nu 
    
    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi/2
    rand_moves = np.random.uniform(low= -hinge_range, # -pi/2
                                   high=hinge_range, # pi/2
                                   size=num_joints) 

    # There are 2 ways to make movements:
    # 1. Set the control values directly (this might result in junky physics)
    # data.ctrl = rand_moves

    # 2. Add to the control values with a delta (this results in smoother physics)
    delta = 0.05
    data.ctrl += rand_moves * delta 

    # Bound the control values to be within the hinge limits.
    # If a value goes outside the bounds it might result in jittery movement.
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

    # Save movement to history
    history.append(to_track[0].xpos.copy())

def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history) 
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.savefig("trajectory.png")
    print("Saved plot to trajectory.png")


def evaluateInd(individual):
    """ Simulate individual and evaluate fitness """
    history = []
    
    # Generate world and robot
    mujoco.mj_resetData(G_MODEL, G_DATA)
    mujoco.set_mjcb_control(None) 
    
    # Decode genotype and assign weights to network
    assign_weights(G_NN, individual)
    
    to_track = G_TO_TRACK
    
    # Start simulation
    mujoco.set_mjcb_control(lambda m, d: controller(m, d, to_track, G_NN, history))
    simulation_time = 10
    
    while G_DATA.time < simulation_time:
        mujoco.mj_step(G_MODEL, G_DATA, nstep= 100)
    
    # Evaluate fitness
    final_pos = np.array(history)[-1, :2]
    fitness = distance_to_target(final_pos, G_GOAL)
    return (float(fitness), )

def evaluateRW(individual):
    """ Simulate individual and evaluate fitness """
    history = []
    
    # Generate world and robot
    mujoco.mj_resetData(G_MODEL, G_DATA)
    mujoco.set_mjcb_control(None) 
    
    to_track = G_TO_TRACK
    
    # Start simulation
    mujoco.set_mjcb_control(lambda m, d: random_move(m, d, to_track, history))
    simulation_time = 20
    
    while G_DATA.time < simulation_time:
        mujoco.mj_step(G_MODEL, G_DATA, nstep= 100)
    
    # Evaluate fitness
    final_pos = np.array(history)[-1, :2]
    fitness = distance_to_target(final_pos, G_GOAL)
    return (float(fitness), )

def renderBest(individual):
    history = []
    # build world+robot in the main process
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # bind geom to track
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(g) for g in geoms if "core" in g.name]

    # build a *local* NN with the same architecture
    input_dim = len(data.qvel)
    output_dim = model.nu
    NN = NeuralNet(input_dim=input_dim, output_dim=output_dim,
                   n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM)
    assign_weights(NN, individual)  # put best weights in the local NN

    mujoco.mj_resetData(model, data)
    mujoco.set_mjcb_control(None)
    mujoco.set_mjcb_control(lambda m, d: controller(m, d, to_track, NN, history))

    viewer.launch(model, data)
    show_qpos_history(history)

def plot_A2(best, averages, std, name):
    """ Plot the best, average, and std of the fitness in every generation"""
    # Convert to np.array
    averages = np.array(averages)
    std = np.array(std)
    x = np.arange(len(best))
    # Plot and save as "A2_plot.png"
    plt.plot(x, best, 'r-', label='Best_fitness')
    plt.plot(x, averages, 'b-', label='Averages')
    plt.fill_between(x, np.array(averages) - std, averages + std, alpha = 0.2, label="Std")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig(name)

def population_diversity(pop):
    """
    Measure the diversity in a population by evaluating the euclidean distance of each
    individual from each other. The average of the distance is a metric for diversity. 
    Higher average distance means 
    """
    # Cast individuals as numpy
    arrays = [np.array(x, dtype = float) for x in pop]
    dists = [np.linalg.norm(a - b) for a, b, in itertools.combinations(arrays, 2)]
    return (float(np.mean(dists)), float(np.min(dists)), float(np.max(dists)))
        
def whole_arithmetic_recomb(ind1, ind2, alpha):
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        cross_value = alpha * x2 + (1 - alpha) * x1
        ind1[i] = cross_value
        ind2[i] = cross_value

    return ind1, ind2

def render_video_of_ind(individual, duration = 30, path="./__videos__"):
    history = []
    # Get single robot to get input and output dims ---    
    world = SimpleFlatWorld()
    gecko_core = gecko()  
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model) 
    # Bind tracked geom(s)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(g) for g in geoms if "core" in g.name]
    # Network structure
    input_dim = len(data.qvel) + len(data.qpos) 
    output_dim = model.nu
    # Create neural net
    NN = NeuralNet(
        input_dim =     input_dim,
        output_dim =    output_dim,
        n_layers =      N_LAYERS, 
        hidden_dim =    HIDDEN_DIM
    )
    # Assign weights
    assign_weights(NN, individual)
    # Assign controls
    mujoco.mj_resetData(model, data)
    mujoco.set_mjcb_control(None)
    mujoco.set_mjcb_control(lambda m, d: controller(m, d, to_track, NN, history))
    # Record video
    video_recorder = VideoRecorder(output_folder=path)
    video_renderer(
        model,
        data,
        duration,
        video_recorder=video_recorder
    )
    
def main(experiment = "Blend", RW = False):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Multi-core pool
    ctx = mp.get_context("spawn")  # portable across OSes
    pool = ctx.Pool(
        processes=max(1, mp.cpu_count() - 1),
        initializer=worker_init,
        initargs=(GOAL, HIDDEN_DIM, N_LAYERS, SEED)
    )
        
    # Get single robot to get input and output dims ---    
    world = SimpleFlatWorld()
    gecko_core = gecko()  
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model) 
    
    # Network structure
    input_dim = len(data.qvel) + len(data.qpos) 
    output_dim = model.nu

    # Create neural net
    NN = NeuralNet(
        input_dim =     input_dim,
        output_dim =    output_dim,
        n_layers =      N_LAYERS, 
        hidden_dim =    HIDDEN_DIM
    )
    
    # Population, individual, and elite size
    global IND_SIZE
    IND_SIZE = size_network_weights(NN) # Don't change!
    
    
    # Setup DEAP toolbox
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    
    # Set population intitialization 
    toolbox.register("attr_float", np.random.normal, loc=0.0, scale=0.1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Set variation operators
    if experiment == "Blend":
        toolbox.register("mate", tools.cxBlend, alpha = 0.4) # Blend Crossover
    elif experiment == "Arithmetic":
        toolbox.register("mate", whole_arithmetic_recomb, alpha = 0.5) # Whole arithmetic crossover
    toolbox.register("mutate", tools.mutGaussian,mu = 0.0, sigma = 0.1, indpb = 0.1)
    
    # Set selection operators
    toolbox.register("select_parents", tools.selTournament, tournsize = 2, k = POP_SIZE) 
    toolbox.register("select_survivors", tools.selBest, k = POP_SIZE - IMMIGRANTS)
    
    # Set evaluation and multi-core processing
    if RW:
        toolbox.register("evaluate", evaluateRW)
    else:
        toolbox.register("evaluate", evaluateInd) 
    toolbox.register("map", pool.map)
    
    # Initialize population and evaluate initial fitnesses
    pop = toolbox.population(n = POP_SIZE)
    init_f = toolbox.map(toolbox.evaluate, pop)
    for ind, f in zip(pop, init_f):
        ind.fitness.values = f
    
    # Containers for best, averages, and std    
    best_individuals = []
    best_fits = []
    averages = []
    stds = []

    print(0.1414 * np.sqrt(IND_SIZE))
    # Simulate NGEN generations
    for gen in range(NGEN):
        # Random walk
        if RW:
            print("RW")
            fit_arr = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
            avg = np.mean(fit_arr)
            std = np.std(fit_arr)
            fmin = np.min(fit_arr)
            fmax = np.max(fit_arr)
            av_dist, min_dist, max_dist= population_diversity(pop)
            # Print EA progress
            print(f"{'NGEN':>3} {'n_pop':>7} {'average':>10.4} {'std':>10.4} {'min':>10.4} {'max':>10.4} {'av_dist.':>10.7} {'min_dist':>10.7} {'max_dist':>10.7}")
            print(f"{gen + 1:>3} {len(pop):>7} {avg:>10.4f} {std:>10.4f} {fmin:>10.4f} {fmax:>10.4f} {av_dist:>10.7f} {min_dist:>10.7f} {max_dist:>10.7f}")
            # Save statistics
            best_individuals.append(tools.selBest(pop, k=1)[0])
            best_fits.append(tools.selBest(pop, k=1)[0].fitness.values)
            averages.append(avg)
            stds.append(std)
            # New random walk, ready for next eval
            pop = toolbox.population(n = POP_SIZE)
            init_f = toolbox.map(toolbox.evaluate, pop)
            for ind, f in zip(pop, init_f):
                ind.fitness.values = f
        # EA with either blend or whole arithmetic
        else:
            # Parent selection
            parents = toolbox.select_parents(pop)
            random.shuffle(parents)
            offspring = list(toolbox.map(toolbox.clone, parents))
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    # child1[:] = np.clip(np.array(child1), -0.5, 0.5).tolist()
                    # child2[:] = np.clip(np.array(child2), -0.5, 0.5).tolist()
                    del child1.fitness.values
                    del child2.fitness.values
            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                    # Clip to keep weights in sensible range and prevent exploding weights
                    mutant[:] = np.clip(np.array(mutant), -1, 1).tolist()
            # Evaluate offspring fitnesses of individuals which had genotypes changed by mating and mutating
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # New generation using age-based selection, elitism, and immigration
            if IMMIGRANTS != 0:
                immigrants = toolbox.population(n = IMMIGRANTS)
                imm_fitnesses = list(toolbox.map(toolbox.evaluate, immigrants))
                for ind, fit in zip(immigrants, imm_fitnesses):
                    ind.fitness.values = fit
            else:
                immigrants = []
            pop[:] = toolbox.select_survivors(offspring + tools.selBest(pop, k = E))  + immigrants
            # Extract statistics
            fit_arr = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
            avg = np.mean(fit_arr)
            std = np.std(fit_arr)
            fmin = np.min(fit_arr)
            fmax = np.max(fit_arr)
            av_dist, min_dist, max_dist= population_diversity(pop)
            # Print EA progress
            print(f"{'NGEN':>3} {'n_pop':>7} {'average':>10.4} {'std':>10.4} {'min':>10.4} {'max':>10.4} {'av_dist.':>10.7} {'min_dist':>10.7} {'max_dist':>10.7}")
            print(f"{gen + 1:>3} {len(offspring):>7} {avg:>10.4f} {std:>10.4f} {fmin:>10.4f} {fmax:>10.4f} {av_dist:>10.7f} {min_dist:>10.7f} {max_dist:>10.7f}")
            # Save statistics
            best_individuals.append(tools.selBest(pop, k=1)[0])
            best_fits.append(tools.selBest(pop, k=1)[0].fitness.values)
            averages.append(avg)
            stds.append(std)
            
            


    # Plot A2 plot
    best_fits = [ind.fitness.values for ind in best_individuals]
    if experiment == "Blend":
        plot_A2(best_fits, averages, stds, "blend_crossover")
    elif experiment == "Arithmetic":
        plot_A2(best_fits, averages, stds, "arithmetic_crossover")
    elif RW:
        plot_A2(best_fits, averages, stds, "random_walk")
    # Record video of best individual
    best_ind = tools.selBest(best_individuals, 1)[0]
    print(f"Best individual fitness: {best_ind.fitness.values}")
    render_video_of_ind(best_ind)

    # Close pool
    pool.close()
    pool.join()
    
    return best_fits, averages, stds, best_ind

if __name__ == "__main__":
    """
    For the output:
    av d: The average euclidean distance between individuals. To measure diversity
    min: The minimum euclidean distance found
    max: The maximum euclidean distance found
    The remaining values are fitness statisticss, population size, and the number of generations passed
    """
    
    # GOAL
    GOAL = [0, -3]
    # Set seed
    SEED = 42
    # Elitism and immigrants
    E = 0
    IMMIGRANTS = 0
    # Population
    POP_SIZE = 100
    # Number of Generations
    NGEN = 20

    # Network Specifications
    HIDDEN_DIM = 128
    N_LAYERS = 3

    # Probability of crossover and mutation occuring on an individual
    CXPB = 0.7                          
    MUTPB = 0.4
    
    # Testing
    # _, _, _, best_ind = main(experiment="Blend")

    
    # Assignment plotting
    rw_best_fits = []
    rw_averages = []
    rw_stds = []
    blend_best_fits = []
    blend_averages = []
    blend_stds = []
    arithmetic_best_fits = []
    arithmetic_averages = []
    arithmetic_stds = []
    
    for _ in range(3):
        best_fit, averages, stds, best_ind = main(RW=True)
        rw_best_fits.append(best_fit)
        rw_averages.append(averages)
        rw_stds.append(stds)
    for _ in range(3):
        best_fit, averages, stds, best_ind = main(experiment = "Blend")
        blend_best_fits.append(best_fit)
        blend_averages.append(averages)
        blend_stds.append(stds)
    for _ in range(3):
        best_fit, averages, stds, best_ind = main(experiment = "Arithmetic")
        arithmetic_best_fits.append(best_fit)
        arithmetic_averages.append(averages)
        arithmetic_stds.append(stds)
        
    rw_best_fits = np.mean(np.reshape(rw_best_fits, shape=(3, NGEN)), axis=0)
    rw_averages = np.mean(np.reshape(rw_averages, shape=(3, NGEN)), axis=0)
    rw_stds = np.mean(np.reshape(rw_stds, shape=(3, NGEN)), axis = 0)
    plot_A2(rw_best_fits, rw_averages, rw_stds, "Random_walk_3_runs")
    
    blend_best_fits = np.mean(np.reshape(blend_best_fits, shape=(3, NGEN)), axis=0)
    blend_averages = np.mean(np.reshape(blend_averages, shape=(3, NGEN)), axis=0)
    blend_stds = np.mean(np.reshape(blend_stds, shape=(3, NGEN)), axis = 0)
    plot_A2(blend_best_fits, blend_averages, blend_stds, "Blend_CO_3_runs")
    
    arithmetic_best_fits = np.mean(np.reshape(arithmetic_best_fits, shape=(3, NGEN)), axis=0)
    arithmetic_averages = np.mean(np.reshape(arithmetic_averages, shape=(3, NGEN)), axis=0)
    arithmetic_stds = np.mean(np.reshape(arithmetic_stds, shape=(3, NGEN)), axis = 0)
    plot_A2(arithmetic_best_fits, arithmetic_averages, arithmetic_stds, "Arithmetic_CO_3_runs")

    
    
    
    
    

    
    
      

        


