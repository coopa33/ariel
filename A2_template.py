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


import random
from deap import base, creator, tools, algorithms
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import multiprocessing as mp

class NeuralNet(nn.Module):
    def __init__(self, 
        input_dim:      int, 
        output_dim:     int, 
        n_layers:       int, 
        hidden_dim:     int, 
        activation_output: nn.Module    =   nn.Tanh(),
        activation: nn.Module           =   nn.LeakyReLU()
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
    return sum(p.numel() for p in model.parameters()) 

@torch.no_grad()
def assign_weights(
    model:      nn.Module,
    w_np:       np.ndarray
    ):
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
    
# Keep track of data / history
HISTORY = []

@torch.no_grad()
def controller(model, data, to_track, NN, history):
    # Get inputs, make sure that input type and device match NN
    p = next(NN.parameters())
    dtype = p.dtype
    device = p.device

    inputs = torch.from_numpy(data.qpos).to(device = device, dtype = dtype).unsqueeze(0)
    
    # Get outputs from NN instance parameter
    outputs = NN(inputs)
    outputs = outputs.squeeze().detach().cpu().numpy()
    
    # Scale outputs cover full movement range of hinges [-pi/2, pi/2]
    delta = 0.05
    scaling = np.pi/2

    data.ctrl[:] = np.clip((outputs * scaling), -np.pi/2, np.pi/2) 

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
    data.ctrl += np.clip(data.ctrl, -np.pi/2, np.pi/2)

    # Save movement to history
    history.append(to_track[0].xpos.copy())

    ##############################################
    #
    # Take all the above into consideration when creating your controller
    # The input size, output size, output range
    # Your network might return ranges [-1,1], so you will need to scale it
    # to the expected [-pi/2, pi/2] range.
    # 
    # Or you might not need a delta and use the direct controller outputs
    #
    ##############################################

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

def evaluateInd(individual, NN):
    history = []
    mujoco.set_mjcb_control(None) # DO NOT REMOVE

    world = SimpleFlatWorld()
    gecko_core = gecko()     # DO NOT CHANGE
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    
    # Decode genotype and assign weights to network
    assign_weights(NN, individual)

    mujoco.set_mjcb_control(lambda m, d: controller(m, d, to_track, NN, history))
    
    simulation_time = 20
    goal = np.array([0, -2])
    while data.time < simulation_time:
        mujoco.mj_step(model, data)
        
    final_pos = np.array(history)[-1, :2]
    fitness = distance_to_target(final_pos, goal)
    
    return (float(fitness), )

def renderBest(individual, NN):
    assign_weights(NN, individual)
    history = []
    mujoco.set_mjcb_control(None) # DO NOT REMOVE

    world = SimpleFlatWorld()
    gecko_core = gecko()     # DO NOT CHANGE
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    
    # Decode genotype and assign weights to network
    assign_weights(NN, individual)
    
    mujoco.set_mjcb_control(lambda m,d: controller(m, d, to_track, NN, history))
        
    # # --- Simulation with rendering
    viewer.launch(
        model,
        data,
    )
    show_qpos_history(history)

def plot_A2(best, averages, std):
    averages = np.array(averages)
    std = np.array(std)
    x = np.arange(len(best))
    plt.plot(x, best, 'r-', label='Best_fitness')
    plt.plot(x, averages, 'b-', label='Averages')
    plt.fill_between(x, np.array(averages) - std, averages + std, alpha = 0.2, label="Std")

    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("A2_plot")

def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Multi-core pool
    pool = mp.Pool(processes = mp.cpu_count() -1)

    # --- Get single robot to get input and output dims ---    
    world = SimpleFlatWorld()
    gecko_core = gecko()  
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model) 
    
    # Network structure
    input_dim = len(data.qpos) 
    output_dim = model.nu
    hidden_dim = 16
    n_layers = 4
    
    #  Fully connected NN
    NN = NeuralNet(
        input_dim =     input_dim,
        output_dim =    output_dim,
        n_layers =      n_layers, 
        hidden_dim =    hidden_dim
    )
    
    # Global Variables
    POP_SIZE = 100
    NGEN = 200
    CXPB = 0.8
    MUTPB = 0.5                          # Probability of mutation occuring on a individual 
    global IND_SIZE
    IND_SIZE = size_network_weights(NN)    # Individual size is exactly the number of weights in NN.
    E = 0
    # Setup DEAP toolbox
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # Register function to sample float values from uniform distribution
    toolbox.register("attr_float", random.uniform, a = -1, b = 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
    # Register population to consist of above defined individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Register variation operators
    toolbox.register("mate", tools.cxBlend, alpha = 0.5)
    toolbox.register("mutate", tools.mutGaussian,mu = 0.0, sigma = 0.1, indpb = 0.1)
    # Register selection operators
    toolbox.register("select_parents", tools.selTournament, tournsize = 3, k = POP_SIZE) 
    toolbox.register("select_survivors", tools.selBest, k = POP_SIZE - E)
    # Register evaluation operator and make evaluation multi-processor
    toolbox.register("evaluate", evaluateInd, NN = NN)
    toolbox.register("map", pool.map)
    
    total_start = time.time()
    
    # Initialize population and evaluate initial fitnesses
    pop = toolbox.population(n = POP_SIZE)
    init_f = toolbox.map(toolbox.evaluate, pop)
    for ind, f in zip(pop, init_f):
        ind.fitness.values = f
    
    print(f"Initial max fitness:{tools.selBest(pop, k=1)[0].fitness.values}")
    
    best_individuals = []
    averages = []
    std = []
    # Simulate NGEN generations
    for _ in tqdm(range(NGEN)):
        # Parent selection
        parents = toolbox.select_parents(pop)
        random.shuffle(parents)
        offspring = list(toolbox.map(toolbox.clone, parents))

        # offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate offspring fitnesses, only those which had genotypes changed by mating and mutating
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # New generation using age-based selection and elitism
        
        pop[:] = toolbox.select_survivors(offspring) #+ tools.selBest(pop, k = E)
        
        fit_arr = np.array([ind.fitness.values[0] for ind in pop], dtype=float)
        best_individuals.append(fit_arr.min()) 
        averages.append(fit_arr.mean()) 
        std.append(fit_arr.std())
        if np.allclose(pop, pop[0], rtol = 0, atol = 1e-6 ):
            break
        
    for ind in pop:
        print(f"After algorithm pop fitness: {ind.fitness.values}")
    total_end = time.time()
    # print(f"Total time: {total_end - total_start}")
    plot_A2(best_individuals, averages, std)
    
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Best individual fitness: {best_ind.fitness.values}")
    print(best_individuals)
    renderBest(best_ind, NN)
        
    pool.close(); pool.join()

if __name__ == "__main__":
    main()

    
    
      

        


