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
        activation:     nn.Module   =   nn.Tanh()
        ):
        super().__init__()
        layers = []
        x_dim = input_dim
        
        for _ in range(n_layers):
            layers.append(nn.Linear(x_dim, hidden_dim))
            layers.append(activation)
            x_dim = hidden_dim # Continue with hidden sized inputs
        layers.append(nn.Linear(x_dim, output_dim))
        
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
def controller(model, data, to_track, NN):
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
    data.ctrl[:] = np.clip((outputs * delta * scaling) + data.ctrl, -np.pi/2, np.pi/2) 
    
    # Save movement to HISTORY
    HISTORY.append(to_track[0].xpos.copy())
    # print(len(HISTORY))
    
def random_move(model, data, to_track) -> None:
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
    HISTORY.append(to_track[0].xpos.copy())

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



# Create an fitness evaluation function
def evaluateInd(individual, NN):
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
    
    mujoco.set_mjcb_control(lambda m,d: controller(m, d, to_track, NN))
    
    # --- Simulation without rendering
    duration = 10
    while data.time < duration:
        mujoco.mj_step(model, data)
        
    # # --- Simulation with rendering
    # viewer.launch(
    #     model,
    #     data,
    # )
    
        
    
    # Extract initial and final x and y position
    initial_pos = np.array(HISTORY)[0, 0:2].flatten()
    target_pos = np.array(HISTORY)[-1, 0:2].flatten()
    
    # Calculate fitness based on euclidean distance
    d = distance_to_target(initial_pos, target_pos)
    
    HISTORY.clear()
    
    return (d, )

# Create an fitness evaluation function
def renderBest(individual, NN):
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
    
    mujoco.set_mjcb_control(lambda m,d: controller(m, d, to_track, NN))
        
    # # --- Simulation with rendering
    viewer.launch(
        model,
        data,
    )
    show_qpos_history(HISTORY)

def main():
    random.seed(1)
    pool = mp.Pool(processes = mp.cpu_count() - 1)


        
    # --- Get single robot to get input and output dims ---    
    world = SimpleFlatWorld()
    gecko_core = gecko()     # DO NOT CHANGE
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model) 
    
    input_dim = len(data.qpos)
    output_dim = model.nu
    hidden_dim = 64
    n_layers = 4
    
    # --- Neural Net
    NN = NeuralNet(
        input_dim =     input_dim,
        output_dim =    output_dim,
        n_layers =      n_layers, 
        hidden_dim =    hidden_dim
    )
    # Global Variables
    POP_SIZE = 100
    NGEN = 10
    CXPB = 0.5
    MUTPB = 0.5                                              # Probability of mutation occuring on a individual 
    global IND_SIZE
    IND_SIZE = sum([p.numel() for p in NN.parameters()])    # Individual size is exactly the number of weights in NN.
    global DURATION
    DURATION = 5

    creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Register function to sample float values from uniform distribution
    toolbox.register("attr_float", random.uniform, a = -1, b = 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
    
    # Register population to consist of above defined individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register variation operators
    toolbox.register("mate", tools.cxBlend, alpha = 0.5)
    toolbox.register("mutate", tools.mutGaussian, mu = 0.0, sigma = 0.5, indpb = 0.5)

    # Register selection operators
    toolbox.register("select_parents", tools.selTournament, tournsize = 4, k = POP_SIZE) 
    toolbox.register("select_survivors", tools.selBest, k = POP_SIZE)
    
    # Register evaluation operator and make evaluation multi-processor
    toolbox.register("evaluate", evaluateInd, NN = NN)
    toolbox.register("map", pool.map)
    
    
    total_start = time.time()
    
    # Initialize population and evaluate initial fitnesses
    pop = toolbox.population(n = POP_SIZE)
    init_f = toolbox.map(toolbox.evaluate, pop)
    for ind, f in zip(pop, init_f):
        ind.fitness.values = f
    
    print(f"Initial max fitness:{max(init_f)}")
        
    # Simulate NGEN generations
    for _ in tqdm(range(NGEN)):
        offspring = map(toolbox.clone, toolbox.select_parents(pop))
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)

        
        # Evaluate offspring fitnesses, only those which had genotypes changed by mating and mutating
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # New generation
        pop[:] = toolbox.select_survivors(pop + offspring)
        
    for ind in pop:
        print(f"After algorithm pop fitness: {ind.fitness.values}")
    total_end = time.time()
    print(f"Total time: {total_end - total_start}")
    
    # Maximum individual
    final_f = toolbox.map(toolbox.evaluate, pop)
    max_idx = np.argmax(final_f)
    best_ind, best_f = pop[max_idx], final_f[max_idx]
    print(f"Best individual fitness: {best_f}")
    renderBest(best_ind, NN)
        
    pool.close(); pool.join()

if __name__ == "__main__":
    main()

    
    
      

        


