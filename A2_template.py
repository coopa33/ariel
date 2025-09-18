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
# import tensorflow


# Keep track of data / history
HISTORY = []

def controller(model, data, to_track, W1, W2, W3):
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    # Get inputs, in this case the position of the actuator motors (hinges)
    inputs = data.qpos

    # Run the input
    # s through the layers of the network
    layer1 = tanh(np.dot(inputs, W1))
    layer2 = tanh(np.dot(layer1, W2))
    outputs = tanh(np.dot(layer2, W3))
    
    
    # Scale outputs to+ data.ctrl [-pi/2, pi/2]
    delta = 0.05
    scaling = np.pi/2
    data.ctrl = np.clip((outputs * delta * scaling) + data.ctrl, -np.pi/2, np.pi/2) 
    
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


def main():
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None) # DO NOT REMOVE
    
    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()     # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    
    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore


    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    
    # Simple 3-layer neural network
    input_size = len(data.qpos) # data.qpos is the positions of the actuator motors
    hidden_size = 8 # Number of nodes in the hidden layer. Means hidden layers have 
                    # identical number of nodes
    output_size = model.nu # Number of manipulable hinges?
    
    # Get inputs, in this case the position of the actuator motors (hinges)
    
    # Initialize the network weights randomly
    W1 = np.random.randn(input_size, hidden_size) * .1
    W2 = np.random.randn(hidden_size, hidden_size) * .1
    W3 = np.random.randn(hidden_size, output_size) * .1 # 2 hidden layers
    

    # Set the control callback function
    # This is called every time step to get the next action. 
    mujoco.set_mjcb_control(lambda m,d: controller(m, d, to_track, W1, W2, W3))
    # mujoco.set_mjcb_control(None)

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    # viewer.launch(
    #     model=model,  # type: ignore
    #     data=data,
    # )
    
    # --- Simulation 
    duration = 60
    while data.time < duration:
        mujoco.mj_step(model, data)
    
    # Extract initial and final x and y position
    initial_pos = np.array([0, 0])
    target_pos = np.array(HISTORY)[-1, 0:2].flatten()
    

    
    # Calculate fitness based on euclidean distance
    d = distance_to_target(initial_pos, target_pos)


    
    # print(len(HISTORY))

    show_qpos_history(HISTORY)
    # If you want to record a video of your simulation, you can use the video renderer.

    # # Non-default VideoRecorder options
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    # video_renderer(
    #     model,
    #     data,
    #     duration=30,
    #     video_recorder=video_recorder,
    # )





# Create an fitness evaluation function
def evaluate(individual):
    mujoco.set_mjcb_control(None) # DO NOT REMOVE

    world = SimpleFlatWorld()
    gecko_core = gecko()     # DO NOT CHANGE
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    input_size = len(data.qpos) # data.qpos is the positions of the actuator motors
    output_size = model.nu # Number of manipulable hinges?
    
    # Hyperparameters
    hidden_size = 8 # Number of nodes in the hidden layer. Means hidden layers have 
                    # identical number of nodes
    # 3 Layers
    # Input 15
    # Output 8
    # Hidden 8
    W1 = np.reshape(individual[:120], shape=(input_size, hidden_size))
    W2 = np.reshape(individual[120:184], shape=(hidden_size, hidden_size))
    W3 = np.reshape(individual[184:248], shape=(hidden_size, output_size))
    
    mujoco.set_mjcb_control(lambda m,d: controller(m, d, to_track, W1, W2, W3))
    
    # --- Simulation without rendering
    duration = 60
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
    
    return (d, )
    
    
    
    

    
    

if __name__ == "__main__":
    # main()
    
    ### --- DEAP EA
    ### Just testing for now!

    import random
    from deap import base, creator, tools

    creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    IND_SIZE = 248 # For future: Set hyperparameters before and make IND_SIZE adaptible

    # Initial weight distribution
    def initial_weights():
        return random.uniform(-1, 1)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", initial_weights)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Create single individual
    ind1 = toolbox.individual()

    # Create population
    n_pop = 10
    pop = toolbox.population(n = n_pop)
    
    # Assign fitness for every individual in population
    for ind in pop:
        ind.fitness.values = evaluate(ind)
        
    # Parent selection
    parents = tools.selTournament(pop, n_pop, 4, "fitness")
    
    # Clone individuals for crossover
    offspring = [toolbox.clone(ind) for ind in parents]
    alpha = 0.5
    for i in range(0, len(offspring), 2):
        # Crossover
        tools.cxBlend(offspring[i], offspring[i+1], alpha)
        del offspring[i].fitness.values
        del offspring[i+1].fitness.values
        
        # Mutation
        tools.mutGaussian(offspring[i], mu = 0.0, sigma = 0.2, indpb = 0.2)
        tools.mutGaussian(offspring[i+1], mu = 0.0, sigma = 0.2, indpb = 0.2)
        offspring[i].fitness.values  = evaluate(offspring[i])
        offspring[i + 1].fitness.values = evaluate(offspring[i+1])

    # Survivor selection
    survivors = tools.selBest(pop + offspring, n_pop)
    
    
    for ind in pop:
        print(f"Original pop fitness: {ind.fitness.values}")
    for ind in survivors:
        print(f"Survivor fitness: {ind.fitness.values}")

    
    
      

        


