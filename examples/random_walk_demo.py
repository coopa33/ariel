"""
Demo script to run a random walk using the existing rw_controller from A3_modified.py
"""

import sys
from pathlib import Path

# Add the current directory to path to import from A3_modified
sys.path.insert(0, str(Path(__file__).parent))

from A3_modified import (
    EAConfig, 
    rw_controller, 
    experiment, 
    create_robot_graph,
    NeuralDevelopmentalEncoding,
    construct_mjspec_from_graph,
    Controller,
    Tracker,
    decode_body_genotype
)
import mujoco as mj
import numpy as np
import random

def run_random_walk_demo():
    """
    Demonstrate a random walk with a randomly generated robot body.
    """
    print("=== Random Walk Demo ===")
    
    # Setup configuration
    sim_config = EAConfig(rng_seed=42)
    
    # Set seeds for reproducibility
    random.seed(sim_config.rng_seed)
    np.random.seed(sim_config.rng_seed)
    
    print("1. Generating random robot body...")
    
    # Generate a random body genotype (3*64 = 192 values between 0 and 1)
    body_genotype_size = 3 * 64
    body_genotype = [random.random() for _ in range(body_genotype_size)]
    
    print(f"   Generated body genotype with {len(body_genotype)} parameters")
    
    # Create robot graph from the body genotype
    print("2. Creating robot graph...")
    robot_graph = create_robot_graph(body_genotype, sim_config)
    
    # Convert to robot specification
    print("3. Building robot specification...")
    robot_spec = construct_mjspec_from_graph(robot_graph)
    
    # Setup tracker and controller
    print("4. Setting up controller and tracker...")
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    
    # Use the rw_controller (random walk controller)
    ctrl = Controller(
        controller_callback_function=rw_controller,
        tracker=tracker
    )
    
    # Run the experiment
    print("5. Running random walk simulation...")
    print("   Mode: 'simple' (no visualization, fastest)")
    print("   Duration: 15 seconds")
    
    experiment(
        robot=robot_spec,
        controller=ctrl,
        matrices=None,  # rw_controller generates its own random matrices
        sim_config=sim_config,
        duration=15,
        mode="simple"  # Change to "launcher" for visualization
    )
    
    # Analyze results
    print("\n6. Results:")
    if tracker.history and "xpos" in tracker.history:
        positions = tracker.history["xpos"][0]
        start_pos = positions[0]
        end_pos = positions[-1]
        
        distance_moved = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        
        print(f"   Start position: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
        print(f"   End position:   ({end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f})")
        print(f"   Distance moved: {distance_moved:.3f} units")
        print(f"   Total frames recorded: {len(positions)}")
        
        # Calculate average velocity
        time_duration = 15  # seconds
        avg_velocity = distance_moved / time_duration
        print(f"   Average velocity: {avg_velocity:.3f} units/second")
        
        if distance_moved > 0.1:
            print("   ✅ Robot moved successfully!")
        else:
            print("   ⚠️ Robot barely moved (might be unstable body)")
    else:
        print("   ❌ No tracking data available")

def run_random_walk_with_visualization():
    """
    Same as above but with visualization (opens MuJoCo viewer).
    """
    print("=== Random Walk Demo with Visualization ===")
    
    sim_config = EAConfig(rng_seed=42)
    random.seed(sim_config.rng_seed)
    np.random.seed(sim_config.rng_seed)
    
    # Generate random body
    body_genotype = [random.random() for _ in range(3 * 64)]
    robot_graph = create_robot_graph(body_genotype, sim_config)
    robot_spec = construct_mjspec_from_graph(robot_graph)
    
    # Setup controller
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=rw_controller, tracker=tracker)
    
    print("Opening MuJoCo viewer...")
    print("Close the viewer window when you're done observing.")
    
    # Run with visualization
    experiment(
        robot=robot_spec,
        controller=ctrl,
        matrices=None,
        sim_config=sim_config,
        duration=30,  # Longer duration for observation
        mode="launcher"  # Opens MuJoCo viewer
    )

def run_multiple_random_walks(num_trials=5):
    """
    Run multiple random walks and compare their performance.
    """
    print(f"=== Running {num_trials} Random Walk Trials ===")
    
    sim_config = EAConfig(rng_seed=42)
    results = []
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Use different seed for each trial
        trial_seed = sim_config.rng_seed + trial
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        
        # Generate random body
        body_genotype = [random.random() for _ in range(3 * 64)]
        
        try:
            robot_graph = create_robot_graph(body_genotype, sim_config)
            robot_spec = construct_mjspec_from_graph(robot_graph)
            
            tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
            ctrl = Controller(controller_callback_function=rw_controller, tracker=tracker)
            
            experiment(
                robot=robot_spec,
                controller=ctrl,
                matrices=None,
                sim_config=sim_config,
                duration=15,
                mode="simple"
            )
            
            # Calculate performance
            if tracker.history and "xpos" in tracker.history:
                positions = tracker.history["xpos"][0]
                start_pos = np.array(positions[0])
                end_pos = np.array(positions[-1])
                distance_moved = np.linalg.norm(end_pos - start_pos)
                
                results.append({
                    'trial': trial + 1,
                    'seed': trial_seed,
                    'distance': distance_moved,
                    'start_pos': positions[0],
                    'end_pos': positions[-1]
                })
                
                print(f"   Distance moved: {distance_moved:.3f}")
            else:
                print(f"   Failed to track robot")
                results.append({
                    'trial': trial + 1,
                    'seed': trial_seed,
                    'distance': 0.0,
                    'start_pos': None,
                    'end_pos': None
                })
                
        except Exception as e:
            print(f"   Error: {e}")
            results.append({
                'trial': trial + 1,
                'seed': trial_seed,
                'distance': 0.0,
                'start_pos': None,
                'end_pos': None
            })
    
    # Summary
    print(f"\n=== Summary of {num_trials} Trials ===")
    distances = [r['distance'] for r in results if r['distance'] > 0]
    
    if distances:
        print(f"Successful trials: {len(distances)}/{num_trials}")
        print(f"Average distance: {np.mean(distances):.3f}")
        print(f"Best distance: {np.max(distances):.3f}")
        print(f"Worst distance: {np.min(distances):.3f}")
        print(f"Standard deviation: {np.std(distances):.3f}")
        
        best_trial = max(results, key=lambda x: x['distance'])
        print(f"\nBest performing trial: #{best_trial['trial']} (seed: {best_trial['seed']})")
    else:
        print("No successful trials!")

if __name__ == "__main__":
    print("Random Walk Demo Options:")
    print("1. Basic random walk (no visualization)")
    print("2. Random walk with MuJoCo viewer")
    print("3. Multiple random walk comparison")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_random_walk_demo()
        elif choice == "2":
            run_random_walk_with_visualization()
        elif choice == "3":
            num_trials = input("How many trials? (default: 5): ").strip()
            num_trials = int(num_trials) if num_trials else 5
            run_multiple_random_walks(num_trials)
        else:
            print("Invalid choice, running basic demo...")
            run_random_walk_demo()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()