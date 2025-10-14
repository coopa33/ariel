"""Extract trajectory data from run_2 best performer and save to trajectory.json"""

import pickle
import json
import numpy as np
from pathlib import Path
import mujoco as mj
from typing import Any

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker

def find_latest_generation(run_path: Path) -> int:
    """Find the latest generation in the run directory."""
    generations = []
    for gen_dir in run_path.iterdir():
        if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
            try:
                gen_num = int(gen_dir.name.split("_")[1])
                # Check if both required files exist
                if (gen_dir / "body_population.pkl").exists() and (gen_dir / "best_performers.pkl").exists():
                    generations.append(gen_num)
            except (ValueError, IndexError):
                continue
    return max(generations) if generations else -1

def load_best_performer(run_path: Path, generation: int):
    """Load the best performer from a specific generation."""
    gen_dir = run_path / f"generation_{generation:03d}"
    
    # Load best performers data
    with open(gen_dir / "best_performers.pkl", "rb") as f:
        best_data = pickle.load(f)
    
    return best_data

def simulate_robot_and_get_trajectory(body_genotype, brain_genotype, sim_config):
    """Simulate the robot and return its trajectory."""
    # Decode the genotypes
    decoder = HighProbabilityDecoder()
    body_graph = decoder.decode(body_genotype)
    
    # Construct robot specification
    robot_spec = construct_mjspec_from_graph(body_graph)
    
    # Decode brain
    nde = NeuralDevelopmentalEncoding()
    brain_graph = nde.decode(brain_genotype)
    
    # Setup controller with tracker
    tracker = Tracker(track_xpos=True, track_base_velocity=True)
    controller = Controller(tracker=tracker)
    
    # Initialize world and spawn robot
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(robot_spec, position=sim_config.spawn_position.copy())
    
    # Generate model and data
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    
    # Setup tracker
    tracker.setup(world.spec, data)
    
    # Set control callback (dummy for now)
    def dummy_control(m, d):
        pass
    
    mj.set_mjcb_control(dummy_control)
    
    # Run simulation
    simple_runner(model, data, duration=15)
    
    # Extract trajectory
    trajectory = tracker.history["xpos"]
    return trajectory

def main():
    # Configuration similar to A3_modified.py
    class SimConfig:
        def __init__(self):
            self.spawn_position = [-0.8, 0, 0]
            self.target_position = [5, 0, 0.5]
            self.data = Path("__data__")
    
    sim_config = SimConfig()
    
    # Find run_2 directory
    run_2_path = Path("c:/Uni projects/EC assignment 3/ariel/examples/A3_modified/run_2")
    
    if not run_2_path.exists():
        print(f"Run directory not found: {run_2_path}")
        return
    
    # Find latest generation
    latest_gen = find_latest_generation(run_2_path)
    if latest_gen == -1:
        print("No valid generations found in run_2")
        return
    
    print(f"Found latest generation: {latest_gen}")
    
    # Load best performer
    best_data = load_best_performer(run_2_path, latest_gen)
    print(f"Loaded best performer data: {list(best_data.keys())}")
    
    # Extract best genotypes
    best_body = best_data["best_body_genotype"]
    best_brain = best_data["best_brain_genotype"]
    best_fitness = best_data["body_fitness"]
    
    print(f"Best fitness: {best_fitness}")
    print(f"Body genotype type: {type(best_body)}")
    print(f"Brain genotype type: {type(best_brain)}")
    
    try:
        # Simulate and get trajectory
        print("Running simulation to extract trajectory...")
        trajectory = simulate_robot_and_get_trajectory(best_body, best_brain, sim_config)
        
        print(f"Trajectory extracted: {len(trajectory)} points")
        print(f"Start position: {trajectory[0]}")
        print(f"End position: {trajectory[-1]}")
        
        # Convert numpy arrays to lists for JSON serialization
        trajectory_list = []
        for pos in trajectory:
            if isinstance(pos, np.ndarray):
                trajectory_list.append(pos.tolist())
            else:
                trajectory_list.append(list(pos))
        
        # Save trajectory to JSON
        output_file = Path("c:/Uni projects/EC assignment 3/ariel/examples/z_ec_course/trajectory.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        trajectory_data = {
            "generation": latest_gen,
            "fitness": float(best_fitness),
            "trajectory": trajectory_list,
            "spawn_position": sim_config.spawn_position,
            "target_position": sim_config.target_position,
            "num_points": len(trajectory_list)
        }
        
        with open(output_file, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"Trajectory saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        print("Attempting to save available data without simulation...")
        
        # Save what we can without simulation
        output_file = Path("c:/Uni projects/EC assignment 3/ariel/examples/z_ec_course/trajectory.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        trajectory_data = {
            "generation": latest_gen,
            "fitness": float(best_fitness),
            "trajectory": [],  # Empty trajectory if simulation failed
            "spawn_position": sim_config.spawn_position,
            "target_position": sim_config.target_position,
            "error": str(e),
            "note": "Simulation failed, trajectory not available"
        }
        
        with open(output_file, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"Partial data saved to: {output_file}")

if __name__ == "__main__":
    main()