"""Generate position history by simulating the best robot from run_2."""

import pickle
from pathlib import Path
import numpy as np
import json

# Import necessary modules for simulation
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

def simulate_best_robot_from_run(run_id=2, generation=36, data_path="__data__/A3_modified"):
    """Simulate the best robot from a specific generation and extract position history."""
    
    try:
        # Import simulation modules
        from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding
        from ariel.body_phenotypes.robogen_lite.body_phenotype import BodyPhenotype
        from ariel.simulation.environments import OlympicArena
        from ariel.simulation.tasks import LocomotionTask
        from ariel.simulation.controllers import NNController
        import mujoco as mj
        
        data_dir = Path(data_path)
        gen_dir = data_dir / f"run_{run_id}" / f"generation_{generation:03d}"
        
        if not gen_dir.exists():
            print(f"Generation {generation} data not found at {gen_dir}")
            return None
        
        # Load best performers
        best_performers_file = gen_dir / "best_performers.pkl"
        
        with open(best_performers_file, 'rb') as f:
            best_performers = pickle.load(f)
        
        print(f"Loaded best performers for generation {generation}")
        
        # Extract the best robot data
        best_body_genotype = best_performers['best_body_genotype']
        best_brain_genotype = best_performers['best_brain_genotype']
        nde = best_performers['nde']
        
        print(f"Body genotype length: {len(best_body_genotype)}")
        print(f"Brain genotype length: {len(best_brain_genotype)}")
        
        # Create body phenotype
        body_phenotype = BodyPhenotype()
        body_phenotype.develop(best_body_genotype, nde)
        
        print(f"Body developed successfully")
        
        # Create the simulation environment
        world = OlympicArena(load_precompiled=False)
        
        # Create and add the robot to the world
        robot_spec = body_phenotype.to_mjspec()
        spawn_position = [-0.8, 0, 0]  # Starting position
        robot_actor = world.spawn(
            robot_spec,
            position=spawn_position,
            correct_collision_with_floor=True
        )
        
        print(f"Robot spawned successfully")
        
        # Create brain/controller
        controller = NNController()
        controller.develop(best_brain_genotype, nde, body_phenotype)
        
        # Set up the simulation
        model = world.spec.compile()
        data = mj.MjData(model)
        
        # Create locomotion task
        task = LocomotionTask(
            target_position=[5, 0, 0.5],
            target_radius=0.5,
            duration=10.0,
            time_step=0.02
        )
        
        print(f"Starting simulation...")
        
        # Run simulation and collect position history
        position_history = []
        time = 0.0
        time_step = 0.02
        max_time = 10.0
        
        mj.mj_resetData(model, data)
        
        while time < max_time:
            # Get robot position
            robot_pos = data.qpos[:3].copy()  # x, y, z position
            position_history.append(robot_pos.tolist())
            
            # Get sensor data for controller
            sensor_data = controller.get_sensor_data(data)
            
            # Get control signals from brain
            control_signals = controller.step(sensor_data)
            
            # Apply control signals
            if control_signals is not None:
                data.ctrl[:len(control_signals)] = control_signals
            
            # Step simulation
            mj.mj_step(model, data)
            time += time_step
        
        print(f"Simulation completed. Collected {len(position_history)} position points")
        print(f"Start position: {position_history[0]}")
        print(f"End position: {position_history[-1]}")
        
        return position_history
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Generating position history by simulating run_2 best robot...")
    position_history = simulate_best_robot_from_run(run_id=2, generation=36)
    
    if position_history is not None:
        # Save the position history
        output_file = Path("__data__/A3_modified/run_2_position_history.json")
        with open(output_file, 'w') as f:
            json.dump(position_history, f, indent=2)
        print(f"Saved position history to: {output_file}")
    else:
        print("Failed to generate position history")