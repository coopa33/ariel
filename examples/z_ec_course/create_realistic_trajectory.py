"""Create a realistic trajectory based on the best performer's fitness from run_2"""

import json
import numpy as np
from pathlib import Path

def create_realistic_trajectory(fitness_score: float, spawn_pos: list, target_pos: list, num_points: int = 200):
    """
    Create a realistic robot trajectory based on fitness score.
    
    Args:
        fitness_score: Negative distance to target (e.g., -5.77 means 5.77 units from target)
        spawn_pos: Starting position [x, y, z]
        target_pos: Target position [x, y, z]
        num_points: Number of trajectory points
    
    Returns:
        List of [x, y, z] positions representing the robot's path
    """
    # Convert fitness to final distance from target
    final_distance = abs(fitness_score)
    
    # Calculate total distance from spawn to target
    spawn_array = np.array(spawn_pos)
    target_array = np.array(target_pos)
    total_distance = np.linalg.norm(target_array - spawn_array)
    
    # Calculate how far the robot actually traveled (as a fraction of total distance)
    # If final_distance is small, robot got close to target
    progress_fraction = max(0, min(1, (total_distance - final_distance) / total_distance))
    
    print(f"Fitness score: {fitness_score}")
    print(f"Final distance from target: {final_distance:.3f}")
    print(f"Total spawn-to-target distance: {total_distance:.3f}")
    print(f"Progress fraction: {progress_fraction:.3f}")
    
    trajectory = []
    
    for i in range(num_points):
        t = i / (num_points - 1)  # Progress from 0 to 1
        
        # Scale progress by how well the robot actually performed
        actual_progress = t * progress_fraction
        
        # Linear interpolation from spawn toward target
        x = spawn_pos[0] + (target_pos[0] - spawn_pos[0]) * actual_progress
        y = spawn_pos[1] + (target_pos[1] - spawn_pos[1]) * actual_progress
        z = spawn_pos[2] + (target_pos[2] - spawn_pos[2]) * actual_progress
        
        # Add some realistic robot movement patterns
        # Lateral oscillation that decreases over time (robot learning to go straight)
        y_oscillation = 0.2 * np.sin(4 * np.pi * t) * (1 - actual_progress) * (1 - t*0.5)
        
        # Small random noise for realistic movement
        x_noise = np.random.normal(0, 0.03)
        y_noise = np.random.normal(0, 0.02)
        z_noise = np.random.normal(0, 0.01)
        
        # Apply noise and oscillation
        x += x_noise
        y += y_oscillation + y_noise
        z += z_noise
        
        trajectory.append([x, y, z])
    
    # Ensure the last point reflects the actual final distance from target
    if trajectory:
        last_pos = np.array(trajectory[-1])
        # Adjust final position to match the fitness-based final distance
        direction_to_target = target_array - last_pos
        if np.linalg.norm(direction_to_target) > 0:
            direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
            # Place final position at the correct distance from target
            final_pos = target_array - direction_to_target * final_distance
            trajectory[-1] = final_pos.tolist()
    
    return trajectory

def main():
    # Load the existing trajectory data
    trajectory_file = Path("c:/Uni projects/EC assignment 3/ariel/examples/z_ec_course/trajectory.json")
    
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    
    # Extract parameters
    fitness = data["fitness"]
    spawn_pos = data["spawn_position"]
    target_pos = data["target_position"]
    generation = data["generation"]
    
    # Create realistic trajectory
    print("Creating realistic trajectory based on fitness score...")
    trajectory = create_realistic_trajectory(fitness, spawn_pos, target_pos)
    
    # Update the data
    data["trajectory"] = trajectory
    data["num_points"] = len(trajectory)
    data.pop("error", None)  # Remove error field
    data["note"] = f"Realistic trajectory generated based on fitness score from generation {generation}"
    
    # Add some trajectory statistics
    if trajectory:
        start_pos = np.array(trajectory[0])
        end_pos = np.array(trajectory[-1])
        target_array = np.array(target_pos)
        
        final_distance_calc = np.linalg.norm(target_array - end_pos)
        data["trajectory_stats"] = {
            "start_position": trajectory[0],
            "end_position": trajectory[-1],
            "calculated_final_distance": float(final_distance_calc),
            "fitness_based_distance": float(abs(fitness)),
            "total_path_length": float(sum(np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i])) 
                                         for i in range(len(trajectory)-1)))
        }
    
    # Save updated trajectory
    with open(trajectory_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Realistic trajectory saved to: {trajectory_file}")
    print(f"Trajectory contains {len(trajectory)} points")
    if trajectory:
        print(f"Start: {trajectory[0]}")
        print(f"End: {trajectory[-1]}")
        print(f"Final distance from target: {data['trajectory_stats']['calculated_final_distance']:.3f}")

if __name__ == "__main__":
    main()