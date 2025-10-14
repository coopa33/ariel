"""Test the extracted trajectory with our arena visualization function"""

import json
from pathlib import Path
from examples.z_ec_course.A3_plot_function4 import show_xpos_history

def main():
    # Load the trajectory data
    trajectory_file = Path("trajectory.json")
    
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    
    trajectory = data["trajectory"]
    fitness = data["fitness"]
    generation = data["generation"]
    
    print(f"Loaded trajectory from generation {generation}")
    print(f"Fitness: {fitness}")
    print(f"Number of points: {len(trajectory)}")
    print(f"Start position: {trajectory[0]}")
    print(f"End position: {trajectory[-1]}")
    
    # Visualize the trajectory
    print("Visualizing trajectory in arena...")
    show_xpos_history(trajectory)

if __name__ == "__main__":
    main()