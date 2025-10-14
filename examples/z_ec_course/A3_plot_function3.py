# Standard library
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np

# Local libraries
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer

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
SPAWN_POS = [-0.8, 0, 0]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]


def show_xpos_history(history: list[float]) -> None:
    """Show robot path as a line overlaid on the MuJoCo arena background."""
    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena(
        load_precompiled=False,
    )

    # Add reference objects to the world (spawn, target positions)
    target_box = r"""
    <mujoco>
        <worldbody>
            <geom name="magenta_box"
                size=".1 .1 .1"
                type="box"
                rgba="1 0 1 0.75"/>
        </worldbody>
    </mujoco>
    """
    spawn_box = r"""
    <mujoco>
        <worldbody>
            <geom name="gray_box"
            size=".1 .1 .1"
            type="box"
            rgba="0.5 0.5 0.5 0.5"/>
        </worldbody>
    </mujoco>
    """
    
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Add reference positions to the world
    adjustment = np.array((0, 0, TARGET_POSITION[2] + 1))
    
    # Target position
    world.spawn(
        mj.MjSpec.from_string(target_box),
        position=TARGET_POSITION + adjustment,
        correct_collision_with_floor=False,
    )

    # Spawn position of robot
    world.spawn(
        mj.MjSpec.from_string(spawn_box),
        position=SPAWN_POS,
        correct_collision_with_floor=False,
    )

    # Render the background
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        save_path=save_path,
        save=True,
        width=600,
        height=800,
        fovy=8,
    )

    # Setup background image
    img = plt.imread(save_path)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(img)

    # Now overlay the robot path as a line
    # Convert world coordinates to image coordinates
    x_coords = pos_data[:, 0]
    y_coords = pos_data[:, 1]
    
    print(f"Robot path X range: {x_coords.min():.2f} to {x_coords.max():.2f}")
    print(f"Robot path Y range: {y_coords.min():.2f} to {y_coords.max():.2f}")
    
    img_height, img_width = img.shape[:2]
    print(f"Image size: {img_width} x {img_height}")
    
    # More conservative coordinate mapping based on actual robot path
    # Use the actual min/max of the robot path with some padding
    path_x_min, path_x_max = x_coords.min() - 0.5, x_coords.max() + 0.5
    path_y_min, path_y_max = y_coords.min() - 0.5, y_coords.max() + 0.5
    
    # Map coordinates to use most of the image area
    x_scale = (img_width * 0.8) / (path_x_max - path_x_min)  # Use 80% of width
    y_scale = (img_height * 0.8) / (path_y_max - path_y_min)  # Use 80% of height
    
    # Center the path in the image
    x_offset = img_width * 0.1  # 10% margin on left
    y_offset = img_height * 0.1  # 10% margin on top
    
    # Convert to image coordinates
    img_x = (x_coords - path_x_min) * x_scale + x_offset
    img_y = img_height - ((y_coords - path_y_min) * y_scale + y_offset)  # Flip Y axis
    
    print(f"Image X range: {img_x.min():.0f} to {img_x.max():.0f}")
    print(f"Image Y range: {img_y.min():.0f} to {img_y.max():.0f}")
    
    # Ensure coordinates are within bounds (safety check)
    img_x = np.clip(img_x, 0, img_width - 1)
    img_y = np.clip(img_y, 0, img_height - 1)
    
    # Plot the path line with thinner width
    ax.plot(img_x, img_y, 'yellow', linewidth=2, alpha=0.9, label='Robot Path')
    
    # Mark start and end points with smaller markers
    ax.plot(img_x[0], img_y[0], 'go', markersize=8, label='Start Position', markeredgecolor='white', markeredgewidth=2)
    ax.plot(img_x[-1], img_y[-1], 'ro', markersize=8, label='End Position', markeredgecolor='white', markeredgewidth=2)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))

    # Remove axis ticks but keep labels
    ax.set_xlabel("Arena View")
    ax.set_ylabel("Top-Down Perspective")
    ax.set_xticks([])
    ax.set_yticks([])

    # Title
    plt.title("Robot Path in MuJoCo Arena")

    # Save the plot
    save_path_plot = DATA / "robot_path_arena.png"
    fig.savefig(save_path_plot, dpi=300, bbox_inches='tight')
    print(f"Arena path plot saved to: {save_path_plot}")

    # Show results
    plt.show()

if __name__ == "__main__":
    # For now, use sample trajectory data that represents a typical robot path from run_2
    # This simulates movement from spawn position towards target
    # You can replace this with actual extracted data later
    
    print("Using sample trajectory data for run_2 visualization")
    
    # Create a realistic trajectory from spawn to target
    start_pos = [-0.8, 0, 0]  # SPAWN_POS
    target_pos = [5, 0, 0.5]  # TARGET_POSITION
    
    # Generate a trajectory with some realistic robot movement
    n_steps = 200
    position_history = []
    
    for i in range(n_steps):
        t = i / (n_steps - 1)  # Progress from 0 to 1
        
        # Linear interpolation with some noise for realistic movement
        x = start_pos[0] + (target_pos[0] - start_pos[0]) * t
        # Add some lateral movement (y direction) to make it more realistic
        y = start_pos[1] + 0.3 * np.sin(3 * np.pi * t) * (1 - t)  # Oscillation that dampens
        z = start_pos[2] + (target_pos[2] - start_pos[2]) * t
        
        # Add some random noise for realism
        x += np.random.normal(0, 0.05)
        y += np.random.normal(0, 0.02)
        z += np.random.normal(0, 0.01)
        
        position_history.append([x, y, z])
    
    print(f"Generated sample trajectory with {len(position_history)} points")
    print(f"Start position: {position_history[0]}")
    print(f"End position: {position_history[-1]}")
    
    # Save this sample data for future use
    import json
    sample_file = Path("__data__/A3_modified/run_2_sample_trajectory.json")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_file, 'w') as f:
        json.dump(position_history, f, indent=2)
    print(f"Saved sample trajectory to: {sample_file}")
    
    show_xpos_history(position_history)