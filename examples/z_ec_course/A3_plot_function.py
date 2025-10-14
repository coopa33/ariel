"""Assignment 3 template code."""

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
    """Show robot path overlaid on the arena background with horizontal orientation."""
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
        position=SPAWN_POS + adjustment,
        correct_collision_with_floor=False,
    )

    # Generate the arena background with horizontal orientation
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        save_path=save_path,
        save=True,
        width=1000,  # Wider for horizontal view
        height=600,  # Shorter for horizontal view
        fovy=8,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img)

    # Convert trajectory to image coordinates
    trajectory_array = np.array(history)
    x_coords = trajectory_array[:, 0]
    y_coords = trajectory_array[:, 1]
    
    # Debug: Print coordinate ranges
    print(f"Trajectory X range: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
    print(f"Trajectory Y range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
    print(f"Trajectory start: ({x_coords[0]:.3f}, {y_coords[0]:.3f})")
    print(f"Trajectory end: ({x_coords[-1]:.3f}, {y_coords[-1]:.3f})")
    
    # Set world bounds based on actual arena size
    world_bounds = [-1.5, 6.0, -2.5, 2.5]  # [x_min, x_max, y_min, y_max]
    img_height, img_width = img.shape[:2]
    
    # Convert to image coordinates
    # Note: Image Y axis is flipped (0 at top), so we need to flip Y mapping
    x_img = (x_coords - world_bounds[0]) / (world_bounds[1] - world_bounds[0]) * img_width
    y_img = img_height - (y_coords - world_bounds[2]) / (world_bounds[3] - world_bounds[2]) * img_height
    
    # Debug: Print image coordinate ranges
    print(f"Image X range: [{x_img.min():.1f}, {x_img.max():.1f}] (image width: {img_width})")
    print(f"Image Y range: [{y_img.min():.1f}, {y_img.max():.1f}] (image height: {img_height})")
    
    # Plot trajectory as a line
    ax.plot(x_img, y_img, 'b-', linewidth=3, alpha=0.9, label='Robot Path')
    
    # Mark start and end points
    ax.plot(x_img[0], y_img[0], 'go', markersize=10, label='Start Position')
    ax.plot(x_img[-1], y_img[-1], 'ro', markersize=10, label='End Position')

    # Add legend to the plot
    plt.rc("legend", fontsize="medium")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    # Add labels and title
    ax.set_xlabel("Arena X Coordinate")
    ax.set_ylabel("Arena Y Coordinate")
    ax.set_title("Robot Trajectory in Arena (Horizontal View)")
    
    # Remove axis ticks for cleaner look
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create a realistic trajectory from spawn to target
    start_pos = SPAWN_POS.copy()  # Use the same spawn position as the global constant
    target_pos = TARGET_POSITION.copy()  # Use the same target position as the global constant

    
    print(f"Start position (spawn): {start_pos}")
    print(f"Target position (finish): {target_pos}")
    
    # Generate a simple linear trajectory
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
    print(f"First position (should be near spawn): {position_history[0]}")
    print(f"Last position (should be near target): {position_history[-1]}")
    #  Save this sample data for future use
    import json
    sample_file = Path("examples/z_ec_course/trajectory.json")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_file, 'w') as f:
        json.dump(position_history, f, indent=2)
    print(f"Saved sample trajectory to: {sample_file}")
    
    show_xpos_history(position_history)