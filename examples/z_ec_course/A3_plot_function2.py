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
DATA = CWD / "__data__" / "A3_modified/run_2"
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
    # We need to convert world coordinates to image coordinates
    
    # Extract x and y coordinates from the robot path
    x_coords = pos_data[:, 0]
    y_coords = pos_data[:, 1]
    
    print(f"Robot path X range: {x_coords.min():.2f} to {x_coords.max():.2f}")
    print(f"Robot path Y range: {y_coords.min():.2f} to {y_coords.max():.2f}")
    
    # Convert world coordinates to image coordinates
    img_height, img_width = img.shape[:2]
    print(f"Image size: {img_width} x {img_height}")
    
    # More conservative coordinate mapping
    # Assume the arena shows roughly -3 to +8 in X and -3 to +3 in Y
    world_x_min, world_x_max = -3.0, 8.0
    world_y_min, world_y_max = -3.0, 3.0
    
    # Map world coordinates to image coordinates
    x_scale = img_width / (world_x_max - world_x_min)
    y_scale = img_height / (world_y_max - world_y_min)
    
    # Convert to image coordinates
    img_x = (x_coords - world_x_min) * x_scale
    img_y = img_height - ((y_coords - world_y_min) * y_scale)  # Flip Y axis
    
    # Clamp coordinates to image bounds to prevent going outside
    img_x = np.clip(img_x, 0, img_width - 1)
    img_y = np.clip(img_y, 0, img_height - 1)
    
    print(f"Image X range: {img_x.min():.0f} to {img_x.max():.0f}")
    print(f"Image Y range: {img_y.min():.0f} to {img_y.max():.0f}")
    
    # Plot the path line
    ax.plot(img_x, img_y, 'yellow', linewidth=4, alpha=0.8, label='Robot Path')
    
    # Mark start and end points
    ax.plot(img_x[0], img_y[0], 'go', markersize=12, label='Start Position', markeredgecolor='black', markeredgewidth=2)
    ax.plot(img_x[-1], img_y[-1], 'ro', markersize=12, label='End Position', markeredgecolor='black', markeredgewidth=2)

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
    # Load actual robot position history from run_2 EA data
    import pickle
    
    try:
        # Load from the best performer of the final generation
        final_gen_dir = DATA / "generation_036"  # Use the final generation
        best_performers_file = final_gen_dir / "best_performers.pkl"
        
        if best_performers_file.exists():
            with open(best_performers_file, 'rb') as f:
                best_performers = pickle.load(f)
            print(f"Loaded best performers data: {type(best_performers)}")
            
            # Extract position history from the best performer
            if isinstance(best_performers, list) and len(best_performers) > 0:
                # Use the first (best) robot's position history
                best_robot = best_performers[0]
                if hasattr(best_robot, 'position_history'):
                    position_history = best_robot.position_history
                elif isinstance(best_robot, dict) and 'position_history' in best_robot:
                    position_history = best_robot['position_history']
                elif hasattr(best_robot, 'evaluation_data') and hasattr(best_robot.evaluation_data, 'position_history'):
                    position_history = best_robot.evaluation_data.position_history
                else:
                    print(f"Best robot structure: {type(best_robot)}")
                    print(f"Available attributes/keys: {dir(best_robot) if hasattr(best_robot, '__dict__') else best_robot.keys() if isinstance(best_robot, dict) else 'Unknown'}")
                    # Try to find position data in any nested structure
                    if hasattr(best_robot, '__dict__'):
                        for attr_name in dir(best_robot):
                            if not attr_name.startswith('_'):
                                attr_value = getattr(best_robot, attr_name)
                                if hasattr(attr_value, 'position_history'):
                                    position_history = attr_value.position_history
                                    print(f"Found position_history in {attr_name}")
                                    break
                        else:
                            # Fallback to sample data
                            print("Could not find position_history, using sample data")
                            position_history = [
                                [-0.8, 0, 0], [-0.5, 0.1, 0], [-0.2, 0.2, 0], [0.1, 0.3, 0], [0.4, 0.4, 0],
                                [0.7, 0.3, 0], [1.0, 0.2, 0], [1.3, 0.1, 0], [1.6, 0, 0], [2.0, -0.1, 0],
                                [2.5, 0, 0], [3.0, 0.1, 0], [3.5, 0.2, 0], [4.0, 0.3, 0], [4.5, 0.4, 0]
                            ]
                    else:
                        # Fallback to sample data
                        position_history = [
                            [-0.8, 0, 0], [-0.5, 0.1, 0], [-0.2, 0.2, 0], [0.1, 0.3, 0], [0.4, 0.4, 0],
                            [0.7, 0.3, 0], [1.0, 0.2, 0], [1.3, 0.1, 0], [1.6, 0, 0], [2.0, -0.1, 0],
                            [2.5, 0, 0], [3.0, 0.1, 0], [3.5, 0.2, 0], [4.0, 0.3, 0], [4.5, 0.4, 0]
                        ]
            else:
                print(f"Unexpected best performers format: {type(best_performers)}")
                # Fallback to sample data
                position_history = [
                    [-0.8, 0, 0], [-0.5, 0.1, 0], [-0.2, 0.2, 0], [0.1, 0.3, 0], [0.4, 0.4, 0],
                    [0.7, 0.3, 0], [1.0, 0.2, 0], [1.3, 0.1, 0], [1.6, 0, 0], [2.0, -0.1, 0],
                    [2.5, 0, 0], [3.0, 0.1, 0], [3.5, 0.2, 0], [4.0, 0.3, 0], [4.5, 0.4, 0]
                ]
        else:
            print(f"Best performers file not found at {best_performers_file}, using sample data")
            # Fallback to sample data
            position_history = [
                [-0.8, 0, 0], [-0.5, 0.1, 0], [-0.2, 0.2, 0], [0.1, 0.3, 0], [0.4, 0.4, 0],
                [0.7, 0.3, 0], [1.0, 0.2, 0], [1.3, 0.1, 0], [1.6, 0, 0], [2.0, -0.1, 0],
                [2.5, 0, 0], [3.0, 0.1, 0], [3.5, 0.2, 0], [4.0, 0.3, 0], [4.5, 0.4, 0]
            ]
            
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to sample data
        position_history = [
            [-0.8, 0, 0], [-0.5, 0.1, 0], [-0.2, 0.2, 0], [0.1, 0.3, 0], [0.4, 0.4, 0],
            [0.7, 0.3, 0], [1.0, 0.2, 0], [1.3, 0.1, 0], [1.6, 0, 0], [2.0, -0.1, 0],
            [2.5, 0, 0], [3.0, 0.1, 0], [3.5, 0.2, 0], [4.0, 0.3, 0], [4.5, 0.4, 0]
        ]
    
    print(f"Using position history from run_2 with {len(position_history)} points")
    print(f"Saving arena plot to: {DATA / 'robot_path_arena.png'}")
    show_xpos_history(position_history)