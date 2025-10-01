# Standard library
from typing import TYPE_CHECKING, Any, Sequence

import mujoco as mj

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.simulation.environments import OlympicArena


def find_in_out_size(
    robot_graph : Any, 
    SPAWN_POS : Sequence[float]
    ) -> Sequence[int]:
    """Finds out the input and output sizes, based on a robot specification

    Args:
        robot_graph (Any): The robot graph json file
        SPAWN_POS (List[float...]): Must be the global variable for Spawning position
                                    to ensure no undefined behaviors.

    Returns:
        int: Required input size for NN controller
        int: Required output size for NN controller
    """
    core_test = construct_mjspec_from_graph(robot_graph)
    
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world_test = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world_test.spawn(core_test.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    model_test = world_test.spec.compile()
    data_test = mj.MjData(model_test)
    
    # Extract input and output sizes
    input_size = len(data_test.qpos)
    output_size = model_test.nu
    
    return (input_size, output_size)
