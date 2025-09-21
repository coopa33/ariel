<<<<<<< HEAD
from .gate_learning import xy_displacement, x_speed, y_speed
=======
from .gate_learning import x_speed, xy_displacement, y_speed
>>>>>>> 34ddee643458582126abaec42c288599ccd9e640
from .targeted_locomotion import distance_to_target
from .turning_in_place import turning_in_place

tasks = [
    "Gate Learning",
    "Targeted Locomotion",
    "Turning In Place"]

_task_fitness_function_map_ = {
    "Gate Learning": [xy_displacement, x_speed, y_speed],
    "Targeted Locomotion": [distance_to_target],
    "Turning In Place": [turning_in_place],
}
