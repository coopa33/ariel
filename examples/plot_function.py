import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import uuid
from typing import Tuple, List

def save_unique_png(fig, path="__data__/A3_modified", ext=".png"):
    """Save figure with unique filename to avoid overwrites"""
    Path(path).mkdir(parents=True, exist_ok=True)
    
    # Create unique filename
    base_name = "fitness_plot"
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{base_name}_{unique_id}{ext}"
    
    full_path = Path(path) / filename
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {full_path}")
    plt.show()


def load_population_from_generation(generation, run_id=0, data_path="__data__/A3_modified"):
    """
    Load saved generation data, by specifying which run and generation to load.

    Args:
        generation (int):           The generation number to load
        run_id (int):               The run number to load (default: 0)
        data_path (str):            Path to the data directory (default: "__data__")
        
    Returns:
        population (list[DEAP lists]): The population loaded from the generation
        best_data (dict):              The relevant data for the best performing individual
                                        of that generation.
    """
    data_dir = Path(data_path)
    gen_dir = data_dir / f"run_{run_id}" / f"generation_{generation:03d}"
    
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generation {generation} data not found at {gen_dir}")
    
    # Load population
    with open(gen_dir / "body_population.pkl", "rb") as f:
        population = pickle.load(f)
    
    # Load best data
    with open(gen_dir / "best_performers.pkl", "rb") as f:
        best_data = pickle.load(f)
    
    # Print statement
    print(f"Resumed from generation {generation} with {len(population)} individuals")
    print(f"Best fitness from that generation: {best_data.get('body_fitness', 'Unknown')}")
    return population, best_data

# def plot_run_statistics(run_id, data_path="__data__/A3_modified"):
#     """
#     Plot fitness statistics for a specific run.
    
#     Args:
#         run_id (int): The run ID to plot
#         data_path (str): Path to the data directory (default: "__data__")
#     """
#     data_dir = Path(data_path)
#     run_dir = data_dir / f"run_{run_id}"
    
#     if not run_dir.exists():
#         raise FileNotFoundError(f"Run directory {run_dir} does not exist.")
    
#     run_means = []
#     run_stds = []
#     run_bests = []
#     generations = []
    
#     for gen_dir in run_dir.iterdir():
#         if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
#             gen_num = int(gen_dir.name.split("_")[1])
#             try:
#                 pop, best_data = load_population_from_generation(gen_num, run_id, data_path)
#                 # Extract fitness values from population - these should be negative distance values
#                 fitness_values = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
#                 mean = np.mean(fitness_values)
#                 std = np.std(fitness_values)
                
#                 # Use the fitness from best_data directly
#                 best = best_data.get("body_fitness", None)
                
#                 # Debug print to see what values we're getting
#                 print(f"Gen {gen_num}: Pop fitness range: {min(fitness_values):.3f} to {max(fitness_values):.3f}, Best: {best}")
                
#                 generations.append(gen_num)
#                 run_means.append(mean)
#                 run_stds.append(std)
#                 run_bests.append(best)
#             except Exception as e:
#                 print(f"Warning: Could not load generation {gen_num}: {e}")
    
#     if not generations:
#         print(f"No valid generation data found for run {run_id}")
#         return
    
#     # Sort by generation number
#     sorted_data = sorted(zip(generations, run_means, run_stds, run_bests))
#     generations, run_means, run_stds, run_bests = zip(*sorted_data)
    
#     run_means = np.array(run_means)
#     run_stds = np.array(run_stds)
#     run_bests = np.array(run_bests)
#     x = np.array(generations)
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(x, run_means, linestyle="--", linewidth=1.5, color="blue", label="Mean Fitness")
#     ax.fill_between(x, run_means - run_stds, run_means + run_stds, color="blue", alpha=0.2, label="Std. Dev.")
#     ax.plot(x, run_bests, linestyle="-", linewidth=2, color="red", label="Best Fitness")
#     ax.set_xlabel("Generation")
#     ax.set_ylabel("Fitness")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.title(f"Fitness over Generations of Best Run")
    
#     # Create save directory
#     save_path = run_dir / "plots"
#     save_unique_png(fig, path=str(save_path), ext=".png")

def load_random_walk_data(rw_id: int, data_path: str = "__data__/A3_modified"):
    """
    Load random walk data from all generations.
    
    Args:
        rw_id: The random walk ID to load (e.g., 2 for rw_2)
        data_path: Path to the data directory
        
    Returns:
        tuple: (generations, fitness_values) where fitness_values is list of fitness per generation
    """
    data_dir = Path(data_path)
    rw_dir = data_dir / f"rw_{rw_id}"
    
    if not rw_dir.exists():
        raise FileNotFoundError(f"Random walk directory {rw_dir} does not exist.")
    
    generations = []
    fitness_values = []
#     fitness = [
#     -4.615587450539792,
#     -4.426138259894928,
#     -4.635487971474094,
#     -4.256202428618411,
#     -4.3892039721983425,
#     -4.284932330792765,
#     -4.03441432412094,
#     -4.216674230005249,
#     -4.4944661023938375,
#     -4.004103843885551,
#     -4.004103843885551,
#     -4.730236551026537,
#     -4.229122645512975,
#     -3.890914358224581,
#     -3.890914358224581,
#     -3.879901090784399,
#     -3.879901090784399,
#     -3.879901090784399,
#     -3.879901090784399,
#     -3.879901090784399,
#     -3.879901090784399,
#     -3.879901090784399,
#     -3.879901090784399,
#     -3.879901090784399,
#     -3.7756815388890557,
#     -3.7756815388890557,
#     -3.7756815388890557,
#     -3.7756815388890557,
#     -3.7756815388890557,
#     -3.7756815388890557,
#     -3.7756815388890557,
#     -3.7756815388890557,
#     -3.7756815388890557,
#     -3.599128243191348,
#     -3.599128243191348,
#     -3.599128243191348,
#     -3.599128243191348,
# ]
    for gen_dir in rw_dir.iterdir():
        if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
            gen_num = int(gen_dir.name.split("_")[1])
            try:
                # Load the random walk data - need to modify the path to point to rw directory
                gen_path = rw_dir / f"generation_{gen_num:03d}"
                
                # Load population and best data files (using the correct filenames)
                pop_file = gen_path / "body_population.pkl"
                best_file = gen_path / "best_performers.pkl"
                
                if pop_file.exists() and best_file.exists():
                    with open(pop_file, 'rb') as f:
                        pop = pickle.load(f)
                    with open(best_file, 'rb') as f:
                        best_data = pickle.load(f)
                    
                    # For random walk, we typically just have one individual per generation
                    if pop and len(pop) > 0:
                        generations.append(gen_num)
                        for f in fitness:
                            # print(f"Random walk gen {gen_num}: fitness = {f}")
                            fitness_values.append(f)
                        
            except Exception as e:
                print(f"Warning: Could not load random walk generation {gen_num}: {e}")
    
    # Sort by generation number
    if generations:
        sorted_data = sorted(zip(generations, fitness_values))
        generations, fitness_values = zip(*sorted_data)
        return list(generations), list(fitness_values)
    else:
        return [], []

def plot_ea_vs_random_walk(ea_run_id: int, rw_run_id: int, data_path: str = "__data__/A3_modified"):
    """
    Compare EA run performance against random walk baseline.
    
    Args:
        ea_run_id: The EA run ID to compare
        rw_run_id: The random walk run ID to compare against
        data_path: Path to the data directory
    """
    # print(f"Loading EA run {ea_run_id}...")
    
    # Load EA data
    data_dir = Path(data_path)
    ea_run_dir = data_dir / f"run_{ea_run_id}"
    
    if not ea_run_dir.exists():
        raise FileNotFoundError(f"EA run directory {ea_run_dir} does not exist.")
    
    ea_generations = []
    ea_means = []
    ea_stds = []
    ea_bests = []
    
    for gen_dir in ea_run_dir.iterdir():
        if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
            gen_num = int(gen_dir.name.split("_")[1])
            try:
                pop, best_data = load_population_from_generation(gen_num, ea_run_id, data_path)
                fitness_values = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
                mean = np.mean(fitness_values)
                std = np.std(fitness_values)
                best = best_data.get("body_fitness", None)
                
                print(f"EA Gen {gen_num}: Pop fitness range: {min(fitness_values):.3f} to {max(fitness_values):.3f}, Best: {best}")
                
                ea_generations.append(gen_num)
                ea_means.append(mean)
                ea_stds.append(std)
                ea_bests.append(best)
            except Exception as e:
                print(f"Warning: Could not load EA generation {gen_num}: {e}")
    
    if not ea_generations:
        raise ValueError(f"No EA data found for run_{ea_run_id}")
    
    # Sort EA data by generation
    sorted_ea_data = sorted(zip(ea_generations, ea_means, ea_stds, ea_bests))
    ea_generations, ea_means, ea_stds, ea_bests = zip(*sorted_ea_data)
    ea_means = np.array(ea_means)
    ea_stds = np.array(ea_stds)
    ea_bests = np.array(ea_bests)
    ea_x = np.array(ea_generations)
    
    print(f"Loading random walk run {rw_run_id}...")
    
    # Load random walk data
    rw_generations, rw_fitness = load_random_walk_data(rw_run_id, data_path)
    
    # print(rw_generations, rw_fitness)

    if not rw_generations:
        raise ValueError(f"No random walk data found for rw_{rw_run_id}")
    
    rw_x = np.array(rw_generations)
    rw_y = np.array(rw_fitness)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot EA data
    ax.plot(ea_x, ea_means, linestyle="--", linewidth=1.5, color="blue", label=f"EA Run {ea_run_id} - Mean Fitness")
    ax.fill_between(ea_x, ea_means - ea_stds, ea_means + ea_stds, color="blue", alpha=0.2, label=f"EA Run {ea_run_id} - Std. Dev.")
    ax.plot(ea_x, ea_bests, linestyle="-", linewidth=2, color="red", label=f"EA Run {ea_run_id} - Best Fitness")
    
    # Plot random walk data
    ax.plot(rw_x, rw_y, linestyle="-", linewidth=2, color="green", marker="o", markersize=4, 
            label=f"Random Walk {rw_run_id}", alpha=0.7)
    
    
    # Add horizontal line for random walk mean
    rw_mean = np.mean(rw_y)
    ax.axhline(y=rw_mean, color="green", linestyle=":", alpha=0.5, 
               label=f"Random Walk Mean: {rw_mean:.3f}")
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Negative Distance to Target)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"EA Run {ea_run_id} vs Random Walk {rw_run_id} Performance Comparison")
    
    # Add statistics text box
    ea_final_best = ea_bests[-1] if ea_bests[-1] is not None else 0
    ea_final_mean = ea_means[-1]
    improvement_over_rw = ((ea_final_best - rw_mean) / abs(rw_mean) * 100) if rw_mean != 0 else 0
    
    stats_text = f"EA Final Best: {ea_final_best:.3f}\n"
    stats_text += f"EA Final Mean: {ea_final_mean:.3f}\n"
    stats_text += f"Random Walk Mean: {rw_mean:.3f}\n"
    stats_text += f"EA Improvement: {improvement_over_rw:.1f}%"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    save_dir = data_dir / "comparison_plots"
    save_dir.mkdir(exist_ok=True)
    
    # Create unique filename for the comparison plot
    base_name = f"ea_run_{ea_run_id}_vs_rw_{rw_run_id}_comparison"
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{base_name}_{unique_id}.png"
    
    full_path = save_dir / filename
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {full_path}")
    
    plt.show()



if __name__ == "__main__":
    # print(load_random_walk_data(2))
    plot_ea_vs_random_walk(ea_run_id = 2, rw_run_id = 2)

