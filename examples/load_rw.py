import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import uuid
from typing import Tuple, List

def load_EA(generation, run_id=0, data_path="__data__/A3_modified"):
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

def load_RW(generation, run_id=0, data_path="__data__\A3_modified"):
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
    gen_dir = data_dir / f"randomwalk_{run_id}" / f"generation_{generation:03d}"

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

def plot_ea_vs_random_walk(ea_run_id: int, rw_run_id: int, data_path: str = "__data__\A3_modified"):
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
                pop, best_data = load_EA(gen_num, ea_run_id, data_path)
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
    
    # Sort EA data by generation
    sorted_ea_data = sorted(zip(ea_generations, ea_means, ea_stds, ea_bests))
    ea_generations, ea_means, ea_stds, ea_bests = zip(*sorted_ea_data)
    ea_means = np.array(ea_means)
    ea_stds = np.array(ea_stds)
    ea_bests = np.array(ea_bests)
    ea_x = np.array(ea_generations)
    

    # Load EA data
    data_dir = Path(data_path)
    rw_run_dir = data_dir / f"randomwalk_{rw_run_id}"

    if not ea_run_dir.exists():
        raise FileNotFoundError(f"EA run directory {ea_run_dir} does not exist.")
    
    rw_generations = []
    rw_means = []
    rw_stds = []
    rw_bests = []

    for gen_dir in rw_run_dir.iterdir():
        if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
            gen_num = int(gen_dir.name.split("_")[1])
            try:
                pop, best_data = load_RW(gen_num, rw_run_id, data_path)
                fitness_values = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
                mean = np.mean(fitness_values)
                std = np.std(fitness_values)
                best = best_data.get("body_fitness", None)
                
                print(f"RW Gen {gen_num}: Pop fitness range: {min(fitness_values):.3f} to {max(fitness_values):.3f}, Best: {best}")

                rw_generations.append(gen_num)
                rw_means.append(mean)
                rw_stds.append(std)
                rw_bests.append(best)
            except Exception as e:
                print(f"Warning: Could not load RW generation {gen_num}: {e}")

      
    # Sort RW data by generation
    sorted_rw_data = sorted(zip(rw_generations, rw_means, rw_stds, rw_bests))
    rw_generations, rw_means, rw_stds, rw_bests = zip(*sorted_rw_data)
    rw_means = np.array(rw_means)
    rw_stds = np.array(rw_stds)
    rw_bests = np.array(rw_bests)
    rw_x = np.array(rw_generations)

    if not ea_generations:
        raise ValueError(f"No EA data found for run_{ea_run_id}")
    
###################################PLOTTING###########################################

    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot EA data
    ax.plot(ea_x, ea_means, linestyle="--", linewidth=1.5, color="blue", label=f"EA Run - Mean Fitness")
    # ax.fill_between(ea_x, ea_means - ea_stds, ea_means + ea_stds, color="blue", alpha=0.2, label=f"EA Run - Std. Dev.")
    ax.plot(ea_x, ea_bests, linestyle="-", linewidth=2, color="red", label=f"EA Run - Best Fitness")
    
    # Plot random walk data
    ax.plot(rw_x, rw_bests, linestyle="-", linewidth=2, color="green", marker="o", markersize=4, 
            label=f"Random Walk", alpha=0.7)
    
    
    # Add horizontal line for random walk mean
    # rw_mean_value = np.mean(rw_bests)  # Calculate mean of random walk best values

    ax.plot(rw_x, rw_means, linestyle=":", linewidth=2, color="green", marker="o", markersize=4, alpha=0.5,
            label=f"Random Walk Mean")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Negative Distance to Target)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"EA Run vs Random Walk Performance Comparison")
    
    # Add statistics text box
    ea_final_best = ea_bests[-1] if ea_bests[-1] is not None else 0
    ea_final_mean = ea_means[-1]
    ea_final_std = ea_stds[-1]  # Standard deviation of the final generation population
    ea_best_std = np.std(ea_bests)  # Standard deviation of best fitness across all generations
    rw_std_value = np.std(rw_bests)  # Calculate standard deviation of random walk best values
    rw_mean_value = np.mean(rw_means)  # Get the mean of random walk means
    improvement_over_rw = ((ea_final_best - rw_mean_value) / abs(rw_mean_value) * 100) if rw_mean_value != 0 else 0
    
    rw_best_fitness = np.max(rw_bests)  # Get the best (highest) fitness from random walk
    # print(f"Random Walk Best Fitness: {rw_best_fitness:.3f}")

    stats_text = f"EA Final Best: {ea_final_best:.3f}\n"
    stats_text += f"EA Final Mean: {ea_final_mean:.3f}\n"
    # stats_text += f"EA Best Std Dev: {ea_best_std:.3f}\n"
    # stats_text += f"EA Final Std Dev: {ea_final_std:.3f}\n"
    stats_text += f"Random Walk Best: {rw_best_fitness:.3f}\n"
    stats_text += f"Random Walk Mean: {rw_mean_value:.3f}\n"
    # stats_text += f"Random Walk Std Dev: {rw_std_value:.3f}\n"
    stats_text += f"EA Improvement (compared to RW): {improvement_over_rw:.1f}%"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
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
    # # List available runs first
    # data_dir = Path("__data__/A3_modified")
    # print("Available EA runs:")
    # for item in data_dir.iterdir():
    #     if item.is_dir() and item.name.startswith("run_"):
    #         run_id = item.name.split("_")[1]
    #         print(f"  - run_{run_id}")
    
    # print("\nAvailable Random Walk runs:")
    # for item in data_dir.iterdir():
    #     if item.is_dir() and item.name.startswith("rw_"):
    #         rw_id = item.name.split("_")[1]
    #         print(f"  - rw_{rw_id}")
    
    # Plot run_2 vs available random walk
    plot_ea_vs_random_walk(ea_run_id=2, rw_run_id=3)  # Change rw_run_id as needed

