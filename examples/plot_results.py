"""
Standalone plotting script for EA results.
This script can be used to plot fitness statistics from your evolutionary algorithm runs.
"""

import sys
import os
from pathlib import Path

# Add the examples directory to the path so we can import from A3_modified
sys.path.insert(0, str(Path(__file__).parent))

try:
    from A3_modified import EAConfig
    from plot_function import plot_run_statistics, load_population_from_generation
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have matplotlib installed and A3_modified.py is in the same directory")
    sys.exit(1)


def plot_ea_results(run_id=0, rng_seed=42):
    """
    Plot the results of an evolutionary algorithm run.
    
    Args:
        run_id (int): The run ID to plot (default: 0)
        rng_seed (int): Random seed used in the original run (default: 42)
    """
    try:
        # Create sim_config with the same settings as your run
        sim_config = EAConfig(rng_seed=rng_seed)
        
        print(f"Plotting results for run {run_id}")
        print(f"Looking for data in: {sim_config.data}")
        
        # Check if run directory exists
        run_dir = sim_config.data / f"run_{run_id}"
        if not run_dir.exists():
            print(f"Error: Run directory {run_dir} does not exist")
            print("Available runs:")
            data_dir = sim_config.data
            if data_dir.exists():
                for item in data_dir.iterdir():
                    if item.is_dir() and item.name.startswith("run_"):
                        print(f"  - {item.name}")
            else:
                print(f"  Data directory {data_dir} does not exist")
            return
        
        # Plot the statistics
        plot_run_statistics(sim_config, run_id)
        
        print(f"Successfully plotted results for run {run_id}")
        
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()


def plot_multiple_runs(run_ids, rng_seed=42):
    """
    Plot results from multiple runs for comparison.
    
    Args:
        run_ids (list): List of run IDs to plot
        rng_seed (int): Random seed used in the original runs (default: 42)
    """
    try:
        sim_config = EAConfig(rng_seed=rng_seed)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for run_id in run_ids:
            run_dir = sim_config.data / f"run_{run_id}"
            if not run_dir.exists():
                print(f"Warning: Run {run_id} directory not found, skipping...")
                continue
            
            # Collect data for this run
            generations = []
            best_fitnesses = []
            
            for gen_dir in sorted(run_dir.iterdir()):
                if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
                    gen_num = int(gen_dir.name.split("_")[1])
                    try:
                        _, best_data = load_population_from_generation(sim_config, gen_num, run_id)
                        best_fitness = best_data.get("body_fitness", None)
                        if best_fitness is not None:
                            generations.append(gen_num)
                            best_fitnesses.append(best_fitness)
                    except Exception as e:
                        print(f"Warning: Could not load generation {gen_num} from run {run_id}: {e}")
            
            if generations:
                ax.plot(generations, best_fitnesses, marker='o', label=f'Run {run_id}', linewidth=2)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Best Fitness Comparison Across Runs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save the comparison plot
        save_path = sim_config.data / "comparison_plots"
        save_path.mkdir(exist_ok=True)
        
        filename = f"runs_comparison_{'_'.join(map(str, run_ids))}.png"
        full_path = save_path / filename
        
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {full_path}")
        plt.show()
        
    except Exception as e:
        print(f"Error plotting multiple runs: {e}")
        import traceback
        traceback.print_exc()


def list_available_runs(rng_seed=42):
    """List all available runs in the data directory."""
    try:
        sim_config = EAConfig(rng_seed=rng_seed)
        data_dir = sim_config.data
        
        if not data_dir.exists():
            print(f"Data directory {data_dir} does not exist")
            return []
        
        runs = []
        print(f"Available runs in {data_dir}:")
        
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                run_id = item.name.split("_")[1]
                runs.append(int(run_id))
                
                # Count generations in this run
                gen_count = len([g for g in item.iterdir() 
                               if g.is_dir() and g.name.startswith("generation_")])
                print(f"  - Run {run_id}: {gen_count} generations")
        
        return sorted(runs)
        
    except Exception as e:
        print(f"Error listing runs: {e}")
        return []


if __name__ == "__main__":
    print("EA Results Plotting Script")
    print("=" * 30)
    
    # List available runs
    available_runs = list_available_runs()
    
    if not available_runs:
        print("No runs found. Make sure you have run the evolutionary algorithm first.")
    else:
        print(f"\nFound {len(available_runs)} runs: {available_runs}")
        
        # Plot the first available run
        first_run = available_runs[0]
        print(f"\nPlotting results for run {first_run}...")
        plot_ea_results(run_id=first_run)
        
        # If multiple runs available, create comparison plot
        if len(available_runs) > 1:
            print(f"\nCreating comparison plot for all runs...")
            plot_multiple_runs(available_runs[:5])  # Limit to first 5 runs for readability