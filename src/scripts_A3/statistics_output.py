import numpy as np

def print_statistics(population):
    """Print fitness statistics for a population. Requires a population of genotypes with the attribute .fitness.values present!

    Args:
        population (list): Population of individuals with fitness values
    """
    # Extract fitness values
    pop_fitness = [ind.fitness.values[0] for ind in population]
    
    # Calculate statistics
    avg_fitness = np.mean(pop_fitness)
    std_fitness = np.std(pop_fitness)
    best_fitness = np.max(pop_fitness)
    
    # Print
    print(f'{"Av. fitness":<12} {"std":<12} {"Best fitness":<12}')
    print(f"{avg_fitness:<12.2f} {std_fitness:<12.2f} {best_fitness:<12.2f}")
