import numpy as np
import itertools
def euclidean_distance(ind1, ind2):
    """Compute the Euclidean distance between two individuals.

    Args:
        ind1 (list): First individual
        ind2 (list): Second individual

    Returns:
        float: Euclidean distance between the two individuals
    """
    x = np.array(ind1, dtype = float)
    y = np.array(ind2, dtype = float)
    return float(np.linalg.norm(x - y))

def _find_allele_range(population):
    X = np.vstack(population)
    return np.min(X, axis = 0), np.max(X, axis = 0)

def mean_abs_gene_difference(ind1, ind2, low_vec, up_vec):
    """Compute the mean absolute gene difference between two individuals, normalized by the allele ranges."""
    x = np.array(ind1, dtype = float)
    y = np.array(ind2, dtype = float)
    span = up_vec - low_vec
    diff = np.zeros_like(span, dtype = float)
    mask = span > 0.0
    if np.any(mask):
        diff[mask] = np.abs(x[mask] - y[mask]) / (span[mask] + 1e-32) 
    return float(np.clip(diff, 0.0, 1.0).mean())


def print_statistics(population, bounds=(0.0, 1.0)):
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
    
    # Cast individuals as numpy
    arrays = [np.array(ind, dtype = float) for ind in population]
    pairs = list(itertools.combinations(arrays, 2))
    
    # Calculate raw distances
    dists = [euclidean_distance(ind1, ind2) for ind1, ind2 in pairs]
    av_dists, min_dists, max_dists = float(np.mean(dists)), float(np.min(dists)), float(np.max(dists))
    

    
    
    print(f'{"Av. fitness":<12} {"std":<12} {"Best fitness":<12} {"Av. distance":<12} {"Min. distance":<12} {"Max. distance":<12}')
    print(f"{avg_fitness:<12.2f} {std_fitness:<12.2f} {best_fitness:<12.2f} {av_dists:<12.2f} {min_dists:<12.2f} {max_dists:<12.2f}")
