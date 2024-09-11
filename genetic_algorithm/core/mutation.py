from typing import Tuple, List

import numpy as np
from numba import jit

@jit
def mutate(individual: np.ndarray,
           mutation_rate: float,
           mutation_range: float,
           gene_range: List[Tuple[float]]) -> np.ndarray:
    """
    Mutate an individual in a genetic algorithm with real number representation using NumPy.

    Parameters:
    - individual: NumPy array of real numbers representing an individual.
    - mutation_rate: Probability of mutation for each gene.
    - mutation_range: Range of mutation for each gene.
    - gene_range: Range of valid gene values.

    Returns:
    - Mutated individual (NumPy array).
    """
    # Create a mask for genes to be mutated
    mask = np.random.rand(*individual.shape) < mutation_rate

    # Generate random mutations within the specified range and apply mask
    mutations = np.random.uniform(-mutation_range, mutation_range, size=individual.shape)
    individual[mask] += mutations[mask]

    # Ensure mutated genes stay within the valid range using vectorized np.clip
    gene_min_values, gene_max_values = np.array(gene_range).T
    np.clip(individual, gene_min_values, gene_max_values, out=individual)

    return individual
