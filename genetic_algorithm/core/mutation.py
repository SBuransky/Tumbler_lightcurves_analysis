from typing import Tuple, List

import numpy as np


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

    # Generate random mutations within the specified range
    mutations = np.random.uniform(-mutation_range, mutation_range, size=individual.shape)

    # Apply mutations to the individual based on the mask
    mutated_individual = individual + mask * mutations

    # Ensure mutated genes remain within the valid range
    gene_min_values, gene_max_values = zip(*gene_range)
    mutated_individual = np.clip(mutated_individual, gene_min_values, gene_max_values)

    return mutated_individual
