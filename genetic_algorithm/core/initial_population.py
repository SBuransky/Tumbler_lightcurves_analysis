from typing import Tuple, List
import numpy as np


def initialize_population(population_size: int,
                          num_genes: int,
                          gene_range: List[Tuple[float]]) -> np.ndarray:
    """
    Initialize the initial population for a genetic algorithm with real number representation using NumPy.

    Parameters:
    - population_size: Number of individuals in the population.
    - num_genes: Number of genes in each individual.
    - gene_ranges: List of tuples, each specifying the range for a gene.

    Returns:
    - NumPy array representing the initial population.
    """
    gene_min_values, gene_max_values = zip(*gene_range)
    initial_population = np.random.uniform(gene_min_values, gene_max_values, size=(population_size, num_genes))

    return initial_population
