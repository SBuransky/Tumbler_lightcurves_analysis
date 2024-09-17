from typing import List, Callable, Tuple
from joblib import Parallel, delayed
import numpy as np


def evaluate_population(
    population: List[np.ndarray], fitness_function: Callable
) -> List[Tuple[np.ndarray, float]]:
    """
    Evaluate the fitness of the entire population in a genetic algorithm with real number representation.

    Parameters:
    - population: List of individuals in the population.
    - fitness_function: Callable function that calculates the fitness of an individual.

    Returns:
    - List of tuples, each containing an individual and its corresponding fitness value.
    """
    fitness_results = [
        (individual, fitness_function(individual)) for individual in population
    ]
    return fitness_results
