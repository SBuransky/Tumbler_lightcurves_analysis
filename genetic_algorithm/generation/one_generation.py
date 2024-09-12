from typing import List, Callable, Tuple
from joblib import Parallel, delayed
import numpy as np

from genetic_algorithm.core.crossover import crossover
from genetic_algorithm.core.fitness_evaluation import evaluate_population
from genetic_algorithm.core.mutation import mutate
from genetic_algorithm.core.selection import rank_based_selection


def one_gen(population: List,
            fitness_function: Callable,
            crossover_rate: float,
            mutation_rate: float,
            mutation_range: float,
            gene_range: List[Tuple[float]],
            elitism: int = 0) -> np.ndarray:
    """
    Perform one generation of a genetic algorithm with real number representation.

    Parameters:
    - population: List of individuals in the current generation.
    - crossover_rate: Probability of crossover.
    - mutation_rate: Probability of mutation for each gene.
    - mutation_range: Range of mutation for each gene.

    Returns:
    - List of individuals in the next generation.
    """
    # Evaluate fitness of the current population
    fitness_results = evaluate_population(population, fitness_function)

    # Select parents for crossover
    selected_parents = rank_based_selection(population, fitness_results, elitism)

    # Perform crossover to create offspring
    offspring = []

    while len(offspring) < len(selected_parents) - elitism:
        parent_indices = np.random.choice(len(selected_parents), size=2, replace=False)
        parent1, parent2 = selected_parents[parent_indices[0]], selected_parents[parent_indices[1]]
        if not np.array_equal(parent1, parent2):
            offspring.append(crossover(parent1, parent2, crossover_rate))

    # Perform mutation on offspring
    mutated_offspring = [mutate(individual, mutation_rate, mutation_range, gene_range) for individual in offspring]

    # Combine parents and mutated offspring for the next generation
    next_generation = np.vstack((np.array(selected_parents[:elitism]), mutated_offspring))

    return next_generation
