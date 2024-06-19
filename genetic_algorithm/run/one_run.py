from typing import Callable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from genetic_algorithm.core.fitness_evaluation import evaluate_population
from genetic_algorithm.core.initial_population import initialize_population
from genetic_algorithm.generation.one_generation import one_gen


def run_genetic_algorithm(population_size: int,
                          fitness_function: Callable,
                          num_genes: int,
                          gene_range: List[Tuple[float]],
                          num_generations: int,
                          crossover_rate: float = 0.85,
                          mutation_rate: float = 0.01,
                          mutation_range: float = 0.1,
                          elitism: int = 2,
                          name: str = None) -> Tuple[np.ndarray, List[float]]:
    """
    Run a genetic algorithm with real number representation for a specified number of generations.

    Parameters:
    - population_size: Number of individuals in the population.
    - num_genes: Number of genes in each individual.
    - gene_range: Range for generating initial gene values.
    - crossover_rate: Probability of crossover.
    - mutation_rate: Probability of mutation for each gene.
    - mutation_range: Range of mutation for each gene.
    - num_generations: Number of generations to run the algorithm.

    Returns:
    - Tuple containing the best individual in the final generation and a list of fitness values across generations.
    """
    mutation_rate_0 = mutation_rate
    mutation_range_0 = mutation_range
    elitism_0 = elitism

    # Initialize the initial population
    population = initialize_population(population_size, num_genes, gene_range)

    best_in_pop = []
    fitness_in_pop = []

    # Run the genetic algorithm for a specified number of generations
    for generation in range(num_generations):
        print(f"Generation {generation + 1}:")

        # Perform one generation
        population = one_gen(population, fitness_function, crossover_rate, mutation_rate, mutation_range, gene_range,
                             elitism)

        # Display the best individual in the current generation
        best_individual, best_fitness = max(evaluate_population(population, fitness_function), key=lambda x: x[1])
        print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")

        # Add fitness of the best individual in the list
        fitness_in_pop.append(best_fitness)
        best_in_pop.append(best_individual)

        # Adaptive mutation
        if generation > 200 and \
                generation % 100 == 0 and \
                fitness_in_pop[generation] == fitness_in_pop[generation - 200]:
            mutation_rate = 20 * mutation_rate_0
            mutation_range = mutation_range_0
            elitism = 0
        else:
            mutation_rate = mutation_rate_0
            mutation_range = mutation_range_0
            elitism = elitism_0

    # Plot the fitness over generations
    plt.plot(np.arange(len(fitness_in_pop)), fitness_in_pop)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Over Generations')
    plt.savefig('Results/genetic_algorithm/fitness/fitness_' + name + '.pdf')
    plt.close()

    # Return the best individual and fitness values across generations
    # the best individual = best in last generation
    # population = las pop
    # fitness_in_pop = fitness of the best in every pop
    # best_in_pop = best in every pop
    # best_in_pop[np...] = best over all gen
    return best_individual, population, fitness_in_pop, best_in_pop, best_in_pop[np.argmax(fitness_in_pop)]
