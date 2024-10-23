from typing import Callable, List, Tuple

import numpy as np

from genetic_algorithm.core.fitness_evaluation import evaluate_population
from genetic_algorithm.core.initial_population import initialize_population
from genetic_algorithm.generation.one_generation import one_gen


def run_genetic_algorithm(
    population_size: int,
    fitness_function: Callable,
    num_genes: int,
    gene_range: List[Tuple[float]],
    num_generations: int,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.01,
    mutation_range: float = 0.1,
    elitism: int = 2,
) -> Tuple[
    np.ndarray, np.ndarray, List[float], List[np.ndarray], np.ndarray, float, float
]:
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

    # Initialize the initial population
    population = initialize_population(population_size, num_genes, gene_range)
    fitness_results = evaluate_population(population, fitness_function)
    # Lists to store best individuals and their fitness values
    fitness_in_pop = []
    best_in_pop = []

    # Initial parameters for mutation and elitism
    mutation_rate_0 = mutation_rate
    mutation_range_0 = mutation_range
    elitism_0 = elitism

    # Initialize the best fitness for tracking changes
    last_best_fitness = -np.inf
    prev_pop = population
    # Run the genetic algorithm for a specified number of generations
    for generation in range(num_generations):
        # Perform one generation
        population = one_gen(
            prev_pop,
            fitness_function,
            crossover_rate,
            mutation_rate,
            mutation_range,
            gene_range,
            elitism,
            fitness_results=fitness_results,
        )

        # Evaluate population and find the best individual
        fitness_results = evaluate_population(population, fitness_function)
        best_individual, best_fitness = max(fitness_results, key=lambda x: x[1])

        # Store the best individual and fitness values
        fitness_in_pop.append(best_fitness)
        best_in_pop.append(best_individual)

        # Controlled Printing: Print only if the best fitness changes
        if best_fitness != last_best_fitness:
            print(
                f"Generation {generation + 1}: Best Individual: {best_individual}, Best Fitness: {best_fitness}"
            )
            last_best_fitness = best_fitness

        # Adaptive mutation
        if (
            generation > 200
            and generation % 100 == 0
            and fitness_in_pop[generation] == fitness_in_pop[generation - 100]
        ):
            mutation_rate = 0.2  # 20 * mutation_rate_0
            mutation_range = 10 * mutation_range_0
            elitism = 5 * elitism
            print("adapt")
        elif (
            generation > 2000
            and generation % 500 == 0
            and fitness_in_pop[-1] == fitness_in_pop[-1000]
        ):  # reset stacked pop
            mutation_rate = 1
            mutation_range = 10 * mutation_range_0
            elitism = population_size // 2
            print("reset")
        else:
            mutation_rate = mutation_rate_0
            mutation_range = mutation_range_0
            elitism = elitism_0

        # Stopping criteria
        if (
            generation > 1000
            and generation % 1000 == 0
            and (fitness_in_pop[-1] - fitness_in_pop[generation // 2])
            / (fitness_in_pop[generation // 2] - fitness_in_pop[0])
            < 10 ** (-100)
        ):
            break

        prev_pop = population

    # Best individual and fitness in the final generation
    final_best_individual = best_in_pop[-1]
    final_best_fitness = fitness_in_pop[-1]

    # Print final generation results
    print(
        f"Generation {generation + 1}: Best Individual: {final_best_individual}, Best Fitness: {final_best_fitness}"
    )
    print(
        f"Overall Best Individual: {best_in_pop[np.argmax(fitness_in_pop)]}, "
        f"Overall Best Fitness: {np.max(fitness_in_pop)}"
    )
    # Return the best individual and fitness values across generations
    # final_best_individual ... best individual in last generation
    # population ... last pop
    # fitness_in_pop ... best fitness over generation
    # best_in_pop ... best individuals over generation
    # best_in_pop[np.argmax(fitness_in_pop)] ... overall best individual
    # np.max(fitness_in_pop) ... overall best fitness
    # final_best_fitness ... best fitness in last generation
    return (
        final_best_individual,
        population,
        fitness_in_pop,
        best_in_pop,
        best_in_pop[np.argmax(fitness_in_pop)],
        np.max(fitness_in_pop),
        final_best_fitness,
        generation,
    )
