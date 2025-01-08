from typing import List, Tuple

import numpy as np


def tournament_selection(
    population: List,
    fitness_results: List[Tuple],
    elitism_count: int,
    tournament_size: int = 2,
) -> np.ndarray:
    """
    Perform tournament selection to choose individuals as parents based on their fitness using NumPy.

    Parameters:
    - population: List of individuals in the population.
    - fitness_results: List of tuples, each containing an individual and its fitness value.
    - elitism_count: Number of top individuals to preserve through elitism (default is 0).
    - tournament_size: Number of individuals participating in each tournament.

    Returns:
    - List of selected parents.
    """
    # Ensure population and fitness_results have the same length
    assert len(population) == len(
        fitness_results
    ), "Population and fitness_results must have the same length."

    # Initialize the selected parents list
    selected_parents_indices = []

    # Perform tournament selection
    for _ in range(len(population) - elitism_count):
        # Randomly choose participants for the tournament
        tournament_participants_indices = np.random.choice(
            len(population), size=tournament_size, replace=False
        )

        # Evaluate fitness for each participant
        tournament_participants_fitness = [
            fitness
            for _, fitness in [
                fitness_results[i] for i in tournament_participants_indices
            ]
        ]

        # Select the winner (participant with the highest fitness)
        winner_index = tournament_participants_indices[
            np.argmax(tournament_participants_fitness)
        ]

        # Add the winner to the selected parents list
        selected_parents_indices.append(winner_index)

    # Preserve top individuals through elitism
    if elitism_count > 0:
        top_individuals_indices = np.argsort(
            [fitness for _, fitness in fitness_results]
        )[::-1][:elitism_count]
        selected_parents_indices = np.concatenate(
            (top_individuals_indices, selected_parents_indices)
        )

    # Construct selected parents list using array indexing
    selected_parents = np.array(population)[selected_parents_indices]

    # Prevent to have the same vector among whole population
    # If there is only one vector in whole population, do selection again
    if np.all(selected_parents[:-1] == selected_parents[1:]):
        selected_parents = tournament_selection(
            population, fitness_results, elitism_count, tournament_size
        )
    return np.array(selected_parents)


'''def rank_based_selection(
    population: np.ndarray, fitness_results: List[Tuple], elitism_count: int
) -> np.ndarray:
    """
    Perform rank-based selection with elitism.

    Parameters:
    - population: The population of individuals (NumPy array).
    - fitness_results: A list of tuples where each tuple contains an individual and its fitness value.
    - elitism_count: The number of top individuals to retain directly (elitism).

    Returns:
    - The new population after selection (NumPy array).
    """
    # Extract fitness values and sort them to get the ranks
    fitness_values = np.array([item[1] for item in fitness_results])
    sorted_indices = np.argsort(fitness_values)  # Indices that would sort the array
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = (
        np.arange(len(fitness_values)) + 1
    )  # Assign ranks based on sorted indices
    # Calculate selection probabilities from ranks
    total_ranks = np.sum(ranks)
    selection_probabilities = ranks / total_ranks

    # Retain elite individuals
    if elitism_count == 0:
        elite_indices = []
    else:
        elite_indices = sorted_indices[-elitism_count:]

    elites = population[elite_indices]

    # Select individuals based on rank probabilities
    selected_indices = np.random.choice(
        len(population),
        size=(len(population) - elitism_count),
        p=selection_probabilities,
        replace=False,
    )
    selected_individuals = population[selected_indices]

    # Combine elite individuals with the selected ones
    select_pool = np.concatenate((elites, selected_individuals))

    return select_pool'''

'''def rank_based_selection(
    population: np.ndarray, fitness_results: List[Tuple], elitism_count: int
) -> np.ndarray:
    """
    Perform rank-based selection with elitism.

    Parameters:
    - population: The population of individuals (NumPy array).
    - fitness_results: A list of tuples where each tuple contains an individual and its fitness value.
    - elitism_count: The number of top individuals to retain directly (elitism).

    Returns:
    - The new population after selection (NumPy array).
    """
    # Extract fitness values and sort them to get the ranks
    fitness_values = np.array([item[1] for item in fitness_results])
    sorted_indices = np.argsort(fitness_values)  # Indices that would sort the array
    # Retain elite individuals
    if elitism_count == 0:
        elite_indices = []
    else:
        elite_indices = sorted_indices[-elitism_count:]
    elites = population[elite_indices]

    population = np.delete(population, elite_indices, axis=0)
    fitness_values = np.delete(fitness_values, elite_indices)
    sorted_indices = np.argsort(fitness_values)

    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = (
        np.arange(len(fitness_values)) + 1
    )  # Assign ranks based on sorted indices
    # Calculate selection probabilities from ranks
    total_ranks = np.sum(ranks)
    selection_probabilities = ranks / total_ranks

    # Select individuals based on rank probabilities
    selected_indices = np.random.choice(
        len(population),
        size=(len(population)),
        p=selection_probabilities,
        replace=False,
    )
    selected_individuals = population[selected_indices]

    # Combine elite individuals with the selected ones
    select_pool = np.concatenate((elites, selected_individuals))

    return select_pool'''


def rank_based_selection(
    population: np.ndarray, fitness_results: List[Tuple], elitism_count: int
) -> np.ndarray:
    """
    Perform rank-based selection with elitism.

    Parameters:
    - population: The population of individuals (NumPy array).
    - fitness_results: A list of tuples where each tuple contains an individual and its fitness value.
    - elitism_count: The number of top individuals to retain directly (elitism).

    Returns:
    - The new population after selection (NumPy array).
    """
    # Extract fitness values and sort by fitness
    fitness_values = np.array([item[1] for item in fitness_results])
    sorted_indices = np.argsort(fitness_values)

    # Identify elite individuals
    elite_indices = sorted_indices[-elitism_count:] if elitism_count > 0 else []
    elites = population[elite_indices]

    # Compute ranks (higher fitness = higher rank)
    ranks = np.argsort(np.argsort(-fitness_values)) + 1

    # Exclude elites from rank-based selection
    non_elite_mask = np.ones(len(population), dtype=bool)
    non_elite_mask[elite_indices] = False
    non_elite_population = population[non_elite_mask]
    non_elite_ranks = ranks[non_elite_mask]

    # Calculate selection probabilities for non-elites
    total_ranks = np.sum(non_elite_ranks)
    selection_probabilities = non_elite_ranks / total_ranks

    # Select individuals based on rank probabilities
    selected_indices = np.random.choice(
        len(non_elite_population),
        size=len(non_elite_population),
        p=selection_probabilities,
        replace=False,
    )
    selected_individuals = non_elite_population[selected_indices]

    # Combine elite individuals with the selected ones
    new_population = np.concatenate((elites, selected_individuals))

    return new_population
