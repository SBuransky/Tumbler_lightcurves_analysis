import os

import pandas as pd
from typing import Callable, Tuple

from genetic_algorithm.run.one_run import run_genetic_algorithm
import matplotlib.pyplot as plt
from utils.fourier_series_value import double_fourier_sequence
from datetime import datetime as dt
import numpy as np


def tumbler_genetic_algorithm_fit(data: pd.DataFrame,
                                  fitness_function: Callable[[np.ndarray], float],
                                  name: str,
                                  m_: int = 1,
                                  population_size: int = 500,
                                  gene_range: Tuple[
                                      Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[
                                          float, float]] = ((-0.2, 0.2), (0.85, 1.15), (0.5, 1.5), (0.5, 1.5)),
                                  num_generations: int = 100,
                                  elitism: int = 2,
                                  crossover_rate: float = 0.95,
                                  mutation_rate: float = 0.01,
                                  mutation_range: float = 0.5) -> None:
    """
    Run a genetic algorithm to fit a model to data and visualize results.

    Parameters:
    - data: DataFrame or structured array containing 'julian_day', 'noisy_flux', and 'deviation_used'.
    - fitness_function: Function to evaluate fitness of individuals in the population.
    - name: Name to use for saving plots and results.
    - m_: Parameter controlling the number of genes in the genetic algorithm.
    - population_size: Number of individuals in each generation.
    - gene_range: Range of values for each gene in the chromosome.
    - num_generations: Number of generations to run the genetic algorithm.
    - elitism: Number of top individuals to carry over to the next generation unchanged.
    - crossover_rate: Probability of crossover between two individuals.
    - mutation_rate: Probability of mutation for each gene.
    - mutation_range: Range of mutation for each gene.

    Returns:
    - None. Save plots and results to disk.

    This function sets up directories for saving results, runs the genetic algorithm,
    and then plots the best fit and fitness progression over generations. It saves
    these plots and the best results to specified directories.

    """
    # Ensure directories exist
    os.makedirs('Results/genetic_algorithm/graphs/', exist_ok=True)
    os.makedirs('Results/genetic_algorithm/fitness/', exist_ok=True)
    os.makedirs('Results/genetic_algorithm/results/', exist_ok=True)

    # Run genetic algorithm
    final_generation = run_genetic_algorithm(
        population_size=population_size,
        fitness_function=fitness_function,
        num_genes=2 * m_ + 2 * m_ * (2 * m_ + 1) + 3,
        gene_range=(2 * m_ + 2 * m_ * (2 * m_ + 1)) * (gene_range[0],) + gene_range[1:],
        num_generations=num_generations,
        elitism=elitism,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_range=mutation_range
    )

    # Generate timestamp for file names
    ts = dt.now()
    ending = str(ts.year)[2:] + '{:02d}'.format(ts.month) + '{:02d}'.format(ts.day) + '-' + '{:02d}'.format(
        ts.hour) + '{:02d}'.format(ts.minute) + '{:02d}'.format(ts.second)

    # Plotting best fit and overall best
    days = data['julian_day'].values
    plt.plot(days, double_fourier_sequence(final_generation[0], m_, days), label='Last Generation Best')
    plt.plot(days, double_fourier_sequence(final_generation[4], m_, days), label='Overall Best')

    # Plot noisy data
    plt.scatter(days, data['noisy_flux'].values, c='gray', marker='+', s=5)
    plt.errorbar(days, data['noisy_flux'].values, yerr=data['deviation_used'].values, fmt='none', color='black',
                 elinewidth=1.5, capsize=0)

    # Plot settings
    plt.xlabel('Time [days]')
    plt.ylabel('Normalized light flux')
    plt.title('Genetic Algorithm Fit')
    plt.legend()

    # Save and display plot
    plt.savefig(f'Results/genetic_algorithm/graphs/{name}_graph_{ending}.pdf')
    plt.show()
    plt.close()

    # Plot fitness over generations
    plt.plot(np.arange(len(final_generation[2])), final_generation[2])
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Over Generations')

    # Save fitness plot
    plt.savefig(f'Results/genetic_algorithm/fitness/{name}_fitness_{ending}.pdf')
    plt.close()

    # Save best results to text file
    with open(f'Results/genetic_algorithm/results/{name}_result_{ending}.txt', 'w') as file:
        file.write('Best in last gen:\n')
        file.write(str(final_generation[0]))
        file.write('\nBest fitness in last gen:\n')
        file.write(str(final_generation[5]))
        file.write('\n\nBest in all:\n')
        file.write(str(final_generation[4]))
        file.write('\nBest fitness in all:\n')
        file.write(str(final_generation[6]))
