# name: Samuel Buranský
# MUNI UČO: 506073
# mail: 506073@mail.muni.cz

import os
from datetime import datetime as dt
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.fourier_series_value import double_fourier_value, double_fourier_sequence
from utils.load_dataset import load_data
from genetic_algorithm.run.one_run import run_genetic_algorithm
from periodogram.clean_periodogram import frequency_grid, fourier_transform, clean
from periodogram.lomb_scargle import lomb_scargle

np.set_printoptions(threshold=np.inf)


def tumbler_periodogram(t: np.ndarray,
                        y: np.ndarray,
                        name: str,
                        n_iter=100,
                        gain=0.5,
                        n_b=4,
                        dev_use_for_ls=None,
                        dev: Optional[np.ndarray] = None) -> None:
    """
    Compute Lomb-Scargle periodogram and plot results.

    Parameters:
    - t: Array of time values (Julian Date).
    - y: Array of corresponding flux values.
    - name: Name to use for saving plots and results.
    - frequency: Array of frequencies to evaluate the periodogram.
    - dev: Optional array of uncertainties in flux values (error bars).

    Returns:
    - None. Save plots and results to disk.
    """
    # Ensure directories exist
    os.makedirs('Results/lomb_scargle/Graphs/', exist_ok=True)
    os.makedirs('Results/lomb_scargle/Periodograms/', exist_ok=True)
    os.makedirs('Results/lomb_scargle/Results/', exist_ok=True)
    frequency = frequency_grid(t, n_b)
    # Compute the periodogram and maxima
    periodogram_lomb, maximas_lomb = lomb_scargle(t, y, frequency, dev, dev_use_for_ls=dev_use_for_ls)
    periodogram_fourier = fourier_transform(t, y)[0], fourier_transform(t, y)[2]
    clean_periodogram, clean_maximas = clean(fourier_transform(t, y)[0],
                                             fourier_transform(t, y)[1],
                                             fourier_transform(t, y)[2],
                                             n_iter=n_iter, gain=gain)

    # Plot the observed data with error bars
    plt.errorbar(t, y, yerr=dev, fmt='.', label='Data')
    plt.xlabel('Julian Date (JD)')
    plt.ylabel('Normalized Flux')
    plt.title('Observed Data')
    plt.legend()
    plt.savefig(f'Results/lomb_scargle/Graphs/{name}_graph.pdf')
    plt.show()
    plt.close()

    # Plot the periodograms
    ax1 = plt.subplot(311)
    plt.title(f'Periodogram_{name}')
    ax1.plot(periodogram_lomb[0], periodogram_lomb[1], label='Lomb-Scargle Periodogram')
    # ax1.scatter(maximas_lomb[0], maximas_lomb[1], color='red', label='Lomb-Scargle Maxima')
    plt.legend()
    plt.xlim(-0.5, 10)

    ax2 = plt.subplot(312)
    ax2.plot(periodogram_fourier[0][:len(fourier_transform(t, y)[0]) // 2], np.abs(periodogram_fourier[1]) ** 2,
             label='Fourier Periodogram')
    plt.legend()
    plt.xlim(-0.5, 10)

    ax3 = plt.subplot(313)
    ax3.plot(clean_periodogram[0][:len(fourier_transform(t, y)[0]) // 2], np.abs(clean_periodogram[1]) ** 2,
             label='CLEAN Periodogram')
    # ax3.scatter(clean_maximas[0], clean_maximas[1], color='red', label='CLEAN Maxima')
    plt.legend()
    plt.xlim(-0.5, 10)

    plt.savefig(f'Results/lomb_scargle/Periodograms/{name}_PERIODOGRAM.pdf')
    plt.show()
    plt.close()

    # Save the maxima to a text file as two columns
    ls_maximas_file = np.column_stack((maximas_lomb[0], maximas_lomb[1]))
    clean_maximas_file = np.column_stack((clean_maximas[0], clean_maximas[1]))
    np.savetxt(f'Results/lomb_scargle/Results/{name}_LS.txt', ls_maximas_file,
               delimiter=" ", header='Frequency Power', comments='')
    np.savetxt(f'Results/lomb_scargle/Results/{name}_CLEAN.txt', clean_maximas_file,
               delimiter=" ", header='Frequency Power', comments='')


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
                                  mutation_range: float = 0.1) -> None:
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


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # load data
    name = 'ID1916_007'
    data = load_data(name, column_names=('julian_day', 'noiseless_flux', 'noisy_flux', 'sigma', 'deviation_used'),
                     appendix='.txt')

    tumbler_periodogram(data['julian_day'].values, data['noisy_flux'].values,
                        name=name, n_iter=2000, gain=0.1, dev=data['deviation_used'])

    m_ = 2

    '''
    def fitness(solution):
        """
        Fitness function
        :param solution: set of the free parameters
        :return: fitness value
        """
        x, y, delta = data['julian_day'], data['noisy_flux'], data['deviation_used']

        # Vectorized calculation of Fourier values
        y_model = double_fourier_sequence(solution, m_, x)

        # calculation of the chi^2 and returning 1/chi^2
        chi2 = np.sum((y - y_model) ** 2 / delta ** 2)
        return 1 / chi2


    tumbler_genetic_algorithm_fit(data,
                                  fitness,
                                  m_=m_,
                                  population_size=10,
                                  gene_range=((-0.2, 0.2), (0.90, 1.10), (0, 4), (0, 4)),
                                  name=name, num_generations=1, elitism=1, mutation_rate=0.05, mutation_range=0.05)'''

'''
class TestCases(unittest.TestCase):
    def test_ls(self):
        tumbler_periodogram(data['julian_day'].values, data['noisy_flux'].values, name=name)

    def test_ga(self):
        tumbler_genetic_algorithm_fit(data,
                                      fitness,
                                      m_=m_,
                                      population_size=500,
                                      gene_range=((-0.2, 0.2), (0.85, 1.15), (0.5, 1.5), (0.5, 1.5)),
                                      name=name, num_generations=100)
'''
