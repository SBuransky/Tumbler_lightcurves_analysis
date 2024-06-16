# name: Samuel Buranský
# MUNI UČO: 506073
# mail: 506073@mail.muni.cz
import unittest
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from GA.run.one_run import run_genetic_algorithm
from LS_periodogram.fourier_transform import fourier_transform
from utils.fourier_series_value import double_fourier_value, double_fourier_sequence
from utils.load_dataset import load_data

from LS_periodogram.clean_periodogram import iteration_scheme, dirty_spectrum_lombscargle

np.set_printoptions(threshold=np.inf)

# load data
name = 'ID1918_001'
data = load_data(name, column_names=('julian_day', 'noiseless_flux', 'noisy_flux', 'sigma', 'deviation_used'),
                 appendix='.flux')


def fitness(solution):
    """
    Fitness function
    :param solution: set of the free parameters
    :return: fitness value
    """
    m_ = 1
    x, y, delta = data['julian_day'], data['noisy_flux'], data['deviation_used']

    # Vectorized calculation of Fourier values
    y_model = double_fourier_sequence(solution, m_, x)

    # calculation of the chi^2 and returning 1/chi^2
    chi2 = np.sum((y - y_model) ** 2 / delta ** 2)
    return 1 / chi2


# ---------------------------------------------------------------------------------------------------------------------
# Periodogram of lightcurve
def tumbler_periodogram():
    result = fourier_transform(
        data['julian_day'].values,
        data['noisy_flux'].values,
        data['deviation_used'].values,
        path_graph='Results/LS/Graphs/' + name + '_graph.pdf',
        path_periodogram='Results/LS/Periodograms/' + name + '_LS.pdf'
    )

    # Example usage:
    time = data['julian_day'].values  # Time array
    flux = data['noisy_flux'].values  # Data array with noise
    dev = np.abs(data['deviation_used'].values)
    freqs = np.linspace(0.25, 10, 90000)  # Frequency array for Lomb-Scargle
    g = 0.1  # Fraction

    result_ = iteration_scheme(time, flux, dev, freqs, g, 10)
    plt.plot(freqs, dirty_spectrum_lombscargle(time, flux, dev, freqs))
    plt.plot(freqs, np.abs(result_) ** 2)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Cleaned Spectrum')
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Genetic algorithm
# add:  repair one_generation to do not crossover same gene
#          stopping function,
#       adaptive mutation...
def tumbler_genetic_algorithm_fit():
    m_ = 1
    start_time = time.time()
    final_generation = run_genetic_algorithm(
        population_size=500,
        fitness_function=fitness,
        num_genes=2 * m_ + 2 * m_ * (2 * m_ + 1) + 3,
        gene_range=(2 * m_ + 2 * m_ * (2 * m_ + 1)) * [(-0.2, 0.2)] + [(0.85, 1.15), (0.5, 1.5), (0.5, 1.5)],
        num_generations=10,
        elitism=2,
        crossover_rate=0.95,
        mutation_rate=0.01,
        mutation_range=0.5,
        name=name
    )

    days = data['julian_day'].values
    plt.plot(days, double_fourier_sequence(final_generation[0], m_, days), label='last')
    plt.plot(days, double_fourier_sequence(final_generation[4], m_, days), label='best')

    plt.scatter(days, data['noisy_flux'].values, c='gray', marker='+', s=5)
    plt.errorbar(days, data['noisy_flux'].values, yerr=data['deviation_used'].values, fmt='none', color='black',
                 elinewidth=1.5, capsize=0)

    plt.xlabel('Time [days]')
    plt.ylabel('Normalized light flux')
    plt.legend()

    plt.savefig('Results/GA/' + name + '_graph.pdf')
    plt.show()
    plt.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


# ---------------------------------------------------------------------------------------------------------------------
class TestCases(unittest.TestCase):
    def test_ls(self):
        tumbler_periodogram()

    def test_ga(self):
        tumbler_genetic_algorithm_fit()
