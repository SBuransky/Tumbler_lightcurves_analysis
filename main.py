# name: Samuel Buranský
# MUNI UČO: 506073
# mail: 506073@mail.muni.cz

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

# set part_to_run to 'GA' for Genetic algorithm or 'LS' for Lomb-Scargle periodogram or 'GA_LS', 'LS_GA' for both
part_to_run = 'LS'
# part_to_run = 'GA'
# part_to_run = 'LS_GA'

# load data
name = 'ID1918_001'
data = load_data(name, column_names=('julian_day', 'noiseless_flux', 'noisy_flux', 'sigma', 'deviation_used'),
                 appendix='.flux')

# ---------------------------------------------------------------------------------------------------------------------
# Periodogram of lightcurve
if part_to_run in ['LS', 'GA_LS', 'LS_GA']:
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
if part_to_run in ['GA', 'GA_LS', 'LS_GA']:
    start_time = time.time()
    m_ = 1


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


    final_generation = run_genetic_algorithm(
        population_size=500,
        fitness_function=fitness,
        num_genes=2 * m_ + 2 * m_ * (2 * m_ + 1) + 3,
        gene_ran