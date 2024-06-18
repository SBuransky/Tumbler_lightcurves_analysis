# name: Samuel Buranský
# MUNI UČO: 506073
# mail: 506073@mail.muni.cz
import unittest
import numpy as np
from utils.fourier_series_value import double_fourier_value, double_fourier_sequence
from utils.load_dataset import load_data

from service.ls_service import tumbler_periodogram
from service.ga_service import tumbler_genetic_algorithm_fit

np.set_printoptions(threshold=np.inf)

# load data
name = 'ID1916_007'
data = load_data(name, column_names=('julian_day', 'noiseless_flux', 'noisy_flux', 'sigma', 'deviation_used'),
                 appendix='.txt')
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


# ---------------------------------------------------------------------------------------------------------------------
class TestCases(unittest.TestCase):
    def test_ls(self):
        tumbler_periodogram(data, name=name)

    def test_ga(self):
        tumbler_genetic_algorithm_fit(data,
                                      fitness,
                                      m_=m_,
                                      population_size=500,
                                      gene_range=((-0.2, 0.2), (0.85, 1.15), (0.5, 1.5), (0.5, 1.5)),
                                      name=name, num_generations=5)
