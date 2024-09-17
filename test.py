from test_data_generator import generate_pa_rotator
from service import (
    tumbler_periodogram,
    tumbler_genetic_algorithm_fit,
    pa_rotator_genetic_algorithm_fit,
)
from utils.single_fourier_series_value import (
    single_fourier_value,
    single_fourier_sequence,
)

import numpy as np
import pandas as pd

t, y, delta = generate_pa_rotator(
    frequency=1,
    num_periods=5,
    sampling_rate=1000,
    noise_amplitude=0.1,
    num_holes=10,
    min_hole_length=50,
    max_hole_length=200,
    num_components=10,
    seed=0,
)

data = pd.DataFrame({"julian_day": t, "noisy_flux": y, "deviation_used": delta})
name = "test"
m_ = 3


def fitness(solution):
    """
    Fitness function
    :param solution: set of the free parameters
    :return: fitness value
    """
    x, y, delta = data["julian_day"], data["noisy_flux"], data["deviation_used"]

    # Vectorized calculation of Fourier values
    y_model = single_fourier_sequence(solution, m_, x)

    # calculation of the chi^2 and returning 1/chi^2
    chi2 = np.sum((y - y_model) ** 2 / delta**2)
    return 1 / chi2


tumbler_periodogram(t, y, name=name, n_iter=10000, gain=0.1, final_noise=0.0028)

pa_rotator_genetic_algorithm_fit(
    data,
    fitness,
    m_=m_,
    population_size=200,
    gene_range=((-2, 2), (-0.01, 0.01), (0.9, 1.1)),
    name=name,
    num_generations=10,
    elitism=1,
    mutation_rate=0.2,
    mutation_range=0.005,
)
