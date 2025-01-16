from test_data_generator import generate_pa_rotator
from service import (
    tumbler_periodogram,
    tumbler_genetic_algorithm_fit,
    pa_rotator_genetic_algorithm_fit,
)
from utils.single_fourier_series_value import single_fourier_sequence

import numpy as np
import pandas as pd

t, y, delta = generate_pa_rotator(
    frequency=2,
    num_periods=2,
    sampling_rate=80,
    noise_amplitude=1,
    num_holes=2,
    min_hole_length=20,
    max_hole_length=50,
    num_components=10,
)

data = pd.DataFrame({"julian_day": t, "noisy_flux": y, "deviation_used": delta})
name = "test_001"
print(data)
m_ = 5


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


print(len(data))
tumbler_periodogram(
    t, y, dev=delta, name=name, n_iter=10000, gain=0.8, final_noise=0.0004
)


"""pa_rotator_genetic_algorithm_fit(
    data,
    fitness,
    m_=m_,
    population_size=500,
    gene_range=((-1, 1), (0.95, 1.05), (-1, 1), (0.95, 1.05)),
    name=name,
    num_generations=1000,
    elitism=2,
    mutation_rate=0.05,
    mutation_range=0.05,
    limit_fitness=0.001,
)
"""
