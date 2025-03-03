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
    sampling_rate=1440 / 10,
    noise_amplitude=0.03,
    num_holes=2,
    min_hole_length=20,
    max_hole_length=1440 / 20,
    num_components=8,
    seed=0,
)

data = pd.DataFrame({"julian_day": t, "noisy_flux": y, "deviation_used": delta})
name = "ID0003"
data.to_csv("data/" + name + ".txt", sep="\t", index=False)

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
    t, y, dev=delta, name=name, n_iter=10000, gain=0.5, final_noise=0.0005
)

"""pa_rotator_genetic_algorithm_fit(
    data,
    fitness,
    m_=m_,
    population_size=500,
    num_genes=2 * m_ + 3,
    gene_range=np.array(
        [(-0.5, 0.5)] * m_ + [(-0.5, 0.5)] * m_ + [(0.98, 1.02), (-0.00001, 0.00001), (1.98, 2.02)]
    ),
    name=name,
    num_generations=10000,
    elitism=2,
    mutation_rate=0.01,
    mutation_range=np.concatenate(
        (
            np.full(m_, 0.5),
            np.full(m_, 0.5),
            np.array([0.02, 0.00001, 0.02]),
        )
    ),
    limit_fitness=0.002,
)"""
