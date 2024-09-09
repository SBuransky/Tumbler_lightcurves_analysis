import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_pa_rotator(frequency,
                        num_periods,
                        sampling_rate=500,
                        noise_amplitude=10,
                        num_holes=20,
                        min_hole_length=50,
                        max_hole_length=500,
                        num_components=5,
                        seed=42):
    """
    Generate a noisy sine wave with random long holes and random coefficients.

    Parameters:
    - frequency: Base frequency of the sine wave.
    - num_periods: Number of periods to generate.
    - sampling_rate: Number of samples per second.
    - noise_amplitude: Amplitude of the added noise.
    - num_holes: Number of random holes to generate.
    - min_hole_length: Minimum length of each hole.
    - max_hole_length: Maximum length of each hole.
    - num_components: Number of sine/cosine components to generate.
    - seed: Random seed for reproducibility.

    Returns:
    - t: Time array with holes.
    - y: Signal array with noise, random coefficients, and holes.
    """

    np.random.seed(seed)

    # Calculate total time and generate time array
    period = 1 / frequency
    total_time = num_periods * period
    t = np.linspace(0, total_time, int(total_time * sampling_rate), endpoint=False)

    # Randomly generate long holes
    num_points = len(t)
    hole_indices = []

    for _ in range(num_holes):
        # Randomly choose a starting point for the hole
        start_index = np.random.randint(0, num_points - max_hole_length)
        # Randomly choose the length of the hole within the specified range
        hole_length = np.random.randint(min_hole_length, max_hole_length)
        # Create the hole by adding the indices to be removed
        hole_indices.append(np.arange(start_index, start_index + hole_length))

    # Combine all hole indices and remove them from the time array
    hole_indices = np.concatenate(hole_indices)
    hole_indices = np.unique(hole_indices)  # Ensure no duplicate indices
    t_with_holes = np.delete(t, hole_indices)

    # Randomly generate coefficients for the sine and cosine components
    sine_coefficients = np.random.rand(num_components)
    cosine_coefficients = np.random.rand(num_components)
    print(sine_coefficients)
    print(cosine_coefficients)

    # Generate the signal with multiple harmonics using the random coefficients
    y = np.zeros(len(t_with_holes))
    for i in range(1, num_components + 1):
        y += sine_coefficients[i - 1] * np.sin(2 * np.pi * i * frequency * t_with_holes)
        y += cosine_coefficients[i - 1] * np.cos(2 * np.pi * i * frequency * t_with_holes)

    # Add random noise to the signal
    delta = noise_amplitude * np.random.normal(size=len(t_with_holes))
    y += delta

    return t_with_holes, y, delta


# Example usage with random long holes and random coefficients
t, y, delta = generate_pa_rotator(frequency=1,
                                  num_periods=5,
                                  sampling_rate=1000,
                                  noise_amplitude=0.001,
                                  num_holes=20,
                                  min_hole_length=50,
                                  max_hole_length=200,
                                  num_components=2,
                                  seed=0)

from main import tumbler_periodogram, tumbler_genetic_algorithm_fit
from utils.single_fourier_series_value import single_fourier_value
tumbler_periodogram(t, y, name='test', n_iter=10000, gain=0.1, final_noise=0.0028)

data = pd.DataFrame({'julian_day': t, 'noisy_flux': y, 'deviation_used': delta})


m_ = 1
name = 'test'
def fitness(solution):
    """
    Fitness function
    :param solution: set of the free parameters
    :return: fitness value
    """
    x, y, delta = data['julian_day'], data['noisy_flux'], data['deviation_used']

    # Vectorized calculation of Fourier values
    y_model = single_fourier_value(solution, m_, x)

    # calculation of the chi^2 and returning 1/chi^2
    chi2 = np.sum((y - y_model) ** 2 / delta ** 2)
    return 1 / chi2


tumbler_genetic_algorithm_fit(data,
                              fitness,
                              m_=m_,
                              population_size=100,
                              gene_range=((-100, 100), (-100, 100), (0, 2), (0, 2)),
                              name=name, num_generations=100, elitism=1, mutation_rate=0.05, mutation_range=0.05)
