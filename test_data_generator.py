import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_pa_rotator(
    frequency,
    num_periods,
    sampling_rate=500,
    noise_amplitude=10,
    num_holes=20,
    min_hole_length=50,
    max_hole_length=500,
    num_components=5,
    seed=42,
):
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
    sine_coefficients = np.sort(np.random.rand(num_components))[::-1]
    cosine_coefficients = np.sort(np.random.rand(num_components))[::-1]
    sine_coefficients[1:] = sine_coefficients[1:] / 2
    cosine_coefficients[1:] = cosine_coefficients[1:] / 2
    print(sine_coefficients)
    print(cosine_coefficients)

    # Generate the signal with multiple harmonics using the random coefficients
    y = np.zeros(len(t_with_holes))
    for i in range(1, num_components + 1):
        y += sine_coefficients[i - 1] * np.sin(2 * np.pi * i * frequency * t_with_holes)
        y += cosine_coefficients[i - 1] * np.cos(
            2 * np.pi * i * frequency * t_with_holes
        )

    # Add random noise to the signal
    delta = noise_amplitude * np.random.normal(size=len(t_with_holes))
    y += delta

    return t_with_holes, y, delta
