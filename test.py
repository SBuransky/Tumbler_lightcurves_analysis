import numpy as np
import matplotlib.pyplot as plt


def generate_sine_wave(frequency, num_periods, sampling_rate=1000):
    period = 1 / frequency
    total_time = 3 * num_periods * period

    t = np.linspace(0, total_time, int(total_time * sampling_rate), endpoint=False)
    print(len(t))
    indices_to_remove = np.concatenate((np.arange(0, 100),
                                        np.arange(200, 300),
                                        np.arange(500, 650),
                                        np.arange(750, 950),
                                        np.arange(1000, 1200),
                                        np.arange(1300, 1450),
                                        np.arange(1500, 1650),
                                        np.arange(1700, 1750),
                                        np.arange(2000, 2200),
                                        np.arange(2350, 2500),
                                        np.arange(2800, 2950),
                                        np.arange(3200, 3350),
                                        np.arange(3500, 3700),
                                        np.arange(4000, 4200)))
    print(indices_to_remove)
    t = np.delete(t, indices_to_remove)

    y = 4 * np.sin(2 * np.pi * 5 * t) + \
        4 * np.sin(2 * np.pi * 4 * t) + \
        6 * np.cos(2 * np.pi * 3 * t) + \
        5 * np.cos(2 * np.pi * 2 * t) + \
        8 * np.random.normal(size=len(t)) + \
        4 * np.random.normal(size=len(t))

    return t, y


# Example usage
frequency = 2  # Frequency in Hz
num_periods = 3  # Number of periods
sampling_rate = 1000  # Sampling rate in samples per second

# Generate the sine wave
t, y = generate_sine_wave(frequency, num_periods, sampling_rate)

from main import tumbler_periodogram
tumbler_periodogram(t, y, name='test', n_iter=30000, gain=0.09)
