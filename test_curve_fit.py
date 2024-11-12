import numpy as np
import pandas as pd
from test_data_generator import generate_pa_rotator
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

t, y, delta = generate_pa_rotator(
    frequency=1,
    num_periods=5,
    sampling_rate=1000,
    noise_amplitude=0.1,
    num_holes=50,
    min_hole_length=50,
    max_hole_length=200,
    num_components=3,
    seed=0,
)


def func(x, a, b, c, d, e, f, g, h, i, t):
    return (
        a
        + b * np.sin(2 * np.pi * x / t)
        + c * np.cos(2 * np.pi * x / t)
        + d * np.sin(2 * 2 * np.pi * x / t)
        + e * np.cos(2 * 2 * np.pi * x / t)
        + f * np.sin(3 * 2 * np.pi * x / t)
        + g * np.cos(3 * 2 * np.pi * x / t)
        + h * np.sin(4 * 2 * np.pi * x / t)
        + i * np.cos(4 * 2 * np.pi * x / t)
    )


plt.scatter(t, y, s=5)
popt, pcov = curve_fit(func, t, y)
print(popt)

plt.plot(t, func(t, *popt), c="red")
plt.show()
