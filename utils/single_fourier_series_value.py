from typing import Tuple

import numpy as np


def parse_solution(solution: np.ndarray, m: int) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Parse the solution array into respective components.
    Adds one second to periods to prevent division by zero.

    :param solution: set of the Fourier coefficients, C_0, and periods
    :param m: order of the Fourier series
    :return: parsed values P, C0, Cj, Sj
    """
    P = solution[-1] + 1 / 86400  # Period
    C0 = solution[-2]  # Constant term
    Cj = solution[-2 - m:-2]  # Cosine coefficients
    Sj = solution[-2 - 2 * m:-2 - m]  # Sine coefficients

    return P, C0, Cj, Sj


def single_fourier_value(solution: np.ndarray, m: int, t: float) -> float:
    """
    Calculate the Fourier value at a specific time t.

    :param solution: set of the Fourier coefficients, C_0, and period
    :param m: order of the Fourier series
    :param t: time
    :return: Fourier value
    """
    solution = np.asarray(solution)
    P, C0, Cj, Sj = parse_solution(solution, m)

    omega = 2 * np.pi / P  # Angular frequency

    # Calculate Fourier value using NumPy vectorization
    F = C0 + np.sum(Cj * np.cos(np.arange(1, m + 1) * omega * t) + Sj * np.sin(np.arange(1, m + 1) * omega * t))

    return F


def single_fourier_sequence(solution: np.ndarray, m: int, t: np.ndarray) -> np.ndarray:
    """
    Calculate Fourier values for an array of time points.

    :param solution: set of the Fourier coefficients, C_0, and period
    :param m: order of the Fourier series
    :param t: array of time points
    :return: array of Fourier values
    """
    t = np.asarray(t)
    y = np.zeros(len(t))

    for i in range(len(t)):
        y[i] = single_fourier_value(solution, m, t[i])

    return y