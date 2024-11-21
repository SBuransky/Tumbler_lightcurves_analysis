from typing import Tuple

import numpy as np


def parse_solution(
    solution: np.ndarray, m: int
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Parse the solution array into respective components.
    Adds one second to periods to prevent division by zero.

    :param solution: set of the Fourier coefficients, C_0, and periods
    :param m: order of the Fourier series
    :return: parsed values P, C0, Cj, Sj
    """
    P = solution[-1] + 1 / 86400  # Period
    t_0 = solution[-2]
    C0 = solution[-3]  # Constant term
    Cj = solution[m:-3]  # Cosine coefficients
    Sj = solution[:m]  # Sine coefficients

    return P, t_0, C0, Cj, Sj


'''
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
    # F = C0 + np.sum(Cj * np.cos(np.arange(1, m + 1) * omega * t) + Sj * np.sin(np.arange(1, m + 1) * omega * t))
    first_sum = Cj * np.cos(np.arange(1, m + 1) * omega * t) + Sj * np.sin(
        np.arange(1, m + 1) * omega * t
    )

    F = C0 + np.sum(first_sum)

    return F'''


def single_fourier_sequence(solution: np.ndarray, m: int, t: np.ndarray) -> np.ndarray:
    """
    Calculate Fourier values for an array of time points.

    :param solution: set of the Fourier coefficients, C_0, and period
    :param m: order of the Fourier series
    :param t: array of time points
    :return: array of Fourier values
    """

    solution = np.asarray(solution)
    P, t_0, C0, Cj, Sj = parse_solution(solution, m)
    t = np.asarray(t)

    omega = 2 * np.pi / P  # Angular frequency

    # Create an outer product of time points and harmonics
    harmonics = np.arange(1, m + 1).reshape(-1, 1)
    omega_t = omega * t

    # Compute Fourier values using vectorized operations
    cos_terms = np.dot(Cj, np.cos(harmonics * omega_t))
    sin_terms = np.dot(Sj, np.sin(harmonics * omega_t))

    return C0 + cos_terms + sin_terms
