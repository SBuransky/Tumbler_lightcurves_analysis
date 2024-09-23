from typing import Tuple

import numpy as np


def parse_solution(
    solution: np.ndarray, m: int
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the solution array into respective components.
    Adds one second to periods to prevent division by zero.

    :param solution: set of the Fourier coefficients, C_0, and periods
    :param m: order of the Fourier series
    :return: parsed values P_psi, P_phi, C0, Cj0, Sj0, Cjk, Sjk
    """
    P_psi = solution[-1] + 1 / 86400
    P_phi = solution[-2] + 1 / 86400
    C0 = solution[-3]
    Cj0 = solution[-3 - m : -3]
    Sj0 = solution[-3 - 2 * m : -3 - m]
    Cjk = solution[: m * (2 * m + 1)]
    Sjk = solution[m * (2 * m + 1) : 2 * m * (2 * m + 1)]

    return P_psi, P_phi, C0, Cj0, Sj0, Cjk, Sjk


'''
def double_fourier_value(solution: np.ndarray, m: int, t: float) -> float:
    """
    Calculate Fourier value at a specific time t.

    :param solution: set of the Fourier coefficients, C_0, and periods
    :param m: order of the Fourier series
    :param t: time
    :return: Fourier value
    """
    solution = np.asarray(solution)
    P_psi, P_phi, C0, Cj0, Sj0, Cjk, Sjk = parse_solution(solution, m)

    psi = 2 * np.pi / P_psi
    phi = 2 * np.pi / P_phi

    # Calculate the first sum using NumPy vectorization
    first_sum = Cj0 * np.cos(np.arange(1, m + 1) * psi * t) + Sj0 * np.sin(
        np.arange(1, m + 1) * psi * t
    )
    F = C0 + np.sum(first_sum)

    # Calculate the second sum
    for k in range(1, m + 1):
        for j in range(-m, m + 1):
            F += Cjk[m * (j + m) + k - 1] * np.cos(((psi * j) + (phi * k)) * t)
            F += Sjk[m * (j + m) + k - 1] * np.sin(((psi * j) + (phi * k)) * t)
    return F
    '''


def double_fourier_sequence(solution: np.ndarray, m: int, t: np.ndarray) -> np.ndarray:
    """
    Calculate Fourier values for an array of time points.

    :param solution: set of the Fourier coefficients, C_0, and periods
    :param m: order of the Fourier series
    :param t: array of time points
    :return: array of Fourier values
    """
    solution = np.asarray(solution)
    t = np.asarray(t)

    P_psi, P_phi, C0, Cj0, Sj0, Cjk, Sjk = parse_solution(solution, m)

    psi = 2 * np.pi / P_psi
    phi = 2 * np.pi / P_phi

    # Time array reshaped for broadcasting
    t = t[:, np.newaxis]

    # First sum vectorized across all-time points
    cos_term = np.cos(np.arange(1, m + 1) * psi * t)
    sin_term = np.sin(np.arange(1, m + 1) * psi * t)
    first_sum = np.dot(cos_term, Cj0) + np.dot(sin_term, Sj0)

    j_range = np.arange(-m, m + 1)
    k_range = np.arange(1, m + 1)

    jk_combinations = np.array(np.meshgrid(j_range, k_range)).T.reshape(-1, 2)
    psi_phi_t = jk_combinations[:, 0] * psi + jk_combinations[:, 1] * phi

    cos_values = np.cos(psi_phi_t * t)
    sin_values = np.sin(psi_phi_t * t)

    second_sum = np.dot(cos_values, Cjk) + np.dot(sin_values, Sjk)

    return C0 + first_sum + second_sum
