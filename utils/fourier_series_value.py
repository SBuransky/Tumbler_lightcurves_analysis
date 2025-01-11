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
    f_psi = solution[-1] + 1 / 86400
    f_phi = solution[-2] + 1 / 86400
    t_0 = solution[-3]
    C0 = solution[-4]
    Cj0 = solution[-4 - m : -4]
    Sj0 = solution[-4 - 2 * m : -4 - m]
    Cjk = solution[: m * (2 * m + 1)]
    Sjk = solution[m * (2 * m + 1) : 2 * m * (2 * m + 1)]
    # print('p',f_psi, f_phi, t_0, C0, Cj0, Sj0, Cjk, Sjk)
    return f_psi, f_phi, t_0, C0, Cj0, Sj0, Cjk, Sjk


def double_fourier_sequence(solution: np.ndarray, m: int, t: np.ndarray) -> np.ndarray:
    """
    Calculate Fourier values for an array of time points.

    :param solution: set of the Fourier coefficients, C_0, and periods
    :param m: order of the Fourier series
    :param t: array of time points
    :return: array of Fourier values
    """
    solution = np.asarray(solution, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    f_psi, f_phi, t_0, C0, Cj0, Sj0, Cjk, Sjk = parse_solution(solution, m)

    psi, phi = 2 * np.pi * f_psi, 2 * np.pi * f_phi
    t = t - t_0  # Center time array

    # Precompute values for first sum
    indices = np.arange(1, m + 1)
    psi_t = psi * indices[:, None] * t[None, :]  # Shape (m, len(t))
    cos_term = np.cos(psi_t).T  # Shape (len(t), m)
    sin_term = np.sin(psi_t).T  # Shape (len(t), m)

    first_sum = (
        np.dot(cos_term, Cj0) +  # (len(t), m) @ (m,) -> (len(t),)
        np.dot(sin_term, Sj0)    # (len(t), m) @ (m,) -> (len(t),)
    )

    # Precompute values for second sum
    j_range = np.arange(-m, m + 1)
    k_range = np.arange(1, m + 1)
    jk_combinations = np.array(np.meshgrid(j_range, k_range)).T.reshape(-1, 2)

    psi_phi = jk_combinations[:, 0] * psi + jk_combinations[:, 1] * phi  # Shape (2m(m+1),)
    psi_phi_t = psi_phi[:, None] * t[None, :]  # Shape (2m(m+1), len(t))
    cos_values = np.cos(psi_phi_t)  # Shape (2m(m+1), len(t))
    sin_values = np.sin(psi_phi_t)  # Shape (2m(m+1), len(t))

    second_sum = (
        np.dot(Cjk, cos_values) +
        np.dot(Sjk, sin_values)
    )

    return C0 + first_sum + second_sum
