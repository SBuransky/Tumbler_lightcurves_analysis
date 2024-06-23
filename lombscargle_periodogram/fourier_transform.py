import numpy as np
from astropy.timeseries import LombScargle
from matplotlib import pyplot as plt

from typing import Tuple, Optional


def find_local_maxima(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds local maxima in an array of y values corresponding to given x values.

    Args:
    - x (np.ndarray): Array of x values.
    - y (np.ndarray): Array of y values.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Tuple containing arrays of local maxima x and y values.
    """
    # Convert y to np.ndarray if it's not already
    y = np.array(y)

    # Find indices where y[i-1] < y[i] > y[i+1]
    local_maxima_indices = np.where((y[:-2] < y[1:-1]) & (y[1:-1] > y[2:]))[0] + 1

    # Extract x and y values at local maxima indices
    local_maxima_x = x[local_maxima_indices]
    local_maxima_y = y[local_maxima_indices]

    return local_maxima_x, local_maxima_y


def lomb_scargle(t: np.ndarray,
                 y: np.ndarray,
                 frequency: np.ndarray,
                 dev: Optional[np.ndarray] = None) \
        -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Computes the Lomb-Scargle periodogram and finds local maxima.

    Args:
    - t (np.ndarray): Array of time values.
    - y (np.ndarray): Array of y values.
    - frequency (np.ndarray): Array of frequencies at which to compute the periodogram.
    - dev (Optional[np.ndarray]): Array of uncertainties in y values (optional).

    Returns:
    - Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        - Tuple containing arrays of frequency and corresponding power values.
        - Tuple containing arrays of frequency and power values at local maxima.
    """
    if dev is not None:
        dev = np.abs(dev)

    # Compute Lomb-Scargle periodogram
    ls = LombScargle(t=t, y=y, dy=dev)
    power = ls.power(frequency)

    # Find local maxima in the periodogram
    maxima_x, maxima_y = find_local_maxima(frequency, power)

    # Prepare periodogram and local maxima data for return
    periodogram = (frequency, power)
    local_maxima = (maxima_x, maxima_y)

    return periodogram, local_maxima
