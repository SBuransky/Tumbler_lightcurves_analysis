from typing import Tuple, Optional

import numpy as np
from astropy.timeseries import LombScargle

from utils.find_maxima import find_local_maxima


def lomb_scargle(t: np.ndarray,
                 y: np.ndarray,
                 frequency: np.ndarray,
                 dev: Optional[np.ndarray] = None,
                 dev_use_for_ls=None) \
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
    if dev_use_for_ls == None:
        ls = LombScargle(t=t, y=y)
    else:
        ls = LombScargle(t=t, y=y, dy=dev)

    power = ls.power(frequency)

    # Find local maxima in the periodogram
    maxima_x, maxima_y = find_local_maxima(frequency, power)

    # Prepare periodogram and local maxima data for return
    periodogram = (frequency, power)
    local_maxima = (maxima_x, maxima_y)

    return periodogram, local_maxima
