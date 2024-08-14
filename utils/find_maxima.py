from typing import Tuple

import numpy as np


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
