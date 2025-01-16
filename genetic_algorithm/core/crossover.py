import numpy as np
from numba import jit


@jit(nopython=True, parallel=True)
def crossover(
    parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float
) -> np.ndarray:
    """
    Perform uniform crossover on two parents with real number representation using NumPy.

    Parameters:
    - parent1: NumPy array of real numbers representing the first parent.
    - parent2: NumPy array of real numbers representing the second parent.
    - crossover_rate: Probability of crossover.

    Returns:
    - A single offspring resulting from the crossover (NumPy array).
    """
    n_genes = parent1.shape[0]
    rand_values = np.random.random(2)  # Generate two random values
    if rand_values[0] < crossover_rate:
        alpha = np.random.random(n_genes)  # Alpha for weighted crossover
        offspring = alpha * parent1 + (1 - alpha) * parent2
    else:
        offspring = parent1 if rand_values[1] > 0.5 else parent2
    return offspring
