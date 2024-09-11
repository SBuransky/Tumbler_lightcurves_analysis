import numpy as np


def crossover(parent1: np.ndarray,
              parent2: np.ndarray,
              crossover_rate: float) -> np.ndarray:
    """
    Perform uniform crossover on two parents with real number representation using NumPy.

    Parameters:
    - parent1: NumPy array of real numbers representing the first parent.
    - parent2: NumPy array of real numbers representing the second parent.
    - crossover_rate: Probability of crossover.

    Returns:
    - Two offspring resulting from the crossover (NumPy arrays).
    """

    if np.random.rand() < crossover_rate:
        alpha = np.random.rand(len(parent1))
        return alpha * parent1 + (1 - alpha) * parent2
    else:
        return parent1 if np.random.rand() > 0.5 else parent2
