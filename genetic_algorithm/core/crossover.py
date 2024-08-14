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
    # Ensure parents have the same length
    assert len(parent1) == len(parent2), "Parents must have the same length."

    if np.random.rand() < crossover_rate:
        alpha = np.random.rand(len(parent1))
        offspring = alpha * parent1 + (1 - alpha) * parent2
    else:
        offspring = [parent1, parent2][np.random.choice([0, 1])]

    return offspring
