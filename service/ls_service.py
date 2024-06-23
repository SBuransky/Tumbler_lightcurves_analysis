import os
import matplotlib.pyplot as plt
import numpy as np
from lombscargle_periodogram.fourier_transform import lomb_scargle
from typing import Optional


def tumbler_periodogram(t: np.ndarray,
                        y: np.ndarray,
                        name: str,
                        frequency: np.ndarray = np.linspace(0.1, 10, 10000),
                        dev: Optional[np.ndarray] = None) -> None:
    """
    Compute Lomb-Scargle periodogram and plot results.

    Parameters:
    - t: Array of time values (Julian Date).
    - y: Array of corresponding flux values.
    - name: Name to use for saving plots and results.
    - frequency: Array of frequencies to evaluate the periodogram.
    - dev: Optional array of uncertainties in flux values (error bars).

    Returns:
    - None. Save plots and results to disk.
    """
    # Ensure directories exist
    os.makedirs('Results/lomb_scargle/Graphs/', exist_ok=True)
    os.makedirs('Results/lomb_scargle/Periodograms/', exist_ok=True)
    os.makedirs('Results/lomb_scargle/Results/', exist_ok=True)

    # Compute the periodogram and maxima
    periodogram, maximas = lomb_scargle(t, y, frequency, dev)

    # Plot the observed data with error bars
    plt.errorbar(t, y, yerr=dev, fmt='.', label='Data')
    plt.xlabel('Julian Date (JD)')
    plt.ylabel('Normalized Flux')
    plt.title('Observed Data')
    plt.legend()
    plt.savefig(f'Results/lomb_scargle/Graphs/{name}_graph.pdf')
    plt.show()
    plt.close()

    # Plot the periodogram
    plt.plot(periodogram[0], periodogram[1], label='Periodogram')
    plt.scatter(maximas[0], maximas[1], color='red', label='Maxima')
    plt.xlabel('Frequency [d$^{-1}$]')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle Periodogram')
    plt.legend()
    plt.savefig(f'Results/lomb_scargle/Periodograms/{name}_LS.pdf')
    plt.show()
    plt.close()

    # Save the maxima to a text file as two columns
    maxima_array = np.column_stack((maximas[0], maximas[1]))
    np.savetxt(f'Results/lomb_scargle/Results/{name}_LS.txt', maxima_array, delimiter=" ", header='Frequency Power',
               comments='')
