import os
import matplotlib.pyplot as plt
import numpy as np
from lombscargle_periodogram.fourier_transform import lomb_scargle
from typing import Optional
from lombscargle_periodogram.clean_periodogram import frequency_grid, fourier_transform, clean
from lombscargle_periodogram.find_maxima import find_local_maxima


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
    periodogram_lomb, maximas_lomb = lomb_scargle(t, y, frequency, dev)
    periodogram_fourier = fourier_transform(t, y)[0], fourier_transform(t, y)[2]
    clean_periodogram, clean_maximas = clean(fourier_transform(t, y)[0], fourier_transform(t, y)[1], fourier_transform(t, y)[2])

    # Plot the observed data with error bars
    plt.errorbar(t, y, yerr=dev, fmt='.', label='Data')
    plt.xlabel('Julian Date (JD)')
    plt.ylabel('Normalized Flux')
    plt.title('Observed Data')
    plt.legend()
    plt.savefig(f'Results/lomb_scargle/Graphs/{name}_graph.pdf')
    plt.show()
    plt.close()

    # Plot the periodograms
    ax1 = plt.subplot(311)
    ax1.plt.plot(periodogram_lomb[0], periodogram_lomb[1], label='Periodogram')
    ax1.plt.scatter(maximas_lomb[0], maximas_lomb[1], color='red', label='Maxima')
    ax1.plt.xlabel('Frequency [d$^{-1}$]')
    ax1.plt.ylabel('Power')
    ax1.plt.title('Lomb-Scargle Periodogram')
    ax1.plt.legend()

    ax2 = plt.subplot(312)
    ax2.plt.plot(periodogram_fourier[0], periodogram_fourier[1], label='Periodogram')
    ax2.plt.xlabel('Frequency [d$^{-1}$]')
    ax2.plt.ylabel('Power')
    ax2.plt.title('Fourier Periodogram')
    ax2.plt.legend()

    ax2 = plt.subplot(313)
    ax2.plt.plot(clean_periodogram[0], clean_periodogram[1], label='Periodogram')
    ax3.plt.scatter(clean_maximas[0], clean_maximas[1], color='red', label='Maxima')
    ax2.plt.xlabel('Frequency [d$^{-1}$]')
    ax2.plt.ylabel('Power')
    ax2.plt.title('CLEAN Periodogram')
    ax2.plt.legend()
    plt.savefig(f'Results/lomb_scargle/Periodograms/{name}_PERIODOGRAM.pdf')
    plt.show()
    plt.close()

    # Save the maxima to a text file as two columns
    maxima_array = np.column_stack((maximas[0], maximas[1]))
    np.savetxt(f'Results/lomb_scargle/Results/{name}.txt', maxima_array, delimiter=" ", header='Frequency Power',
               comments='')
