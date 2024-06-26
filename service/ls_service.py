import os
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional
from periodogram.clean_periodogram import frequency_grid, fourier_transform, clean
from periodogram.lomb_scargle import lomb_scargle
from utils.load_dataset import load_data


def tumbler_periodogram(t: np.ndarray,
                        y: np.ndarray,
                        name: str,
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
    frequency = frequency_grid(t)
    # Compute the periodogram and maxima
    periodogram_lomb, maximas_lomb = lomb_scargle(t, y, frequency, dev)
    periodogram_fourier = fourier_transform(t, y)[0], fourier_transform(t, y)[2]
    clean_periodogram, clean_maximas = clean(fourier_transform(t, y)[0],
                                             fourier_transform(t, y)[1],
                                             fourier_transform(t, y)[2],
                                             n_iter=1000, gain=0.05)

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
    ax1.plot(periodogram_lomb[0], periodogram_lomb[1], label='Lomb-Scargle Periodogram')
    ax1.scatter(maximas_lomb[0], maximas_lomb[1], color='red', label='Lomb-Scargle Maxima')
    plt.legend()

    ax2 = plt.subplot(312)
    ax2.plot(periodogram_fourier[0][:len(fourier_transform(t, y)[0]) // 2], np.abs(periodogram_fourier[1])**2, label='Fourier Periodogram')
    plt.legend()

    ax3 = plt.subplot(313)
    ax3.plot(clean_periodogram[0][:len(fourier_transform(t, y)[0]) // 2], np.abs(clean_periodogram[1]**2), label='CLEAN Periodogram')
    ax3.scatter(clean_maximas[0], np.abs(clean_maximas[1])**2, color='red', label='CLEAN Maxima')

    plt.legend()
    plt.title(f'Periodogram_{name}')
    plt.savefig(f'Results/lomb_scargle/Periodograms/{name}_PERIODOGRAM.pdf')
    plt.show()
    plt.close()

    # Save the maxima to a text file as two columns
    #maxima_array = np.column_stack((maximas[0], maximas[1]))
    #np.savetxt(f'Results/lomb_scargle/Results/{name}.txt', maxima_array, delimiter=" ", header='Frequency Power',
    #           comments='')

dat = load_data('ID1919_001')
tumbler_periodogram(dat['julian_day'].values, dat['noiseless_flux'].values, 'ID1918_001')
