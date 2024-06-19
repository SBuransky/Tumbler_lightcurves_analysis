import os
import matplotlib.pyplot as plt
import numpy as np
from lombscargle_periodogram.fourier_transform import fourier_transform


def tumbler_periodogram(t, y, name, frequency=np.linspace(0.1, 10, 10000), dev=None):
    # Ensure directories exist
    os.makedirs('Results/lomb_scargle/Graphs/', exist_ok=True)
    os.makedirs('Results/lomb_scargle/Periodograms/', exist_ok=True)
    os.makedirs('Results/lomb_scargle/Results/', exist_ok=True)

    # Compute the periodogram and maxima
    periodogram, maximas = fourier_transform(t, y, frequency, dev)

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
