from LS_periodogram.fourier_transform import fourier_transform, fourier_transform_without_noise
from LS_periodogram.clean_periodogram import iteration_scheme, dirty_spectrum_lombscargle

import numpy as np
import matplotlib.pyplot as plt

def tumbler_periodogram(data, name, g = 0.25, n_iter = 200):
    result = fourier_transform_without_noise(
        data['julian_day'].values,
        data['noisy_flux'].values,
        #'''data['deviation_used'].values,'''
        path_graph='Results/LS/Graphs/' + name + '_graph.pdf',
        path_periodogram='Results/LS/Periodograms/' + name + '_LS.pdf'
    )

    # Example usage:
    time = data['julian_day'].values  # Time array
    flux = data['noiseless_flux'].values  # Data array with noise
    dev = 0.00001#np.abs(data['deviation_used'].values)
    freqs = np.linspace(0.001, 10, 90000)  # Frequency array for Lomb-Scargle

    result_ = iteration_scheme(time, flux, dev, freqs, g, n_iter)
    plt.plot(freqs, dirty_spectrum_lombscargle(time, flux, freqs))
    plt.plot(freqs, np.abs(result_) ** 2)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Cleaned Spectrum')
    plt.savefig('Results/LS/Periodograms/' + name + '_clean_LS.pdf')
    plt.show()
    plt.close()