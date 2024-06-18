import numpy as np
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt


def dirty_spectrum_lombscargle(time, data, freqs):
    # Compute the dirty spectrum Fs(n) using Lomb-Scargle periodogram
    lomb_scargle = LombScargle(time, data)
    power = lomb_scargle.power(freqs)
    return power


def complex_amplitude(residual, npeak):
    # Calculate the complex amplitude a(npeak)
    return residual[npeak]


def calculate_residual_spectrum(residual, a_npeak, npeak, g, S, freqs):
    # Use Eq. (3) to calculate the contribution of a(npeak) to the dirty spectrum
    contribution = g * (a_npeak * S(freqs - npeak) + np.conj(a_npeak) * S(freqs + npeak))
    # Form the residual spectrum Ri by subtracting the contribution from Ri-1
    return residual - contribution


def S(freqs):
    # Example implementation of the function S, replace with the actual model if needed
    return np.exp(-freqs ** 2 / 2)


def convolve_with_gaussian(clean_component_array, sigma):
    # Convolve the clean component array with a Gaussian function
    gaussian = np.exp(
        -np.arange(-len(clean_component_array) // 2, len(clean_component_array) // 2) ** 2 / (2 * sigma ** 2))
    return np.convolve(clean_component_array, gaussian, mode='same')


def iteration_scheme(time, data, dev, freqs, g, max_iterations=500, tolerance=1e-6):
    # Step 1: Compute the dirty spectrum Fs(n) using Lomb-Scargle
    Fs = dirty_spectrum_lombscargle(time, data, freqs)

    # Step 2: Initialize the residual spectrum R0 = Fs
    residual = Fs.copy()

    # Initialize the clean component array
    clean_component_array = np.zeros_like(Fs)

    for i in range(max_iterations):
        # Step 3: Find the maximum frequency npeak in the previous residual spectrum
        npeak = np.argmax(np.abs(residual))
        a_npeak = complex_amplitude(residual, npeak)

        # Step 4: Calculate the contribution of a(npeak) and form the residual spectrum Ri
        new_residual = calculate_residual_spectrum(residual, a_npeak, freqs[npeak], g, S, freqs)

        # Store the subtracted fraction ga_i to the clean component array
        clean_component_array[npeak] += g * a_npeak
        clean_component_array[-npeak] += g * np.conj(a_npeak)

        # Check for convergence
        if np.linalg.norm(new_residual - residual) < tolerance:
            break

        residual = new_residual

    # Step 5: Convolve the clean component array with a Gaussian function
    clean_component_array = convolve_with_gaussian(clean_component_array, sigma=1.0)

    # Add the residual spectrum of the last iteration
    final_spectrum = clean_component_array + residual

    return final_spectrum
