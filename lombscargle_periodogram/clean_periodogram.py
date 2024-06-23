import matplotlib.pyplot as plt
import numpy as np
from utils.load_dataset import load_data
from typing import Tuple

from fourier_transform import lomb_scargle


def frequency_grid(time: np.ndarray, n_b=4) -> np.ndarray:
    """
        Generate a frequency grid based on the given time array and number of bins.

        Parameters:
        - time: numpy array containing time values, assumed to be sorted
        - n_b: int, optional, number of bins for frequency grid. Default is 4.

        Returns:
        - numpy array of frequencies representing the frequency grid.
    """
    # Calculate maximum resolvable frequency (f_max)
    f_max = 1 / (2 * np.min(np.diff(time)))

    # Calculate frequency resolution (df)
    df = 1 / (np.max(time) - np.min(time))

    # Calculate number of frequency bins (m)
    m = n_b * (f_max / df)

    # Generate frequency grid
    freq = np.arange(0, 2 * (m + 1), 1) * df

    return freq


def fourier_transform(time: np.ndarray, data_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Fourier Transform on the given data_y with respect to time.

    Parameters:
    - time : numpy array containing time values.
    - data_y : numpy array containing the data values corresponding to the time points.

    Returns:
    - freq : numpy array of frequencies from the Fourier Transform.
    - wfn : numpy array, wavefunction corresponding to each frequency component.
    - dft : numpy array
        Discrete Fourier Transform (DFT) coefficients.
    """
    # Center time vector and data_y around their means
    tvec = time - np.mean(time)
    dvec = data_y - np.mean(data_y)

    # Generate frequency grid using the centered time vector
    freq = frequency_grid(tvec)

    # Initialize arrays for wavefunction and DFT coefficients
    wfn = np.zeros(len(freq), dtype=complex)
    dft = np.zeros(int(len(freq) / 2), dtype=complex)

    # Compute Fourier Transform components
    for i in range(len(freq)):
        phase = -2 * np.pi * 1j * freq[i] * tvec
        if i < int(len(freq) / 2):
            wfn[i] = np.sum(np.exp(phase)) / len(tvec)
            dft[i] = np.sum(dvec * np.exp(phase)) / len(tvec)
        else:
            wfn[i] = np.sum(np.exp(phase)) / len(tvec)

    return freq, wfn, dft


def clean_alpha(dirty, window, nu):
    alpha = (dirty[nu] - np.conjugate(dirty[nu]) * window[2 * nu]) / (1 - (np.abs(window[2 * nu])) ** 2)
    return alpha


def comp_substract_clean(wfn, dirty, comp, freq_comp):
    n_window = len(wfn)
    n_dirty = len(dirty)

    return


def clean_beam(wfn):
    return


def clean(freq, wfn, dft, n_iter=100, g=0.5):
    R = dft

    clean_comp = np.zeros_like(dft)

    beam = clean_beam(wfn)

    # CLEAN LOOP for n_iter times
    for i in range(n_iter):
        peak = np.argmax(R)
        component = g * clean_alpha(R, wfn, peak)

        R = comp_substract_clean(wfn, R, component, peak)
        clean_comp[peak] += component

    # elements per half beam
    mb = int((len(beam) - 1) / 2)
    pad = np.repeat([0 + 0j], mb)
    input_array = np.concatenate([pad, ccomp, pad])

    # convolve
    CLEAN = np.roll(convolve(input_array, beam), -mb)
    # strip padding
    CLEAN = CLEAN[mb: len(input_array) - mb]
    return CLEAN


dat = load_data('ID1918_001')
freq, w, d = fourier_transform(dat['julian_day'].values, dat['noiseless_flux'].values)
plt.plot(freq[:len(freq) // 2], np.abs(d))
plt.plot(lomb_scargle(dat['julian_day'].values, dat['noiseless_flux'].values, freq)[0][0],
         lomb_scargle(dat['julian_day'].values, dat['noiseless_flux'].values, freq)[0][1])
plt.show()
plt.close()
