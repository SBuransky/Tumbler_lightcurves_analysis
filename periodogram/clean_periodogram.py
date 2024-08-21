from typing import Tuple

import numpy as np

from utils.find_maxima import find_local_maxima


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
    df = 1 / (n_b * (np.max(time) - np.min(time)))

    # Calculate number of frequency bins (m)
    m = (f_max / df) + 1

    # Generate frequency grid
    freq = np.arange(0, 2 * m, 1) * df
    return freq


def fourier_transform(time: np.ndarray, data_y: np.ndarray, n_b=4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Fourier Transform on the given data_y with respect to time.

    Parameters:
    - time : numpy array containing time values.
    - data_y : numpy array containing the data values corresponding to the time points.

    Returns:
    - freq : numpy array of frequencies from the Fourier Transform.
    - wfn : numpy array, wave function corresponding to each frequency component.
    - dft : numpy array
        Discrete Fourier Transform (DFT) coefficients.
    """
    # Center time vector and data_y around their means
    tvec = time - np.mean(time)
    dvec = data_y - np.mean(data_y)

    # Generate frequency grid using the centered time vector
    freq = frequency_grid(tvec, n_b)

    # Initialize arrays for wave function and DFT coefficients
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


def clean_subtract_ccomp(wfn, dft, ccomp, l):  # TODO #TODO
    wfn_full = np.concatenate((np.conjugate(wfn[::-1]), wfn))
    # print(wfn_full)
    '''print('0',wfn)
    print('1',wfn_full)
    print('2',wfn_full[np.arange(len(dft)) + len(wfn) - l])
    print('3',wfn_full[np.arange(len(dft)) + len(wfn) + l])'''
    dft -= (ccomp * wfn_full[np.arange(len(dft)) + len(wfn) - l] +
            np.conjugate(ccomp) * wfn_full[np.arange(len(dft)) + len(wfn) + l])
    '''
    # max element = nwind - 1
    nwind = len(wfn)
    # max element = ndirt - 1
    ndirt = len(dft)
    # -------------------------------------------------------------------------
    # Compute the effect of +l component
    # -------------------------------------------------------------------------
    cplus = np.zeros(ndirt, dtype=complex)
    # index for wfn, shifted to +l comp
    index = np.arange(ndirt) - l
    # for indices less than zero take the conj(wfn[index[mask1]])
    mask1 = index < 0
    cplus[mask1] = np.conjugate(wfn[abs(index[mask1])])
    # for indices between 0 and nwind-1 take wfn[index[mask2]]
    mask2 = (index >= 0) & (index <= nwind - 1)
    cplus[mask2] = wfn[index[mask2]]
    # for indices greater than equal to nwind set to zero
    mask3 = index >= nwind
    cplus[mask3] = np.repeat([0 + 0j], len(mask3[mask3]))
    # -------------------------------------------------------------------------
    # Compute the effect of -l component
    # -------------------------------------------------------------------------
    cminus = np.zeros(ndirt, dtype=complex)
    # index for wfn, shifted to -l comp
    index = np.arange(ndirt) + l
    # for indices less than zero take the conj(wfn[index[mask1]])
    mask1 = index < 0
    cminus[mask1] = np.conjugate(wfn[abs(index[mask1])])
    # for indices between 0 and nwind-1 take wfn[index[mask2]]
    mask2 = (index >= 0) & (index <= nwind - 1)
    cminus[mask2] = wfn[index[mask2]]
    # for indices greater than equal to nwind set to zero
    mask3 = index >= nwind
    cminus[mask3] = np.repeat([0 + 0j], len(mask3[mask3]))
    # -------------------------------------------------------------------------
    # return realigned, rescaled window function .
    dft = dft - ccomp * cplus - np.conjugate(ccomp) * cminus
    '''
    return dft


def clean(freq, wfn, dft, n_iter=100, gain=0.1):
    clean_components = np.zeros(len(dft), dtype=np.complex_)
    residual_spectrum = dft

    for i in range(n_iter):
        # actual peak index
        peak = np.argmax(np.abs(residual_spectrum) ** 2)

        # amplitude of the peak
        component = gain * (residual_spectrum[peak] - np.conjugate(residual_spectrum[peak]) * wfn[2 * peak]) / (
                1 - (np.abs(wfn[2 * peak])) ** 2)

        residual_spectrum = clean_subtract_ccomp(wfn, residual_spectrum, component, peak)

        # add component to clean spectrum
        clean_components[peak] += component
        print(i)

        if np.std(np.abs(residual_spectrum) ** 2) <= 0.0035:
            print('------xxxx-----', i)
            break
    # ------------------------------------------------------------------------------------------------------------------
    # calculate beam #TODO

    # Calculate half the maximum value of the absolute waveform
    hmax = 0.5 * abs(wfn[0])

    # Find indices where the absolute waveform is less than or equal to hmax
    mask = np.where(abs(wfn) <= hmax)[0]

    # Identify the first index and corresponding waveform value from the mask
    i2 = mask[0]
    w2 = wfn[i2]

    # Linearly interpolate to get a more accurate Half Width at Half Maximum (HWHM) estimate
    if w2 < hmax:
        # If the value at i2 is less than hmax, interpolate between i2 and the previous index i1
        i1 = mask[0] - 1
        w1 = abs(wfn[i1])
        q = (hmax - w1) / (w2 - w1)
        hwidth = i1 + q * (i2 - i1)
    else:
        # If the value at i2 is equal to hmax, use i2 as the hwidth
        hwidth = i2

    # Calculate the standard deviation (sigma) of the Gaussian
    b_sigma = hwidth / np.sqrt(2.0 * np.log(2.0))

    # Calculate the Gaussian normalization constant
    const = 1.0 / (2 * b_sigma ** 2)

    # Determine the size of the restoring beam (truncated at 5 * sigma)
    n_beam = int(5 * b_sigma) + 2

    # Generate the Gaussian function
    x = np.arange(0, n_beam, 1.0)
    y = np.exp(-const * x ** 2)

    # Construct the real part of the beam
    realpart = np.append(y[::-1], y[1:n_beam])

    # Create the complex beam with the real part
    beam = np.array(realpart + 0j)
    # calculate beam #TODO
    # -------------------------------------------------------------------------------------------------------------------
    # TODO
    # Convolve "ccomp" with the "beam" to produce the "clean FT"
    # -------------------------------------------------------------------------
    # elements per half beam
    mb = int((len(beam) - 1) / 2)
    # define padding
    pad = np.repeat([0 + 0j], mb)
    # pad the data
    input_array = np.concatenate([pad, clean_components, pad])
    # convolve and recenter
    cdft = np.roll(np.convolve(input_array, beam), -mb)
    # strip padding
    cdft = cdft[mb: len(input_array) - mb] + residual_spectrum
    cdft = residual_spectrum
    # print(cdft)
    # Return
    clean_max = find_local_maxima(freq, np.abs(cdft) ** 2)
    return (freq, cdft), clean_max
