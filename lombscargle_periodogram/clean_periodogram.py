import matplotlib.pyplot as plt
import numpy as np
from utils.load_dataset import load_data

from fourier_transform import lomb_scargle


def frequency_grid(time, n_b=4):
    f_max = 1 / (2 * np.min(np.diff(time)))
    df = 1 / (np.max(time) - np.min(time))
    m = n_b * (f_max / df)
    freq = np.arange(0, 2 * (m + 1), 1) * df
    return freq


def fourier_transform(time, data_y):
    tvec = time - np.mean(time)
    dvec = data_y - np.mean(data_y)
    freq = frequency_grid(tvec)

    wfn = np.zeros(len(freq), dtype=complex)
    dft = np.zeros(int(len(freq) / 2), dtype=complex)

    for i in range(len(freq)):
        phase = -2 * np.pi * 1j * freq[i] * tvec
        # phvec = np.array(np.cos(phase) + 1j * np.sin(phase))
        if i < int(len(freq) / 2):
            wfn[i] = np.sum(np.e ** phase) / len(tvec)  # np.sum(phvec) / len(tvec)
            dft[i] = np.sum(dvec * np.e ** phase) / len(tvec)  # np.sum(dvec * phvec) / len(tvec)
        else:
            wfn[i] = np.sum(np.e ** phase) / len(tvec)  # np.sum(phvec) / len(tvec)
    return freq, wfn, dft


def clean():
    pass


dat = load_data('ID1918_001')
freq, w, d = fourier_transform(dat['julian_day'].values, dat['noiseless_flux'].values)
plt.plot(freq[:len(freq) // 2], np.abs(d))
plt.plot(lomb_scargle(dat['julian_day'].values, dat['noiseless_flux'].values, freq)[0][0],
         lomb_scargle(dat['julian_day'].values, dat['noiseless_flux'].values, freq)[0][1])
plt.show()
plt.close()

