import matplotlib.pyplot as plt
import numpy as np
from utils.load_dataset import load_data

def frequency_grid(time, n_b=4):
    f_max = 1 / (2 * np.min(np.diff(time)))
    df = 1 / (np.max(time) - np.min(time))
    m = n_b * (f_max / df)
    freq = np.arange(0, m + 1, 1) * df
    return freq


def fourier_transform(time, data_y):
    tvec = time - np.mean(time)
    dvec = data_y - np.mean(data_y)
    freq = frequency_grid(tvec)

    wfn = np.zeros(len(freq), dtype=complex)
    dft = np.zeros(int(len(freq) / 2), dtype=complex)

    for i in range(len(freq)):
        phase = -2 * np.pi * freq[i] * tvec
        phvec = np.array(np.cos(phase) + 1j * np.sin(phase))
        if i < int(len(freq) / 2):
            wfn[i] = np.sum(phvec) / len(tvec)
            dft[i] = np.sum(dvec * phvec) / len(tvec)
        # complete the spectral window function
        else:
            wfn[i] = np.sum(phvec) / len(tvec)
    print(len(freq))
    print(len(wfn))
    print(len(dft))
    return freq, wfn, dft

dat = load_data('ID1918_001')
freq, w, d = fourier_transform(dat['julian_day'].values, dat['noiseless_flux'].values)
plt.plot(freq, np.abs(w))
plt.show()
plt.close()

def clean():
    pass
