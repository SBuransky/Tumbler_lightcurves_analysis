import numpy as np
from astropy.timeseries import LombScargle
from matplotlib import pyplot as plt


def find_local_maxima(x, y):
    y = np.array(y)
    local_maxima_indices = np.where((y[:-2] < y[1:-1]) & (y[1:-1] > y[2:]))[0] + 1

    local_maxima_x = np.array(x)[local_maxima_indices]
    local_maxima_y = y[local_maxima_indices]

    return local_maxima_x, local_maxima_y


def fourier_transform(t, y, frequency, dev=None):
    if dev is not None:
        dev = np.abs(dev)
    ls = LombScargle(t=t, y=y, dy=dev)

    power = ls.power(frequency)

    temp = find_local_maxima(frequency, power)
    periodogram = frequency, power
    return periodogram, temp
