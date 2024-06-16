import numpy as np
from astropy.timeseries import LombScargle
from matplotlib import pyplot as plt


def find_local_maxima(x, y):
    y = np.array(y)
    local_maxima_indices = np.where((y[:-2] < y[1:-1]) & (y[1:-1] > y[2:]))[0] + 1

    local_maxima_x = np.array(x)[local_maxima_indices]
    local_maxima_y = y[local_maxima_indices]

    return local_maxima_x, local_maxima_y


def fourier_transform(t, y, dev, path_graph, path_periodogram):
    frequency = np.linspace(0.1, 10, 1000000)
    ls = LombScargle(t=t, y=y, dy=np.abs(dev))
    power = ls.power(frequency)

    plt.errorbar(t, y, dev)
    plt.xlabel('"JD"')
    plt.ylabel('Normalized flux')
    plt.savefig(path_graph)
    plt.show()
    plt.close()

    temp = find_local_maxima(frequency, power)

    plt.plot(frequency, power)
    plt.scatter(temp[0], temp[1])
    plt.xlabel('Frequency' + r'$[d^{-1}]$')
    plt.ylabel('Power')
    plt.savefig(path_periodogram)
    plt.show()
    plt.close()

    periodogram = frequency, power

    return periodogram,  temp
