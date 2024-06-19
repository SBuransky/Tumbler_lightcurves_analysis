from lombscargle_periodogram.fourier_transform import fourier_transform
import matplotlib.pyplot as plt
import numpy as np

def tumbler_periodogram(t, y, name, frequency=np.linspace(0.1, 10, 10000), dev=None):
    periodogram, maximas = fourier_transform(t, y, frequency, dev)

    plt.errorbar(t, y, dev)
    plt.xlabel('"JD"')
    plt.ylabel('Normalized flux')
    plt.savefig('Results/lomb_scargle/Graphs/' + name + '_graph.pdf')
    plt.show()
    plt.close()

    plt.plot(periodogram[0], periodogram[1])
    plt.scatter(maximas[0], maximas[1])
    plt.xlabel('Frequency' + r'$[d^{-1}]$')
    plt.ylabel('Power')
    plt.savefig('Results/lomb_scargle/Periodograms/' + name + '_LS.pdf')
    plt.show()
    plt.close()

    np.savetxt('Results/lomb_scargle/Results/' + name + '_LS.txt', maximas, delimiter=" ")
