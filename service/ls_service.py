# Periodogram of lightcurve
def tumbler_periodogram(data, g = 0.25, n_iter = 200):
    result = fourier_transform(
        data['julian_day'].values,
        data['noisy_flux'].values,
        data['deviation_used'].values,
        path_graph='Results/LS/Graphs/' + name + '_graph.pdf',
        path_periodogram='Results/LS/Periodograms/' + name + '_LS.pdf'
    )

    # Example usage:
    time = data['julian_day'].values  # Time array
    flux = data['noisy_flux'].values  # Data array with noise
    dev = np.abs(data['deviation_used'].values)
    freqs = np.linspace(0.15, 10, 90000)  # Frequency array for Lomb-Scargle

    result_ = iteration_scheme(time, flux, dev, freqs, g, n_iter)
    plt.plot(freqs, dirty_spectrum_lombscargle(time, flux, dev, freqs))
    plt.plot(freqs, np.abs(result_) ** 2)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Cleaned Spectrum')
    plt.savefig('Results/LS/Periodograms/' + name + '_clean_LS.pdf')
    plt.show()
    plt.close()