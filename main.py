# name: Samuel Buranský
# MUNI UČO: 506073
# mail: 506073@mail.muni.cz
from service import tumbler_periodogram, tumbler_genetic_algorithm_fit
from utils.load_dataset import load_data
from utils.fourier_series_value import double_fourier_sequence
import argparse
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    # load data
    name = 'ID1919_007'
    data = load_data(name, column_names=('julian_day', 'noiseless_flux', 'noisy_flux', 'sigma', 'deviation_used'),
                     appendix='.flux')
    # periodogram LS and CLEAN Fourier
    tumbler_periodogram(data['julian_day'].values, data['noiseless_flux'].values,
                        name=name, n_iter=100000, n_b=20, gain=0.5, final_noise=0.000009, dev=data['deviation_used'])

    # fit by GA
    m_ = 2

    def fitness(solution):
        """
        Fitness function
        :param solution: set of the free parameters
        :return: fitness value
        """
        x, y, delta = data['julian_day'], data['noisy_flux'], data['deviation_used']

        # Vectorized calculation of Fourier values
        y_model = double_fourier_sequence(solution, m_, x)

        # calculation of the chi^2 and returning 1/chi^2
        chi2 = np.sum((y - y_model) ** 2 / delta ** 2)
        return 1 / chi2


    tumbler_genetic_algorithm_fit(data,
                                  fitness,
                                  m_=m_,
                                  population_size=100,
                                  gene_range=((-0.2, 0.2), (0.90, 1.10), (0, 4), (0, 4)),
                                  name=name,
                                  num_generations=200,
                                  elitism=1,
                                  mutation_rate=0.05,
                                  mutation_range=0.05)'''


if __name__ == "__main__":
    # Run by these commands:
    # python main.py --periodogram to run the periodogram.
    # python main.py --genetic_algorithm to run the genetic algorithm.
    # python main.py --periodogram --genetic_algorithm to run both.

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run parts of the script")
    parser.add_argument(
        "--periodogram", action="store_true", help="Run the periodogram analysis"
    )
    parser.add_argument(
        "--genetic_algorithm", action="store_true", help="Run the genetic algorithm fit"
    )
    args = parser.parse_args()

    # Load data (common to both parts)
    name = "ID1918_001"
    data = load_data(
        name,
        column_names=(
            "julian_day",
            "noiseless_flux",
            "noisy_flux",
            "sigma",
            "deviation_used",
        ),
        appendix=".flux",
    )

    # Run periodogram LS and CLEAN Fourier
    if args.periodogram:
        print("Running periodogram analysis...")
        tumbler_periodogram(
            data["julian_day"].values,
            data["noiseless_flux"].values,
            name=name,
            n_iter=100000,
            n_b=20,
            gain=0.5,
            final_noise=0.000025,
            dev=data["deviation_used"],
        )

    # Run genetic algorithm fit
    if args.genetic_algorithm:
        print("Running genetic algorithm fit...")
        m_ = 1

        def fitness(solution):
            """
            Fitness function
            :param solution: set of the free parameters
            :return: fitness value
            """
            x, y, delta = data["julian_day"], data["noisy_flux"], data["deviation_used"]

            # Vectorized calculation of Fourier values
            y_model = double_fourier_sequence(solution, m_, x)

            # Calculation of the chi^2 and returning 1/chi^2
            chi2 = np.sum((y - y_model) ** 2 / delta**2)
            return 1 / chi2

        tumbler_genetic_algorithm_fit(
            data,
            fitness,
            m_=m_,
            population_size=20,
            gene_range=((-0.2, 0.2), (0.95, 1.05), (0.65, 0.95), (1.65, 1.85)),
            name=name,
            num_generations=10,
            elitism=2,
            mutation_rate=0.05,
            mutation_range=0.05,
        )
