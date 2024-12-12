# name: Samuel Buranský
# MUNI UČO: 506073
# mail: 506073@mail.muni.cz
from service import tumbler_periodogram, tumbler_genetic_algorithm_fit
from utils.load_dataset import load_data
from utils.fourier_series_value import double_fourier_sequence
import argparse
import numpy as np
import time

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
    name = "ID1917_001"
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
    data["julian_day"] -= min(data["julian_day"])
    print(len(data["julian_day"]))

    # Run periodogram LS and CLEAN Fourier
    if args.periodogram:
        print("Running periodogram analysis...")
        tumbler_periodogram(
            data["julian_day"].values,
            data["noisy_flux"].values,
            name=name,
            n_iter=1000,
            n_b=10,
            gain=0.5,
            final_noise=0.00005,
            dev=data["deviation_used"],
        )

    # Run genetic algorithm fit
    if args.genetic_algorithm:
        print("Running genetic algorithm fit...")
        m_ = 3

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
            population_size=200,
            gene_range=(
                (-0.1, 0.1),
                (0.98, 1.02),
                (-2, 2),
                (0.65, 1.05),
                (1.10, 1.50),
            ),
            # ID1916_001
            # 6.996631977048450857e-01  7.956271751566013420e-03    1.429259111
            # 1.166105329508075217e+00  4.712898849962869272e-03    0.857555466
            # ID1917_001 --------------------------------------
            # 7.762803903918541470e-01  1.025301180847258339e-02    1.288194333
            # 1.164420585587781165e+00  4.728515400644272941e-03    0.858796222
            # ID1918_001 --------------------------------------
            # 5.812703081591484855e-01  7.293740081353852420e-03    1.720370000
            # 1.259419001011488515e+00  8.964707699941268718e-03    0.794016923
            # ID1919_001 --------------------------------------
            # 1.346956520690677950e+00  3.704314186094919206e-03    0.742414462
            # 2.020434781036017036e+00  2.677347356997863189e-03    0.494942974
            # 2.745719061407920325e+00  3.487168691736520620e-03    0.364203321
            name=name,
            num_generations=10000,
            elitism=2,
            mutation_rate=0.007,
            mutation_range=0.1,
            limit_fitness=0.001
        )
