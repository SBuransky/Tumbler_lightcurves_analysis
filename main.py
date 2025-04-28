# name: Samuel Buranský
# MUNI UČO: 506073
# mail: 506073@mail.muni.cz
from service import (
    tumbler_periodogram,
    tumbler_genetic_algorithm_fit,
    pa_rotator_genetic_algorithm_fit,
)
from utils.load_dataset import load_data
from utils.fourier_series_value import double_fourier_sequence
from utils.single_fourier_series_value import single_fourier_sequence
import argparse
import numpy as np
import time

# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run by these commands:
    # python main.py --periodogram to run the periodogram.
    # python main.py --genetic_algorithm to run the genetic algorithm.
    # python main.py --genetic_algorithm_pa to run the genetic algorithm.
    # python main.py --periodogram --genetic_algorithm to run both.

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run parts of the script")
    parser.add_argument(
        "--periodogram", action="store_true", help="Run the periodogram analysis"
    )
    parser.add_argument(
        "--genetic_algorithm",
        action="store_true",
        help="Run the genetic algorithm fit for NPA",
    )
    parser.add_argument(
        "--genetic_algorithm_pa",
        action="store_true",
        help="Run the genetic algorithm fit for PA",
    )
    args = parser.parse_args()

    # Load data (common to both parts)
    # For use follow these instructions:

    name = "ID1913"  # Set the name of your data file

    data = load_data(
        name,
        column_names=("julian_day", "noisy_flux", "deviation_used"),
        # column_names=("julian_day","noiseless_flux","noisy_flux","sigma","deviation_used",),
        appendix=".txt",  # Set the appendix of your data file
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
            n_iter=500,
            n_b=10,
            gain=0.5,
            final_noise=0.000008,
            dev=data["deviation_used"],
            x_border=(-0.1, 10),
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
            x, y, delta = (
                data["julian_day"],
                data["noisy_flux"],
                data["deviation_used"],
            )

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
            num_genes=2 * m_ + 2 * m_ * (2 * m_ + 1) + 4,
            gene_range=(
                [(-0.05, 0.05)] * (m_ * (2 * m_ + 1))
                + [(-0.05, 0.05)] * (m_ * (2 * m_ + 1))
                + [(-0.05, 0.05)] * m_
                + [(-0.05, 0.05)] * m_
                + [
                    (0.98, 1.02),
                    (-0.00001, 0.00001),
                    (0.26, 0.34),
                    (0.61, 0.69),
                ]  # phi, psi
            ),
            name=name,
            num_generations=15000,
            elitism=2,
            mutation_rate=0.01,
            mutation_range=np.concatenate(
                (
                    np.full(m_ * (2 * m_ + 1), 0.05),
                    np.full(m_ * (2 * m_ + 1), 0.05),
                    np.full(m_, 0.05),
                    np.full(m_, 0.05),
                    np.array([0.02, 0.000001, 0.04, 0.04]),
                )
            ),
            limit_fitness=0.001,
        )

    # Run genetic algorithm fit for PA
    if args.genetic_algorithm_pa:
        print("Running genetic algorithm fit for PA...")
        m_ = 3

        def fitness_pa(solution):
            """
            Fitness function
            :param solution: set of the free parameters
            :return: fitness value
            """
            x, y, delta = (
                data["julian_day"],
                data["noisy_flux"],
                data["deviation_used"],
            )

            # Vectorized calculation of Fourier values
            y_model = single_fourier_sequence(solution, m_, x)

            # Calculation of the chi^2 and returning 1/chi^2
            chi2 = np.sum((y - y_model) ** 2 / delta**2)
            return 1 / chi2

        pa_rotator_genetic_algorithm_fit(
            data,
            fitness_pa,
            m_=m_,
            population_size=100,
            num_genes=2 * m_ + 3,
            gene_range=np.array(
                [(-0.3, 0.3)] * m_
                + [(-0.3, 0.3)] * m_
                + [(0.98, 1.02), (-0.00001, 0.00001), (2.45, 2.55)]
            ),
            name=name,
            num_generations=20000,
            elitism=2,
            mutation_rate=0.01,
            mutation_range=np.concatenate(
                (
                    np.full(m_, 0.2),
                    np.full(m_, 0.2),
                    np.array([0.02, 0.00001, 0.02]),
                )
            ),
            limit_fitness=0.001,
        )
