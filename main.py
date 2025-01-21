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
    name = "ID1916_003"
    data = load_data(
        name,
        column_names=(
            "julian_day",
            "noiseless_flux",
            "noisy_flux",
            "sigma",
            "deviation_used",
        ),
        appendix=".txt",
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
            n_b=4,
            gain=0.1,
            final_noise=0.00002,
            dev=data["deviation_used"],
        )

    # Run genetic algorithm fit
    if args.genetic_algorithm:
        print("Running genetic algorithm fit...")
<<<<<<< HEAD
        m_ = 6
=======
        m_ = 7
>>>>>>> refs/remotes/origin/main

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
<<<<<<< HEAD
            population_size=500,
=======
            population_size=20,
>>>>>>> refs/remotes/origin/main
            num_genes=2 * m_ + 2 * m_ * (2 * m_ + 1) + 4,
            gene_range=(
                [(-0.03, 0.03)] * (m_ * (2 * m_ + 1))
                + [(-0.03, 0.03)] * (m_ * (2 * m_ + 1))
                + [(-0.03, 0.03)] * m_
                + [(-0.03, 0.03)] * m_
                + [(0.98, 1.02), (-0.00001, 0.00001), (1.19, 1.23), (0.66, 0.70)]
            ),
            name=name,
<<<<<<< HEAD
            num_generations=40000,
=======
            num_generations=10000,
>>>>>>> refs/remotes/origin/main
            elitism=2,
            mutation_rate=0.01,
            mutation_range=np.concatenate(
                (
                    np.full(m_ * (2 * m_ + 1), 0.01),
                    np.full(m_ * (2 * m_ + 1), 0.01),
                    np.full(m_, 0.01),
                    np.full(m_, 0.01),
                    np.array([0.02, 0.000001, 0.02, 0.02]),
                )
            ),
            limit_fitness=0.0001,
        )

        # ID1916
        # 001
        # 6.841151266447373924e-01  8.657485849059558258e-03
        # 1.212749542688398074e+00  6.576662857131920607e-03
        # 003
        # 6.841151266447373924e-01  8.565109695075220908e-03
        # 1.181653400568182910e+00  6.437525266968448581e-03
        # 007
        # 6.841151266447373924e-01  7.927626203328231105e-03
        # 1.212749542688398074e+00  6.042995701903110944e-03

        # ID1917 ---------------------------------------------------------
        # 001
        # 7.504043773787922866e-01  1.026350082089539512e-02
        # 1.112668559561657666e+00  5.339715369964900815e-03
        # 003
        # 7.504043773787922866e-01  1.067042796924237702e-02
        # 1.112668559561657666e+00  5.296099721152459222e-03
        # 007
        # 7.504043773787922866e-01  9.486225921637897116e-03
        # 1.112668559561657666e+00  5.543819848102723644e-03

        # ID1918 ----------------------------------------------------------
        # 001
        # 5.812703081591484855e-01  7.293740081353852420e-03    1.720370000
        # 1.259419001011488515e+00  8.964707699941268718e-03    0.794016923
        # 003
        # 007
        # ID1919 ----------------------------------------------------------
        # 001
        # 1.346956520690677950e+00  3.704314186094919206e-03    0.742414462
        # 2.020434781036017036e+00  2.677347356997863189e-03    0.494942974
        # 2.745719061407920325e+00  3.487168691736520620e-03    0.364203321
        # 003
        # 007
