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
    # For use follow these instructions:

    name = "ID1918_003"  # Set the name of your data file

    data = load_data(
        name,
        column_names=(
            "julian_day",
            "noiseless_flux",
            "noisy_flux",
            "sigma",
            "deviation_used",
        ),
        appendix=".flux",  # Set the appendix of your data file
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
            n_iter=100,
            n_b=4,
            gain=0.8,
            final_noise=0.00006,
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
            population_size=250,
            num_genes=2 * m_ + 2 * m_ * (2 * m_ + 1) + 4,
            gene_range=(
                [(-0.04, 0.04)] * (m_ * (2 * m_ + 1))
                + [(-0.04, 0.04)] * (m_ * (2 * m_ + 1))
                + [(-0.04, 0.04)] * m_
                + [(-0.04, 0.04)] * m_
                + [(0.98, 1.02), (-0.00001, 0.00001), (0.27, 0.31), (0.32, 0.36)]
            ),
            name=name,
            num_generations=20000,
            elitism=2,
            mutation_rate=0.01,
            mutation_range=np.concatenate(
                (
                    np.full(m_ * (2 * m_ + 1), 0.04),
                    np.full(m_ * (2 * m_ + 1), 0.04),
                    np.full(m_, 0.04),
                    np.full(m_, 0.04),
                    np.array([0.02, 0.000001, 0.02, 0.02]),
                )
            ),
            limit_fitness=0.001,
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
        # 5.812703081591484855e-01  8.290761219419122419e-03
        # 1.259419001011488515e+00  1.130426637465926226e-02
        # 003
        # 5.812703081591484855e-01  8.354706241670319111e-03
        # 1.259419001011488515e+00  1.120579038334303980e-02
        # 007
        # 5.812703081591484855e-01  7.889197263304212338e-03
        # 1.259419001011488515e+00  1.132145483725020482e-02

        # ID1919 ----------------------------------------------------------
        # 001
        # 1.346956520690677950e+00  3.807578077374085702e-03
        # 2.020434781036017036e+00  2.673548196521290977e-03
        # 2.745719061407920325e+00  3.463454304762446107e-03
        # 003
        # 1.346956520690677950e+00  3.875448589745145601e-03
        # 1.968628761009452388e+00  2.516373685169524550e-03
        # 2.745719061407920325e+00  3.483928700386224996e-03
        # 007
        # 1.346956520690677950e+00 3.331948212817425516e-03
        # 2.020434781036017036e+00 2.675007881692549345e-03
        # 2.745719061407920325e+00 3.196768958069567262e-03
