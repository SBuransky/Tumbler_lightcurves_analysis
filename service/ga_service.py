import time
from genetic_algorithm.run.one_run import run_genetic_algorithm
import matplotlib.pyplot as plt
from utils.fourier_series_value import double_fourier_sequence
from datetime import datetime as dt
import numpy as np


def tumbler_genetic_algorithm_fit(data,
                                  fitness_function,
                                  name,
                                  m_=1,
                                  population_size=500,
                                  gene_range=((-0.2, 0.2), (0.85, 1.15), (0.5, 1.5), (0.5, 1.5)),
                                  num_generations=100,
                                  elitism=2,
                                  crossover_rate=0.95,
                                  mutation_rate=0.01,
                                  mutation_range=0.5):
    final_generation = run_genetic_algorithm(
        population_size=population_size,
        fitness_function=fitness_function,
        num_genes=2 * m_ + 2 * m_ * (2 * m_ + 1) + 3,
        gene_range=(2 * m_ + 2 * m_ * (2 * m_ + 1)) * (gene_range[0],) + gene_range[1:],
        num_generations=num_generations,
        elitism=elitism,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_range=mutation_range
    )
    ts = dt.now()
    ending = str(ts.year)[2:] + '{:02d}'.format(ts.month) + '{:02d}'.format(ts.day) + '-' + '{:02d}'.format(
        ts.hour) + '{:02d}'.format(ts.minute) + '{:02d}'.format(ts.second)

    days = data['julian_day'].values
    plt.plot(days, double_fourier_sequence(final_generation[0], m_, days), label='last')
    plt.plot(days, double_fourier_sequence(final_generation[4], m_, days), label='best')

    plt.scatter(days, data['noisy_flux'].values, c='gray', marker='+', s=5)
    plt.errorbar(days, data['noisy_flux'].values, yerr=data['deviation_used'].values, fmt='none', color='black',
                 elinewidth=1.5, capsize=0)

    plt.xlabel('Time [days]')
    plt.ylabel('Normalized light flux')
    plt.legend()

    plt.savefig('Results/genetic_algorithm/graphs/' + name + '_graph_' + ending + '.pdf')
    plt.show()
    plt.close()

    # Plot the fitness over generations
    plt.plot(np.arange(len(final_generation[2])), final_generation[2])
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Over Generations')
    plt.savefig('Results/genetic_algorithm/fitness/' + name + 'fitness' + ending + '.pdf')
    plt.close()

    with open('Results/genetic_algorithm/results/' + name + '_result_' + ending + '.txt', 'w') as file:
        file.write('Best in last gen:')
        file.write(str(final_generation[0]))
        file.write('Best fitness in last gen:')
        file.write(str(final_generation[5]))

        file.write('\n')

        file.write('Best in all:')
        file.write(str(final_generation[4]))
        file.write('Best fitness in all:')
        file.write(str(final_generation[6]))
