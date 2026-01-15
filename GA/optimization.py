#@naveenj

import numpy as np
from Comp2 import evaluation, random_population, crossover, mutation, local_search, selection, pareto_front_finding

def optimization_algorithm(n_var, lb, ub, pop_size, rate_crossover, rate_mutation, rate_local_search, step_size, maximum_generation, DF_o, Conc_o):

    pop = random_population(n_var, pop_size, lb, ub)
    print(pop.shape)

    for i in range(maximum_generation):
        offspring_from_crossover = crossover(pop, rate_crossover)
        offspring_from_mutation = mutation(pop, rate_mutation)
        offspring_from_local_search = local_search(pop, rate_local_search, step_size,lb, ub)

        pop = np.append(pop, offspring_from_crossover, axis=0)
        pop = np.append(pop, offspring_from_mutation, axis=0)
        pop = np.append(pop, offspring_from_local_search, axis=0)

        fitness_values = evaluation(pop, DF_o, Conc_o)
        pop = selection(pop, fitness_values, pop_size)
        print('iteration:', i)

    fitness_values = evaluation(pop, DF_o, Conc_o)
    index = np.arange(pop.shape[0]).astype(int)
    pareto_front_index = pareto_front_finding(fitness_values, index)
    pop = pop[pareto_front_index, :]
    return pop, fitness_values