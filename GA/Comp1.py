#@naveenj

import random as rn
import numpy as np

#GA
def random_population(n_var, n_sol, lb, ub):
    pop = np.zeros((n_sol, n_var))
    for i in range(n_sol):
        pop[i,:] = np.random.uniform(lb, ub)
    return pop

def crossover(pop, crossover_rate):
    offspring = np.zeros((crossover_rate, pop.shape[1]))
    for i in range(int(crossover_rate/2)):
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == r2:
            r1 = np.random.randint(0, pop.shape[0])
            r2 = np.random.randint(0, pop.shape[0])
        cutting_point = np.random.randint(1, pop.shape[1])
        offspring[2*i, 0:cutting_point] = pop[r1, 0:cutting_point]
        offspring[2*i, cutting_point:] = pop[r2, cutting_point:]
        offspring[2*i+1, 0:cutting_point] = pop[r2, 0:cutting_point]
        offspring[2*i+1, cutting_point:] = pop[r1, cutting_point:]

    return offspring

def mutation(pop, mutation_rate):
    offspring = np.zeros((mutation_rate, pop.shape[1]))
    for i in range(int(mutation_rate/2)):
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == r2:
            r1 = np.random.randint(0, pop.shape[0])
            r2 = np.random.randint(0, pop.shape[0])
        cutting_point = np.random.randint(0, pop.shape[1])
        offspring[2*i] = pop[r1]
        offspring[2*i, cutting_point] = pop[r2, cutting_point]
        offspring[2*i+1] = pop[r2]
        offspring[2*i+1, cutting_point] = pop[r1, cutting_point]

    return offspring

def local_search(pop, n_sol, step_size):
    offspring = np.zeros((n_sol, pop.shape[1]))
    for i in range(n_sol):
        r1 = np.random.randint(0, pop.shape[0])
        chromosome = pop[r1, :]
        r2 = np.random.randint(0, pop.shape[1])
        chromosome[r2] += np.random.uniform(-step_size, step_size)
        if chromosome[r2] < lb[r2]:
            chromosome[r2] = lb[r2]
        if chromosome[r2] > ub[r2]:
            chromosome[r2] = ub[r2]

        offspring[i,:] = chromosome
    return offspring

def evaluation(pop):
    fitness_values = np.zeros((pop.shape[0], 3))
    for i in range(pop.shape[0]):
        pH = pop[i,0]
        Conc = 40
        Temp = pop[i,1]
        DF = pop[i,2]
        obj1 = 9.914 + (4*((pH-9.5)/1.5)) + (1.833*((Conc-50)/25)) + ((-0.611)*((Temp-14)/10)) + ((-0.722)*((DF-20)/10)) + (((pH-9.5)/1.5)*((Conc-50)/25)*2.3125) + (((pH-9.5)/1.5)*((Temp-14)/10)*0.3125) + (((Conc-50)/25)*((Temp-14)/10)*0.1875) + (((pH-9.5)/1.5)*((DF-20)/10)*(-0.3125)) + (((Conc-50)/25)*((DF-20)/10)*0.8125) + (((Temp-14)/10)*((DF-20)/10)*0.3125) + (((pH-9.5)/1.5)*((pH-9.5)/1.5)*1.7809) + (((Conc-50)/25)*((Conc-50)/25)*(-0.7190)) + (((Temp-14)/10)*((Temp-14)/10)*(-1.719)) + (((DF-20)/10)*((DF-20)/10)*(-1.719)) #1hr
        obj2 = 35.378 + (1.86*((pH-9.5)/1.5)) + (0.638*((Conc-50)/25)) + (-(1.194)*((Temp-14)/10)) + ((-0.361)*((DF-20)/10)) + (((pH-9.5)/1.5)*((Conc-50)/25)*(1.156)) + (((pH-9.5)/1.5)*((Temp-14)/10)*0.09375) + (((Conc-50)/25)*((Temp-14)/10)*(-0.156)) + (((pH-9.5)/1.5)*((DF-20)/10)*1.468) + (((Conc-50)/25)*((DF-20)/10)*(-0.781)) + (((Temp-14)/10)*((DF-20)/10)*0.531) + (((pH-9.5)/1.5)*((pH-9.5)/1.5)*(2.329)) + (((Conc-50)/25)*((Conc-50)/25)*(1.329)) + (((Temp-14)/10)*((Temp-14)/10)*(-1.671)) + (((DF-20)/10)*((DF-20)/10)*(1.828)) #2hrs
        obj3 = 59.12 + (20.033*((pH-9.5)/1.5)) + (1.311*((Conc-50)/25))+ ((-1.367)*((Temp-9.5)/1.5)) + ((-0.133)*((DF-20)/10)) + (((pH-9.5)/1.5)*((Conc-50)/25)*(0.35))+ (((pH-9.5)/1.5)*((Temp-14)/10)*(0.275))+ (((Conc-50)/25)*((Temp-14)/10)*(0.9))+ (((pH-9.5)/1.5)*((DF-20)/10)*(0.15))+ (((Conc-50)/25)*((DF-20)/10)*(0.525))+ (((Temp-14)/10)*((DF-20)/10)*(1.4))+ (((pH-9.5)/1.5)*((pH-9.5)/1.5)*(-18.993))+ (((Conc-50)/25)*((Conc-50)/25)*(-2.493))+ (((Temp-14)/10)*((Temp-14)/10)*(0.007))+ (((DF-20)/10)*((DF-20)/10)*(-0.493))
        fitness_values[i,0] = -obj1
        fitness_values[i,1] = -obj2
        fitness_values[i,2] = -obj3
    return fitness_values

def crowding_calculation(fitness_values):
    pop_size = len(fitness_values[:, 0])
    fitness_value_number = len(fitness_values[0, :])
    matrix_for_crowding = np.zeros((pop_size, fitness_value_number))
    normalized_fitness_values = (fitness_values - fitness_values.min(0))/fitness_values.ptp(0)

    for i in range(fitness_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1
        crowding_results[pop_size - 1] = 1
        sorted_normalized_fitness_values = np.sort(normalized_fitness_values[:,i])
        sorted_normalized_values_index = np.argsort(normalized_fitness_values[:,i])
        crowding_results[1:pop_size - 1] = (sorted_normalized_fitness_values[2:pop_size] - sorted_normalized_fitness_values[0:pop_size - 2])
        re_sorting = np.argsort(sorted_normalized_values_index)
        matrix_for_crowding[:, i] = crowding_results[re_sorting]

    crowding_distance = np.sum(matrix_for_crowding, axis=1)
    return crowding_distance

def remove_using_crowding(fitness_values, number_solutions_needed):
    pop_index = np.arange(fitness_values.shape[0])
    crowding_distance = crowding_calculation(fitness_values)
    selected_pop_index = np.zeros(number_solutions_needed)
    selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))
    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, (solution_1), axis=0)
            fitness_values = np.delete(fitness_values, (solution_1), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_1), axis=0)
        else:
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, (solution_2), axis=0)
            fitness_values = np.delete(fitness_values, (solution_2), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_2), axis=0)

    selected_pop_index = np.asarray(selected_pop_index, dtype=int)

    return selected_pop_index

def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)
    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0
                break

    return pop_index[pareto_front]

def selection(pop, fitness_values, pop_size):

    pop_index_0 = np.arange(pop.shape[0])
    pop_index = np.arange(pop.shape[0])
    pareto_front_index = []

    while len(pareto_front_index) < pop_size:
        new_pareto_front = pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
        total_pareto_size = len(pareto_front_index) + len(new_pareto_front)

        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(pareto_front_index)
            selected_solutions = remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed)
            new_pareto_front = new_pareto_front[selected_solutions]

        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front))
        remaining_index = set(pop_index) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))

    selected_pop = pop[pareto_front_index.astype(int)]

    return selected_pop

# Parameters
n_var = 3
lb = [8, 2, 10]
ub = [11, 25, 30]
pop_size = 100
rate_crossover = 20
rate_mutation = 20
rate_local_search = 10
step_size = 0.1
maximum_generation = 150
pop = random_population(n_var, pop_size, lb, ub)
print(pop.shape)


for i in range(maximum_generation):
    offspring_from_crossover = crossover(pop, rate_crossover)
    offspring_from_mutation = mutation(pop, rate_mutation)
    offspring_from_local_search = local_search(pop, rate_local_search, step_size)

    pop = np.append(pop, offspring_from_crossover, axis=0)
    pop = np.append(pop, offspring_from_mutation, axis=0)
    pop = np.append(pop, offspring_from_local_search, axis=0)

    fitness_values = evaluation(pop)
    pop = selection(pop, fitness_values, pop_size)
    print('iteration:', i)

fitness_values = evaluation(pop)
index = np.arange(pop.shape[0]).astype(int)
pareto_front_index = pareto_front_finding(fitness_values, index)
pop = pop[pareto_front_index, :]
print("_________________")
print("Optimal solutions:")
print("       x1               x2                 x3                 x4")
print(pop) # show optimal solutions
fitness_values = fitness_values[pareto_front_index]
print("______________")
print("Fitness values:")
print("  objective 1    objective 2    objective 3")
print(fitness_values)