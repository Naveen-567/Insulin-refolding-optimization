#@naveenj

import pandas as pd
import Comp1
from tools import tool1, tool2
from optimization import optimization_algorithm

column_names1 = ['pH', 'Temp', 'DF', '1hr', '2hr', 'Overall']
column_names2 = ['pH', 'Temp', '3hr', '5hr','6hr', 'Overall']

a1 = Comp1.pop
b1 = Comp1.fitness_values

tool1([a1, b1], column_names1, 'Comp1.csv')
tool2('Comp1.csv', 'Opt1.csv', 'Overall')

df_o = pd.read_csv('Opt1.csv')
DF_o = df_o['DF']
#Conc_o = df_o['Conc']
Conc_o = 65
n_var = 2
lb = [10.7, 6]
ub = [11, 7]
pop_size = 100
rate_crossover = 20
rate_mutation = 20
rate_local_search = 10
step_size = 0.1
maximum_generation = 150

pop, fitness_values = optimization_algorithm(n_var, lb, ub, pop_size, rate_crossover, rate_mutation, rate_local_search, step_size, maximum_generation, DF_o, Conc_o)

a2 = pop
b2 = fitness_values

tool1([a2, b2], column_names2, 'Comp2.csv')
tool2('Comp2.csv', 'Opt2.csv', 'Overall')