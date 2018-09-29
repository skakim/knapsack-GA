import pandas as pd
import numpy as np

class GeneticAlgorithm(object):
    def __init__(self, population_size: int, mutation_rate: float, crossover_rate: float,
                 items: pd.DataFrame, capacity: int):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.items = items
        self.capacity = capacity

        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        pop = []
        for _ in range(self.population_size):
            #genotype = np.random.randint(2, size=len(self.items))
            genotype = [1, 1, 1] + [0] * 17
            ind = Individual(genotype, self.items, self.capacity)
            pop.append(ind)
        return pop

class Individual(object):
    def __init__(self, genotype: list, items: pd.DataFrame, capacity: int):
        self.genotype = genotype
        self.fitness = sum([items.loc[i+1]["Value"] if self.genotype[i] else 0.0 for i in range(len(genotype))])