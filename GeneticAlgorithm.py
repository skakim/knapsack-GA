import pandas as pd
import numpy as np
from copy import copy, deepcopy
from operator import attrgetter
from math import sqrt


class GeneticAlgorithm(object):
    def __init__(self, population_size: int, mutation_rate: float, crossover_rate: float, tournament_size: int,
                 items: pd.DataFrame, capacity: int, capacity_tolerance: float, max_iter: int):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        self.items = items
        self.capacity = capacity
        self.capacity_tolerance = capacity_tolerance
        self.max_iter = max_iter

        self.cache = {}
        self.population = self.generate_initial_population()
        self.best_individual = None
        self.generation = 0

    def terminate_condition(self):
        return self.generation == self.max_iter

    def fitnesses(self):
        l = []
        for ind in self.population:
            l.append(ind.fitness)
        return sorted(l, reverse=True)

    def generate_initial_population(self):
        pop = []
        for _ in range(self.population_size):
            genotype = np.random.randint(2, size=len(self.items))
            ind = Individual(genotype, self.items, self.capacity, self.capacity_tolerance)
            pop.append(ind)
        return pop

    def generate_next_population(self):
        old_population = self.population
        new_population = []

        # test if best_individual needs to be updated
        best_from_population = max(old_population, key=attrgetter('fitness'))
        if not(self.best_individual) or \
            (best_from_population.fitness >= self.best_individual.fitness and
             not(np.array_equal(best_from_population.genotype, self.best_individual.genotype))):
            self.best_individual = copy(best_from_population)

        #print("Generation {} completed".format(self.generation))
        self.generation += 1
        #print("Creating generation {}".format(self.generation))

        # generate offspring
        # elitism
        new_population.append(self.best_individual)
        #print("Copied best individual to new generation", self.best_individual.genotype)

        # let the games begin!
        for i in range(self.population_size//2):
            # crossover
            ind1, ind2 = self.selTournament()
            gen1, gen2 = self.crossover(ind1, ind2)

            # mutation
            gen1 = self.mutation(gen1)
            gen2 = self.mutation(gen2)

            # retrieve from cache or create
            if tuple(gen1) in self.cache:
                new1 = self.cache[tuple(gen1)]
                #print("Cache used for individual", (i*2) + 1, new1.genotype)
            else:
                new1 = Individual(gen1, self.items, self.capacity, self.capacity_tolerance)
                #print("Finished creating individual", (i*2) + 1, new1.genotype)
                if tuple(new1.genotype) not in self.cache:
                    self.cache[tuple(new1.genotype)] = new1

            if tuple(gen2) in self.cache:
                new2 = self.cache[tuple(gen2)]
                #print("Cache used for individual", (i*2) + 2, new2.genotype)
            else:
                new2 = Individual(gen2, self.items, self.capacity, self.capacity_tolerance)
                #print("Finished creating individual", (i*2) + 2, new2.genotype)
                if tuple(new2.genotype) not in self.cache:
                    self.cache[tuple(new2.genotype)] = new2

            new_population.append(new1)
            new_population.append(new2)

        self.population = new_population

    def crossover(self, ind1, ind2):
        r = np.random.rand()
        if r > self.crossover_rate:
            return ind1.genotype, ind2.genotype

        gen1 = []
        gen2 = []

        # Uniform Crossover
        for i in range(len(ind1.genotype)):
            r = np.random.rand()
            if r < 0.5:
                gen1.append(ind1.genotype[i])
                gen2.append(ind2.genotype[i])
            else:
                gen1.append(ind2.genotype[i])
                gen2.append(ind1.genotype[i])

        return gen1, gen2

    def mutation(self, gen):
        for i in range(len(gen)):
            r = np.random.rand()
            if r < self.mutation_rate:
                # binary NOT
                gen[i] = 1 - gen[i]
        return gen

    def selRandom(self, individuals, k):
        """Select *k* individuals at random from the input *individuals* with
        replacement. The list returned contains references to the input
        *individuals*.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.

        This function uses the :func:`~random.choice` function from the
        python base :mod:`random` module.
        """
        return np.random.choice(individuals, size=k)

    def selTournament(self, k=2):
        """Select the best individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.

        :param self.population: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param self.tournament_size: The number of individuals participating in each tournament.
        :param self.fit_attr: The attribute of individuals to use as selection criterion (accuracy, AUC, F1 or MCC)
        :returns: A list of selected individuals.
        """
        population_copy = copy(self.population)
        chosen = []
        # choose first
        aspirants = self.selRandom(population_copy, self.tournament_size)
        chosen1 = max(aspirants, key=attrgetter('fitness'))
        chosen.append(chosen1)
        # choose second
        del population_copy[population_copy.index(chosen1)]
        aspirants = self.selRandom(population_copy, self.tournament_size)
        chosen2 = max(aspirants, key=attrgetter('fitness'))
        chosen.append(chosen2)
        return chosen

class Individual(object):
    def __init__(self, genotype: list, items: pd.DataFrame, capacity: int, capacity_tolerance: float):
        self.genotype = genotype
        self.fitness, self.value, self.overweight = self.calc_fitness(items, capacity, capacity_tolerance)
    
    def calc_fitness(self, items, capacity, capacity_tolerance):
        total_value = int(sum([items.loc[i+1]["Value"] if self.genotype[i] else 0.0 for i in range(len(self.genotype))]))
        total_weight = int(sum([items.loc[i+1]["Weight"] if self.genotype[i] else 0.0 for i in range(len(self.genotype))]))
        
        overweight = total_weight - capacity if total_weight > capacity else 0
        if overweight > capacity_tolerance:
            return total_value - 10*overweight - 999999, total_value, overweight
        else:
            return total_value - 10*overweight, total_value, overweight
