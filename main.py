import argparse
import os
import sys
import pandas as pd
import time

from GeneticAlgorithm import GeneticAlgorithm

dir = os.path.dirname(__file__)

def get_instances(path, instance):
    instance_name, _ = os.path.splitext(os.path.basename(path))
    instance_name = instance_name + "_" + str(instance)
    found = False
    with open(path, 'r') as f:
        for line in f:
            if instance_name in line: #found instance
                found = True
                break
        if not(found):
            raise FileNotFoundError("Instance not found in the file")

        n = int(f.readline().split(" ")[1]) # n_instances
        c = int(f.readline().split(" ")[1]) # capacity
        z = int(f.readline().split(" ")[1]) # best solution value
        t = float(f.readline().split(" ")[1]) # execution time

        d = []
        for _ in range(n):
            line = f.readline().split(",")
            d.append((
                int(line[0]),
                int(line[1]),
                int(line[2])
            ))

        df = pd.DataFrame(d, columns=('ID', 'Value', 'Weight'))
        df.set_index('ID', inplace=True)

        return n,c,z,t,df

def population_size(n):
    return min((5*n), ((2**n)//2))

parser = argparse.ArgumentParser()
parser.add_argument("path", help="the path to the test file")
parser.add_argument("instance", type=int, choices=range(1,101), help="the instance in the file")
parser.add_argument("--max-iter", "-i", type=int, default=100, help="maximum number of iterations")
parser.add_argument("--invalid_policy", choices=['repair', 'punish', 'threshold', 'discard'], default='punish',
                    help="policy to be applied to invalid individuals")
parser.add_argument("--popsize", "-p", type=int, default=None, help="population size")
parser.add_argument("--mutation-rate", "-mr", type=float, default=0.01, help="mutation rate")
parser.add_argument("--crossover-rate", "-cr", type=float, default=0.8, help="crossover rate")
parser.add_argument("--tournament-size", "-ts", type=int, default=None, help="tournament size")
args = parser.parse_args()

args.path = os.path.join(dir, args.path)
#print(args.path, args.instance)

n,c,z,t,df = get_instances(args.path, args.instance)

if not(args.popsize):
    args.popsize = population_size(n)
    print(args.popsize)

if not(args.tournament_size):
    args.tournament_size = args.popsize//10

start = time.time()

timestr = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(os.path.join(dir, "results/{}".format(timestr)))

with open(os.path.join(dir, "results/{}/log.txt".format(timestr)), "w") as log_file:
    GA = GeneticAlgorithm(population_size=args.popsize,
                          mutation_rate=args.mutation_rate,
                          crossover_rate=args.crossover_rate,
                          tournament_size=args.tournament_size,
                          items=df,
                          capacity=c,
                          max_iter=args.max_iter)

    #print("0", None, ';'.join(map(str, GA.fitnesses())), sep=';')
    #log_file.write("0,None," + ';'.join(map(str, GA.fitnesses())) + '\n')
    #log_file.flush()


    while not(GA.terminate_condition()):
        GA.generate_next_population()
        #print(GA.generation, GA.best_individual.genotype, ';'.join(map(str,GA.fitnesses())), sep=',')
        print(GA.generation, GA.best_individual.fitness)
        #log_file.write(str(GA.generation) + ";" + str(GA.best_individual.genotype) + ";" +
        #               ';'.join(map(str, GA.fitnesses())) + '\n')
        log_file.write(str(GA.generation) + "," + str(GA.best_individual.fitness))
        log_file.flush()

end = time.time()
print(end - start) # in seconds

