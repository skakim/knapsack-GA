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
parser.add_argument("--max-iter", "-i", type=int, default=50, help="maximum number of iterations")
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
print("Popsize:", args.popsize)

if not(args.tournament_size):
    args.tournament_size = args.popsize//10
print("Tournamentsize:", args.tournament_size)

for tolerance_rate in reversed([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]):
    print("Tolerance: {} ({})".format(tolerance_rate, c*tolerance_rate))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(dir, "results/tol{}-{}".format(tolerance_rate,timestr)), exist_ok=True)

    start = time.time()
    with open(os.path.join(dir, "results/tol{}-{}/log.txt".format(tolerance_rate,timestr)), "w") as log_file:
        GA = GeneticAlgorithm(population_size=args.popsize,
                              mutation_rate=args.mutation_rate,
                              crossover_rate=args.crossover_rate,
                              tournament_size=args.tournament_size,
                              items=df,
                              capacity=c,
                              capacity_tolerance=c*tolerance_rate,
                              max_iter=args.max_iter)

        #print("0", None, ';'.join(map(str, GA.fitnesses())), sep=';')
        #log_file.write("0,None," + ';'.join(map(str, GA.fitnesses())) + '\n')
        #log_file.flush()
        log_file.write("generation,value,overweight\n")
        log_file.flush()

        while not(GA.terminate_condition()):
            GA.generate_next_population()
            #print(GA.generation, GA.best_individual.genotype, ';'.join(map(str,GA.fitnesses())), sep=',')
            print(GA.generation, GA.best_individual.value, GA.best_individual.overweight)
            #log_file.write(str(GA.generation) + ";" + str(GA.best_individual.genotype) + ";" +
            #               ';'.join(map(str, GA.fitnesses())) + '\n')
            log_file.write(str(GA.generation) + "," + str(GA.best_individual.value) + "," + str(GA.best_individual.overweight)+"\n")
            log_file.flush()

    end = time.time()
    print(end - start) # in seconds

