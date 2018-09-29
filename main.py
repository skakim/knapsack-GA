import argparse
import os
import sys
import pandas as pd
import time

from GeneticAlgorithm import GeneticAlgorithm

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

parser = argparse.ArgumentParser()
parser.add_argument("path", help="the path to the test file")
parser.add_argument("instance", type=int, choices=range(1,101), help="the instance in the file")
parser.add_argument("--maxiter", type=int, default=100, help="maximum number of iterations")
parser.add_argument("--invalid_policy", choices=['repair', 'punish', 'threshold', 'discard'], default='punish',
                    help="policy to be applied to invalid individuals")
parser.add_argument("--popsize", "-p", type=int, default=100, help="population size")
parser.add_argument("--mutation-rate", "-mr", type=float, default=0.01, help="mutation rate")
parser.add_argument("--crossover-rate", "-cr", type=float, default=0.8, help="crossover rate")
args = parser.parse_args()
#print(args.path, args.instance)

n,c,z,t,df = get_instances(args.path, args.instance)

print(len(df))

start = time.time()
GA = GeneticAlgorithm(population_size=args.popsize,
                      mutation_rate=args.mutation_rate,
                      crossover_rate=args.crossover_rate,
                      items = df,
                      capacity = c)
print(GA.population)
end = time.time()
print(end - start) # in seconds

