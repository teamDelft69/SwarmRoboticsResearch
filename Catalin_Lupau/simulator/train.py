import argparse
import os
import pickle
import sys
from copyreg import pickle

import numpy as np
import pygad.torchga
from yaml import parse

from paths import *
from simulation_runner import run_experiment
from swarm_agents.neural_network_boids import BoidNet, NeuralNetworkBoids

parser = argparse.ArgumentParser(description='Run simulation')
parser.add_argument('config_file', metavar='c', type=str,
                    help='The name of the configuration file.')
parser.add_argument('boid_name', metavar='n',
                    type=str, help='The name of the trained boid.')
parser.add_argument('population_size', metavar='p', type=int, help='The initial population size of the experiment')
parser.add_argument('num_parents_mating', metavar='m', type=int, help='The number of parents that are mating per generation')
parser.add_argument('num_generations', metavar='g', type=int,
                    help='The number of generations to run the genetic algorithm for.')




parsed_args = parser.parse_args(sys.argv[1:])

FITNESS_EXPERIMENT_NAME = parsed_args.config_file
BOID_NAME = parsed_args.boid_name
INITIAL_POPULATION_SIZE = parsed_args.population_size
NUM_GENERATIONS = parsed_args.num_generations
NUM_PARENTS_MATING = parsed_args.num_parents_mating



def evaluate_neural_network(boid_net: BoidNet):
  """
  Computes the fitness function of the boid network.
  """

  experiment_results = run_experiment(BOID_NAME, FITNESS_EXPERIMENT_NAME, boid_net)

  return experiment_results['avg_agent_success'][0]


def fitness_func(solution, solution_idx):
  
  boid_net = BoidNet(solution)
  return evaluate_neural_network(boid_net)


def callback_generation(ga_instance):
    print("\n")
    print("++++++++++++++++++++++++++++++++++++++++++")
    print("Generation Summary")
    print("++++++++++++++++++++++++++++++++++++++++++")
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))
    
    gen_num = ga_instance.generations_completed
    nn_name = f'{BOID_NAME}_gen_{gen_num}'

    print(ga_instance.best_solution())
    

    print("++++++++++++++++++++++++++++++++++++++++++")
    print("\n")


def on_stop(ga_instance, last_population_fitness):
  solution = ga_instance.best_solution()[0]
  print(solution)
  np.save(open(os.path.join(NEURAL_NETWORKS_PATH, BOID_NAME) + '.pkl', 'wb'), solution)
  


init_net = BoidNet.init_with_random_weights()

torch_ga = pygad.torchga.TorchGA(
    model=init_net.get_model(), num_solutions=INITIAL_POPULATION_SIZE)

initial_population = torch_ga.population_weights
ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                       num_parents_mating=NUM_PARENTS_MATING,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation,
                       on_stop=on_stop
                       )


ga_instance.run()
# ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness")
