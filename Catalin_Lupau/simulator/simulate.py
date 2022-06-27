import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

from NpEncoder import NpEncoder
from simulation_runner import (plot_experiment_results, run_experiment,
                               run_simulation, save_experiment_results,
                               save_results, save_simulation_envs)

SAVED_SIMULATION_PATH = 'output/saved_simulations/'
SIMULATION_RESULT_PATH = 'output/simulation_results/'
PLOT_PATH = 'output/plots/'

## sort out the arguments.
current_timestamp = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")


parser = argparse.ArgumentParser(description='Run simulation')
parser.add_argument('config_file', metavar='c', type=str, help='The name of the configuration file.')
parser.add_argument('--sim_name', metavar='n', default=current_timestamp, type=str, help='The name of the simulation instance.')
parser.add_argument('--save', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--experiment', default=False,
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--plot', default=False,
                    action=argparse.BooleanOptionalAction)

parsed_args = parser.parse_args(sys.argv[1:])


if parsed_args.experiment:
  # run an entire experiment
  experiment_output = run_experiment(parsed_args.sim_name, parsed_args.config_file)
  save_experiment_results(parsed_args.sim_name, experiment_output)

  if parsed_args.plot:
    plot_experiment_results(parsed_args.sim_name, experiment_output)

else:

  # run a single simulation
  simulation_output, saved_environments = run_simulation(
      parsed_args.sim_name, parsed_args.config_file)

  # save the simulation, if it is the case
  if parsed_args.save:
    save_simulation_envs(parsed_args.sim_name, saved_environments)

  save_results(parsed_args.sim_name, simulation_output)
    





