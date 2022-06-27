"""
Contains a function used to run a simulation based on simulation configs.
"""

import argparse
import json
import os
import sys
from curses import resetty
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from NpEncoder import NpEncoder
from paths import *
from simulator.Simulator import Simulator
from simulator.SimulatorConfiguration import SimulatorConfiguration
from swarm_agents.neural_network_boids import BoidNet


def remove_extension(filename: str) -> str:
  """
  Removes the '.' extension from a file name
  """

  if '.' not in filename:
    return filename
  
  return filename.split('.')[0].strip()


def get_current_timestamp() -> str:
  """
  Return the current timestamp. 
  """
  return datetime.now().strftime("%m_%d_%Y-%H_%M_%S")


def save(name: str, json_data: dict, path: str):
  """
  SAves the provided json_data at the specified path.
  """
  name = name.replace('/', '_')
  with open(os.path.join(path, f'{name}.json'), 'w') as f:
    json.dump(json_data, f, cls=NpEncoder, indent=4)

def save_simulation_envs(name: str, simulation_envs: dict):
  """
  Saved the provided simulations envs object with the provided name.
  """
  save(name, simulation_envs, SAVED_SIMULATION_PATH)

  # with open(os.path.join(SAVED_SIMULATION_PATH, f'{name}.json'), 'w') as f:
  #   json.dump(simulation_envs, f, cls=NpEncoder, indent=4)



def save_results(name: str, results: dict):
  """
  Saves the results of a simulation
  """
  save(name, results, SIMULATION_RESULT_PATH)
  # with open(os.path.join(SIMULATION_RESULT_PATH, f'{name}.json'), 'w') as f:
  #   json.dump(results, f, cls=NpEncoder, indent=4)


def save_experiment_results(name: str, results):
  """
  Saves the results of an experiment.
  """
  save(name, results, EXPERIMENT_RESULTS_PATH)


def remove_base_path(name: str) -> str:
  """
  Removes the base path from the provided name
  """
  return name.split('/')[-1]

def remove_base_path_from_list(names: list) -> list:
  """
  Removes the base path from a list of names.
  """
  return list(map(remove_base_path, names))





def plot_experiment_results(name: str, results: dict, plot_impact_time=False):
  """
  Creates a histogram of the results of the experiment.
  """
  plt.rc('xtick', labelsize=8)
  plt.rcParams['figure.figsize'] = [10, 8]
  avg_success = np.array(results['avg_agent_success'])
  avg_time = np.array(results['avg_time'])
  avg_delta_time = np.array(results['avg_delta_time'])

  delta_text = 0.05

  fig, ax = plt.subplots(3 if plot_impact_time else 2, 1)
  fig.tight_layout(h_pad=2)
  ax[0].bar(remove_base_path_from_list(results['names']),
            avg_success, fc='#7bc5f7', ec='#1f3e54')
  ax[0].set_title('Success Rate')

  for i in range(len(avg_success)):
    ax[0].text(i - delta_text, avg_success[i] / 2, str(round(avg_success[i], 2)))
  # ax[0].text(1 - delta_text, avg_success[1] / 2, str(round(avg_success[1], 2)))

  ax[1].bar(remove_base_path_from_list(results['names']),
            avg_time, fc='#f0a0f7', ec='#82178c')
  ax[1].set_title('Average Time')
  for i in range(len(avg_success)):
    ax[1].text(i - delta_text, avg_time[i] / 2, str(round(avg_time[i], 2)))

  if plot_impact_time:
    ax[2].bar(remove_base_path_from_list(results['names']),
              avg_delta_time, fc='#f97a7c', ec='#ea0003')
    ax[2].set_title('Average Impact Time')
    ax[2].text(0 - delta_text, avg_delta_time[0] /
              2, str(round(avg_delta_time[0], 2)))
    ax[2].text(1 - delta_text, avg_delta_time[1] /
              2, str(round(avg_delta_time[1], 2)))

  fig.suptitle(
      f'Experiment "{name}", results averaged over {results["num_repetitions"]} trials')


  plt.subplots_adjust(top=0.85)

  # fig.canvas.set_window_title(
  #     f'Experiment {name}, results averaged over {results["num_repetitions"]}')

  plt.savefig(os.path.join(PLOT_PATH, f'{name}.png'))

def run_experiment(experiment_name: str, experiment_config_file: str, boid_net: BoidNet = None):
  """
  Runs an experiment involving averaging multiple simulations
  """

  experiment_config = json.load(open(os.path.join(EXPERIMENT_CONFIG_PATH, experiment_config_file), 'r'))

  final_result = {
    'name': experiment_name,
    'timestamp': get_current_timestamp(),
    'config_files': experiment_config['simulation_config_files'],
    'names': [remove_extension(file) for file in experiment_config['simulation_config_files']],
    'num_repetitions': experiment_config['num_repetitions'],
    'avg_agent_success': [],
    'avg_time': [],
    'avg_delta_time': []
  }

  num_configuration_files = len(experiment_config['simulation_config_files'])

  # initialize the histograms
  avg_agent_success_histogram = np.zeros(num_configuration_files)
  avg_time_histogram = np.zeros(num_configuration_files)
  avg_delta_time = np.zeros(num_configuration_files)

  # get the number of repetitions
  num_repetitions = experiment_config['num_repetitions']
  seeds = np.random.choice(list(range(1000)), size=num_repetitions)


  for (sim_config_index, simulation_config), _ in zip(enumerate(experiment_config['simulation_config_files']), tqdm(range(0, 100), total=num_configuration_files, desc="Overall progress")):
    name = remove_extension(simulation_config)

    

    # add the results from the simulations
    for index, _ in zip(range(1,  num_repetitions + 1), tqdm(range(0, num_repetitions), total=num_repetitions, desc='Simulation Batch Progress')):
      simulation_output, simulation_envs = run_simulation(
          experiment_name + "_" + name + '_' + str(index), simulation_config, boid_net, seeds[index - 1])

      if index in experiment_config['saved_repetitions']:
        save_simulation_envs(experiment_name + "_" + name +
                             '_' + str(index), simulation_envs)
      

      # update the histograms
      avg_agent_success_histogram[sim_config_index] += (float(
          simulation_output['result']['num_agents_that_reached_targets']) / float(simulation_output['result']['num_agents']))
      avg_time_histogram[sim_config_index] += float(simulation_output['result']['total_time'])
      avg_delta_time[sim_config_index] += float(
          simulation_output['result']['delta_time_target'])

    # compute the average results
    avg_agent_success_histogram[sim_config_index] /= num_repetitions
    avg_time_histogram[sim_config_index] /= num_repetitions
    avg_delta_time[sim_config_index] /= num_repetitions
  

  # update the final results with the value of the histograms
  final_result['avg_agent_success'] = list(avg_agent_success_histogram)
  final_result['avg_time'] = list(avg_time_histogram)
  final_result['avg_delta_time'] = list(avg_delta_time)

  return final_result

    


def run_simulation(sim_name: str, config_file_name_json: str, boid_net: BoidNet = None, seed = None):
  """
  Function used to run one instance of a simulation.

  sim_name - the name of the simulation
  config_file_name - the name of the configuration file (with the .json extension included)
  """

  current_timestamp = get_current_timestamp()

  config = SimulatorConfiguration.build_configuration_from_file(config_file_name_json, boid_net)
  config.seed = seed

  simulator = Simulator(config)

  simulation_output = {
      'name': sim_name,
      'timestamp': current_timestamp,
      'associated_config_file': config_file_name_json,
      'result': {}
  }

  saved_environments = {
      'name': sim_name,
      'timestamp': current_timestamp,
      'associated_config_file': config_file_name_json,
      'environments': []
  }

  for env in simulator.run():
    saved_environments['environments'].append(env)


  # save the simulation result
  simulation_output['result'] = simulator.get_last_result()

  return simulation_output, saved_environments
