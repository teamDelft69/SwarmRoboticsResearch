import json
import os
from typing import List

"""
Contains controller merhods used by the visualization server. (Used to fetch data used in the visualization.)
"""

SAVED_SIMULATIONS_PATH = 'output/saved_simulations/'
SIMULATION_RESULTS_PATH = 'output/simulation_results/'


def read_json_file_of_path_and_name(path: str, name: str) -> dict:
  """
  Reads the json file located at the provided path, with the provided name and returns its contents as a dictionary.
  """
  name = name + '.json'
  full_path = os.path.join(path, name)
  return json.load(open(full_path, 'r'))

def get_all_json_file_names(path: str) -> List[str]:
  """
  Returns the names of all json files at the given path
  """

  contents = os.listdir(path)
  filtered_contents = list(filter(lambda x: x.endswith('.json'), contents))
  names = list(map(lambda x: x.split('.')[0], filtered_contents))

  return names


def get_saved_simulation(name: str) -> dict:
  """
  Return the saved simulation of the given name
  """
  return read_json_file_of_path_and_name(SAVED_SIMULATIONS_PATH, name)

def get_simulation_result(name: str) -> dict:
  """
  Return the simulation result of the given name
  """
  return read_json_file_of_path_and_name(SIMULATION_RESULTS_PATH, name)

def get_all_saved_simulations() -> List[str]:
  """
  Returns a list with the names of all saved simulations
  """
  return get_all_json_file_names(SAVED_SIMULATIONS_PATH)

def get_all_simulation_results() -> List[str]:
  """
  Returns a list with the names of all simulation results.
  """

  return get_all_json_file_names(SIMULATION_RESULTS_PATH)
