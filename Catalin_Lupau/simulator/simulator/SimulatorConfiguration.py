import json
from typing import List

import numpy as np
from entities.Agent import Agent, AgentPerception
from entities.MapStructure import MapStructure
from entities.Obstacle import Obstacle
from entities.Projectile import Projectile
from entities.TargetArea import TargetArea
from generators.agent_generators import retrieve_agent_generator
from generators.obstacle_generators import retrieve_obstacle_generator
from generators.projectile_generators import retrieve_projectile_generator
from swarm_agents.neural_network_boids import BoidNet

from simulator.Environment import Environment


class SimulatorConfiguration:
  """
  Class encoding configuration data for the simulator.
  """

  def __init__(self, agent_generator, projectile_generator, obstacle_generator, target_area: TargetArea, map_structure: MapStructure, delta_time: float, max_ticks: int, agent_swarm_distance: float, projectile_swarm_distance: float, target_delta_time: float, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int, obstacle_min_size: float, obstacle_max_size: float, num_obstacles: int, projectile_size: float, projectile_acc_limit: float, projectile_perception_distance: float, num_projectiles: int, seed = None) -> None:
    """
    Initializes the simulator configuration

    agent_generator - function that generates the agents of the simulation.
    projectile_generator - function that generates the projectiles of the simulation, based on the current tick.
    obstacle_generator - function that generates the obstacles of the simulation
    target_area - the target area the swarm is trying to reach
    map_structure - the overall structure of the map
    delta_time - the time interval corresponding to moving from one tick to the other (seconds)
    max_ticks - the max number of ticks the simulation should run for
    swarm_distance - the minimum distance between the agents of the swarm that should be maintained
    target_delta_time - the time interval that is admitted between the first and the last entrance inside the target area
    """

    # function used to instatiate an agent of a given type
    self.target_area = target_area
    self.map_structure = map_structure
    self.delta_time = delta_time
    self.max_ticks = max_ticks
    self.agent_swarm_distance = agent_swarm_distance
    self.target_delta_time = target_delta_time

    self.agent_generator = agent_generator
    self.agent_size = agent_size
    self.agent_acc_limit = agent_acc_limit
    self.agent_perception_distance = agent_perception_distance
    self.num_agents = num_agents

    self.obstacle_generator = obstacle_generator
    self.obstacle_min_size = obstacle_min_size
    self.obstacle_max_size = obstacle_max_size
    self.num_obstacles = num_obstacles

    self.projectile_swarm_distance = projectile_swarm_distance
    self.projectile_acc_limit = projectile_acc_limit
    self.projectile_perception_distance = projectile_perception_distance
    self.projectile_generator = projectile_generator
    self.projectile_size = projectile_size
    self.num_projectiles = num_projectiles

    self.seed = seed


  @staticmethod
  def build_configuration_from_file(filename: str, boid_net: BoidNet = None):
    """
    Static methods that initializes a simulator configuration, based on the contants of the provided file
    """

    with open(f'simulator_configs/{filename}', 'r') as f:
      json_contents = '\n'.join(f.readlines())
      return SimulatorConfiguration.build_configuration_from_json_string(json_contents, boid_net)

  @staticmethod
  def build_configuration_from_json_string(configuration_json: str, boid_net: BoidNet = None):
    """
    Instantiated a simulator configuration object, based on the provided json string.
    """

    configuration_dictionary = json.loads(configuration_json)
    return SimulatorConfiguration.build_configuration_from_dictionary(configuration_dictionary, boid_net)



  @staticmethod
  def build_configuration_from_dictionary(configuration_dictionary: dict, boid_net: BoidNet = None):
    """
    Instantiates a simulator configuration object, based on the provided dictionary.

    Pass the boid net if you are instantiating a neural network boid.
    """
    return SimulatorConfiguration(
      retrieve_agent_generator(configuration_dictionary['agent']['agent_generator_name'], boid_net),
      retrieve_projectile_generator(configuration_dictionary['projectile']['projectile_generator_name']),
      retrieve_obstacle_generator(configuration_dictionary['obstacle']['obstacle_generator_name']),
      TargetArea(np.array(configuration_dictionary['target_area']['position']), configuration_dictionary['target_area']['size']),
      MapStructure(configuration_dictionary['map_structure']['map_height'], configuration_dictionary['map_structure']['safe_area_width'], configuration_dictionary['map_structure']['danger_area_width']),
      configuration_dictionary['delta_time'],
      configuration_dictionary['max_ticks'],
      configuration_dictionary['agent']['swarm_distance'],
      configuration_dictionary['projectile']['swarm_distance'],
      configuration_dictionary['target_delta_time'],
      configuration_dictionary['agent']['agent_size'],
      configuration_dictionary['agent']['agent_acc_limit'],
      configuration_dictionary['agent']['agent_perception_distance'],
      configuration_dictionary['agent']['num_agents'],
      configuration_dictionary['obstacle']['obstacle_min_size'],
      configuration_dictionary['obstacle']['obstacle_max_size'],
      configuration_dictionary['obstacle']['num_obstacles'],
      configuration_dictionary['projectile']['projectile_size'],
      configuration_dictionary['projectile']['projectile_acc_limit'],
      configuration_dictionary['projectile']['projectile_perception_distance'],
      configuration_dictionary['projectile']['num_projectiles']
    )

  
  def get_max_ticks(self):
    return self.max_ticks
  
  def get_target_delta_time(self):
    return self.target_delta_time
  

  def build_environment(self):
    """
    Instantiate a new environment our of the configuration.
    """

    return Environment(
      self.agent_generator(self.agent_swarm_distance, self.map_structure, self.agent_size, self.agent_acc_limit, self.agent_perception_distance, self.num_agents),
      lambda : self.projectile_generator(self.projectile_swarm_distance, self.map_structure, self.projectile_size, self.projectile_acc_limit, self.projectile_perception_distance,  self.num_projectiles),
      self.obstacle_generator(self.map_structure, self.obstacle_min_size, self.obstacle_max_size, self.num_obstacles, self.seed),
      self.target_area,
      self.map_structure,
      self.delta_time, 
      self.max_ticks
    )
