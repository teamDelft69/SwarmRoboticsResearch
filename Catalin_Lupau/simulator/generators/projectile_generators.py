"""
Contains projectile generator functions
"""

import numpy as np
from entities.Agent import Agent
from entities.MapStructure import MapStructure
from projectiles.boid_projectile import BoidsProjectile
from projectiles.clustering_boid_projectile import ClusteringBoidsProjectile
from projectiles.pure_png_boid_projectile import PurePngBoidsProjectile
from swarm_agents.boids_agent import BoidsAgent
from swarm_agents.damper_agent import DamperAgent


def boids_projectile_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height() / 4
  min_pos_x = map_structure.get_safe_area_width() / 2
  max_pos_x = map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(BoidsProjectile(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'projectile_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def png_projectile_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height() / 4
  min_pos_x = map_structure.get_safe_area_width() / 2
  max_pos_x = map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(PurePngBoidsProjectile(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'projectile_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def clustering_projectile_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height() / 4
  min_pos_x = map_structure.get_safe_area_width() / 2
  max_pos_x = map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(ClusteringBoidsProjectile(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'projectile_{i + 1}', agent_perception_distance, swarm_distance))

  return agents

def retrieve_projectile_generator(generator_name: str):
  """
  Retrieves the right generator based on the provided generator name
  """

  if generator_name == 'boids':
    return boids_projectile_generator

  if generator_name == 'png':
    return png_projectile_generator
  
  if generator_name == 'clustering':
    return clustering_projectile_generator 

  return (lambda x, y, z, v, a, b: [])

