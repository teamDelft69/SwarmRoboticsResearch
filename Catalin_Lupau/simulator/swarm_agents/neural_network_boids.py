import math
import os
import pickle
from enum import Enum
from typing import List

import numpy as np
import pygad
import pygad.torchga as pt
import torch
import torch.nn as nn
import torch.nn.functional as F
from entities.Agent import Agent, AgentPerception
from entities.Obstacle import Obstacle
from navigation_planners.global_planners.AStarPlanner import AStarPlanner
from navigation_planners.global_planners.GlobalPlanner import GlobalPlanner
from navigation_planners.local_planners.BasicPlanner import BasicPlanner
from numpy import ndarray
from paths import *

from swarm_agents.boids_agent import BoidsAgent


class BoidNet:
  """
  Defines the neural network that drives the boid.
  """

  def __init__(self, solution) -> None:
    self.model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 12), nn.Tanh(), nn.Linear(12, 5))
    self.solution = solution

  def get_model(self):
    return self.model
  

  def forward(self, input: np.ndarray) -> np.ndarray:
    """
    Propagate the data forward through the network.
    Return the results of propagating the provided data through the network.
    """

    predictions = pygad.torchga.predict(
        model=self.model, solution=self.solution, data=input)

    return predictions.detach().numpy()

  
  @staticmethod
  def load_from_file(filename: str):
    """
    load the boid neural network from file.
    """
    solution = np.load(open(os.path.join(NEURAL_NETWORKS_PATH, filename) + '.pkl', 'rb'))
    print(solution)
    
    return BoidNet(solution)
  
  @staticmethod
  def init_with_random_weights():
    """
    Initializes a boid net network with random weights
    """
    random_weights = np.random.random(size=380)
    return BoidNet(random_weights)

    







class NeuralNetworkBoids(BoidsAgent):
  """
  Implements an agent that uses a neural network based model to navigate to the destination.
  """

  def __init__(self, neural_net: BoidNet, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)
    self.neural_net = neural_net

  

  def get_closest_obstacle(self, agent_perception: AgentPerception) -> Obstacle:
    """
    Returns the obstacle that is closest to the agent.
    """

    obstacles = agent_perception.get_obstacles()
    closest_obstacle = obstacles[0]
    
    for obstacle in obstacles:
      if np.linalg.norm(self.position - obstacle.get_position()) < np.linalg.norm(self.position - closest_obstacle.get_position()):
        closest_obstacle = obstacle
    
    return closest_obstacle
  

  def get_swarm_center(self, agent_perception: AgentPerception) -> np.ndarray:
    """
    Return the center of the swarm.
    """

    center_position = np.zeros(2)
    swarm_agents = agent_perception.get_swarm_agents()

    for agent in swarm_agents:
      center_position += agent.get_position()
    
    if len(swarm_agents) > 0:
      center_position /= len(swarm_agents)
    
    return center_position

  def get_closest_projectile(self, agent_perception: AgentPerception) -> np.ndarray:
    nearby_projectiles = self._get_nearby_projectiles(self.id, agent_perception, self.perception_distance)

    if len(nearby_projectiles) == 0:
      return None

    closest_projectile = nearby_projectiles[0]

    for projectile in nearby_projectiles:
      if np.linalg.norm(self.position - projectile.get_position()) < np.linalg.norm(self.position - closest_projectile.get_position()):
        closest_projectile = projectile
    

    return closest_projectile




  def _compute_velocity_parameters(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the parameters used by the boid model using the neural network.

    Updates the following parameters of the boid
    [avoid factor,
    match factor,
    center factor,
    steer factor,
    obstacle steer factor]
    """

    closest_obstacle = self.get_closest_obstacle(agent_perception)
    target = agent_perception.get_target_area()
    swarm_center_position = self.get_swarm_center(agent_perception)
    closest_projectile = self.get_closest_projectile(agent_perception)

    relative_target_pos = target.get_position() - self.get_position()
    relative_obstacle_pos = closest_obstacle.get_position() - self.get_position()
    relative_position_swarm_center = swarm_center_position - self.get_position()

    # generate a large random position, in case there is no projectile in the neighborhood
    gen_random_large_coord = lambda: np.random.choice([1, -1]) * (np.random.random() * 1000 + 1000)
    relative_projectile_pos = np.array([gen_random_large_coord(), gen_random_large_coord()])

    if closest_projectile is not None:
      relative_projectile_pos = closest_projectile.get_position() - self.get_position()
    
    # forward the input data through the neural network

    input = torch.tensor(np.array([
        relative_target_pos[0],
        relative_target_pos[1],
        relative_obstacle_pos[0],
        relative_obstacle_pos[1],
        relative_position_swarm_center[0],
        relative_position_swarm_center[1],
        relative_projectile_pos[0],
        relative_projectile_pos[1]
    ], dtype=np.float32))

    output = self.neural_net.forward(input)
    # print('nn result')
    # print(output)

    self.avoid_factor = output[0]
    self.match_factor = output[1]
    self.center_factor = output[2]
    self.steer_factor = output[3]
    self.obstacle_steer_factor = output[4]

    
  
  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Computes the velocity of the boid
    """
    
    # use the associated neural network to compute the velocity of the
    self._compute_velocity_parameters(current_tick, delta_time, agent_perception)

    self.velocity += self._compute_centering_velocity(current_tick, delta_time, agent_perception) + self._compute_avoidance_velocity(
          current_tick, delta_time, agent_perception) + self._compute_match_velocity(current_tick, delta_time, agent_perception) + self._compute_steering_velocity(current_tick, delta_time, agent_perception) + self._compute_obstacle_avoidance_velocity(current_tick, delta_time, agent_perception)

    if np.linalg.norm(self.velocity) > 1.0:
      self.velocity = 1.0 * self.velocity / np.linalg.norm(self.velocity)

    

    










