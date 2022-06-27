import math
from enum import Enum
from typing import List

import numpy as np
from entities.Agent import Agent, AgentPerception
from entities.Obstacle import Obstacle
from navigation_planners.global_planners.AStarPlanner import AStarPlanner
from navigation_planners.global_planners.GlobalPlanner import GlobalPlanner
from navigation_planners.local_planners.BasicPlanner import BasicPlanner
from numpy import ndarray

from swarm_agents.boids_agent import BoidsAgent
from swarm_agents.steer_away_boids import SteerAwayBoids


class EvasiveSteerAwayBoids(SteerAwayBoids):
  """
  Boids that use an evasive maneuver whenever the projectiles get close.
  """

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)

    self.evasive_factor = 0.3

  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    self.velocity += self._compute_centering_velocity(current_tick, delta_time, agent_perception) + self._compute_avoidance_velocity(
        current_tick, delta_time, agent_perception) + self._compute_match_velocity(current_tick, delta_time, agent_perception) \
           + self._compute_steering_velocity(current_tick, delta_time, agent_perception) \
           + self._compute_evasive_velocity(current_tick, delta_time, agent_perception)

    # compute the rotation matrix making the object avoid the obstacle
    obstacle_avoidance_rotation_matrix: np.ndarray = self._compute_obstacle_avoidance_rotation_matrix(
        current_tick, delta_time, agent_perception)

    # apply the computed rotation matrix
    self.velocity = self.velocity @ obstacle_avoidance_rotation_matrix.T

    if np.linalg.norm(self.velocity) > 1.0:
        self.velocity = 1.0 * self.velocity / np.linalg.norm(self.velocity)
  
  def _compute_evasive_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the avoidance velocity
    """

    move_direction = np.zeros(2)
    neighbors = self._get_nearby_projectiles(self.id, agent_perception, self.perception_distance)

    for neighbor in neighbors:
      if np.linalg.norm(neighbor.get_position() - self.position) <= self.swarm_distance:
        move_direction += (self.position - neighbor.get_position())

    return move_direction * self.evasive_factor
  

  



