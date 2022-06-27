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


class SteerAwayRepulsiveBoids(SteerAwayBoids):
  """
  Boid that combines the steer away strategy with the respulsive field strategy, to avoid obstacles.
  """

  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    self.velocity += self._compute_centering_velocity(current_tick, delta_time, agent_perception) + self._compute_avoidance_velocity(
        current_tick, delta_time, agent_perception) + self._compute_match_velocity(current_tick, delta_time, agent_perception) \
           + self._compute_steering_velocity(current_tick, delta_time, agent_perception) \
             + self._compute_obstacle_avoidance_velocity(current_tick, delta_time, agent_perception)

    # compute the rotation matrix making the object avoid the obstacle
    obstacle_avoidance_rotation_matrix: np.ndarray = self._compute_obstacle_avoidance_rotation_matrix(
        current_tick, delta_time, agent_perception)

    # apply the computed rotation matrix
    self.velocity = self.velocity @ obstacle_avoidance_rotation_matrix.T

    if np.linalg.norm(self.velocity) > 1.0:
        self.velocity = 1.0 * self.velocity / np.linalg.norm(self.velocity)




  
