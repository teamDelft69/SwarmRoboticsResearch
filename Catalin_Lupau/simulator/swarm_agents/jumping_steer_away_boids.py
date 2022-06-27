import math
from enum import Enum
from typing import List

import numpy as np
from entities.Agent import Agent, AgentPerception
from entities.Obstacle import Obstacle
from navigation_planners.global_planners.AStarPlanner import AStarPlanner
from navigation_planners.global_planners.GlobalPlanner import GlobalPlanner
from navigation_planners.local_planners.BasicPlanner import BasicPlanner

from swarm_agents.steer_away_boids import SteerAwayBoids


class JumpingSteerAwayBoids(SteerAwayBoids):
  """
  Adds an 'explosive' evasive behaviour to the boid, i.e. whenever the enemy gets withing the perception distance,
  apply a sudden powerful velocity in a random direction.
  """

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)

    self.avoidance_triggered = False
    self.explosion_factor = 2.0

  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Computes the velocity of each boid.
    """
    super()._compute_velocity(current_tick, delta_time, agent_perception)

    if (self._check_if_projectiles_in_area(agent_perception)) and (not self.avoidance_triggered):
      direction = np.random.random(size = 2)
      direction = direction / np.linalg.norm(direction)
      self.avoidance_triggered = True

      self.velocity += direction * self.explosion_factor
    
    if (not self._check_if_projectiles_in_area(agent_perception)) and (self.avoidance_triggered):
      self.avoidance_triggered = False
  

  def _check_if_projectiles_in_area(self, agent_perception: AgentPerception) -> bool:
    """
    Returns true if there are projectiles in the area, false otherwise.
    """

    nearby_projectiles = self._get_nearby_projectiles(self.id, agent_perception, self.perception_distance)
    return len(nearby_projectiles) > 0


