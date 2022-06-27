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

from projectiles.boid_projectile import BoidsProjectile


class PurePngBoidsProjectile(BoidsProjectile):
  """
  Implements a projectile law that makes use of proportional navigation to enable the projectiles
  to intercept the swarm agents.
  """

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)

    self.png_constant = 3.0
    self.projectile_max_velocity = 1.0

    self.last_line_of_sight_angle = 0

  def _compute_steering_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Avoid adding a steering velocity.
    """

    target_velocity_vector = self._get_target_velocity_vector(current_tick, delta_time, agent_perception)
    target_velocity_speed = np.linalg.norm(target_velocity_vector)
    target_velocity_direction = target_velocity_vector / target_velocity_speed

    line_of_sight_direction = self._get_line_of_sight_direction(current_tick, delta_time, agent_perception)

    coeff = -(target_velocity_speed / self.projectile_max_velocity) * \
        np.linalg.norm(
        np.cross(target_velocity_direction, line_of_sight_direction))

    matrx = np.array([
      [np.sqrt(1 - coeff ** 2), -coeff],
      [coeff, np.sqrt(1 - coeff ** 2)]
    ])

    velocity_direction = matrx.T @ line_of_sight_direction

    return self.steer_factor * self.projectile_max_velocity * velocity_direction

  
  def _compute_acceleration(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the acceleration of the projectile.
    """

    self.acceleration = 0.0
  



  def _get_target_velocity_vector(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the target's velocity vector.
    """

    target_agents = agent_perception.get_swarm_agents()
    target_position = self._get_target_position(current_tick, delta_time, agent_perception)
    swarm_base_position = agent_perception.get_target_area().position

    target_direction = (swarm_base_position - target_position) / np.linalg.norm(swarm_base_position - target_position)

    average_velocity = sum(map(lambda a: a.get_velocity(), target_agents)) / len(target_agents)

    average_target_velocity_vector = np.dot(average_velocity, target_direction) * target_direction

    return average_target_velocity_vector


  # def _get_los_change(selc, current_tick: int, delta_time: float, agent_perception: AgentPerception)-> float:
  #   """
  #   Return the change rate of the line of sight angle.
  #   """

  #   return 0.0

  # def _compute_steering_acceleration(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
  #   """
  #   Compute the steering acceleration (that achieves the pronav)
  #   """

  #   target_velocity_vector = self._get_target_velocity_vector(current_tick, delta_time, agent_perception)


  #   target_velocity_direction = target_velocity_vector / \
  #       np.linalg.norm(target_velocity_vector)

  #   line_of_sight_direction = self._get_line_of_sight_direction(current_tick, delta_time, agent_perception)
  #   target_speed = np.linalg.norm(target_velocity_vector)

    
  #   # compute the acceleration coefficient.
  #   acc_coeff = np.linalg.norm(np.cross(target_velocity_direction, line_of_sight_direction)) * (target_speed / self.projectile_max_velocity)

  #   # compute the absolute value for the acceleration
  #   acc_absolute_value = self.png_constant * self.projectile_max_velocity * self._get_los_change(current_tick, delta_time, agent_perception)

  #   # compute the direction of the acceleration
  #   rot_matrix = np.array([
  #     [-acc_coeff, -np.sqrt(1 - acc_coeff** 2)],
  #     [np.sqrt(1 - acc_coeff**2), -acc_coeff]
  #   ])

  #   acc_direction = line_of_sight_direction @ rot_matrix.T


  #   return acc_absolute_value / acc_direction

    





