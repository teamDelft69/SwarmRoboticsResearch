import math
from enum import Enum
from typing import List

import numpy as np
from entities.Agent import Agent, AgentPerception
from entities.Obstacle import Obstacle
from entities.Projectile import Projectile
from navigation_planners.global_planners.AStarPlanner import AStarPlanner
from navigation_planners.global_planners.GlobalPlanner import GlobalPlanner
from navigation_planners.local_planners.BasicPlanner import BasicPlanner

from swarm_agents.steer_away_boids import SteerAwayBoids


class SteerAwayDirectedWhirlwindRepulsiveBoids(SteerAwayBoids):
  """
  Implements a whirlwind repuslive field, to deter obstacles.
  """

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size,
                     acc_limit, id, perception_distance, swarm_distance)

    self.whirlwind_constant = 10
    self.obstacle_steer_factor = 1.0

  def _anti_clockwise_field_equation(self, position: np.ndarray) -> np.ndarray:
    return np.array([position[0] - position[1] + self.whirlwind_constant * position[0] / (position[0] ** 2 + position[1] ** 2), position[0] + position[1] + self.whirlwind_constant * position[1] / (position[0] ** 2 + position[1] ** 2)])

  def _clockwise_field_equation(self, position: np.ndarray) -> np.ndarray:
    return np.array([position[0] + position[1] + self.whirlwind_constant * position[0] / (position[0] ** 2 + position[1] ** 2), position[1] - position[0] + self.whirlwind_constant * position[1] / (position[0] ** 2 + position[1] ** 2)])

  def _get_whirlwind_direction(self, object_position: np.ndarray, agent_position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
    """
    Return the direction of the repulsive vector corresponding to the whirlwind.
    """

    steering_direction = target_position - agent_position
    relative_object_position = object_position - agent_position

    if np.dot(steering_direction, relative_object_position) > 0:
      return self._clockwise_field_equation(agent_position - object_position)

    return self._anti_clockwise_field_equation(agent_position - object_position)

  def _compute_obstacle_avoidance_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    move_direction = np.zeros(2)
    neighbor_obstacles: List[Obstacle] = self._get_nearby_obstacles(
        self.id, agent_perception, self.swarm_distance)

    target_position = agent_perception.get_target_area().position

    for obstacle in neighbor_obstacles:
      move_direction += self._get_whirlwind_direction(
          obstacle.get_position(), self.position, target_position)

    return move_direction * self.obstacle_steer_factor

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
