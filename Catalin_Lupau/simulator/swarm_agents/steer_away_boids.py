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


class SteerAwayBoids(BoidsAgent):
  """
  Implements a set of boids that work by steering away, when they have an obstacle in front of them.
  """

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)

    self.delta_steer_angle = 0.15 * np.pi
    self.steer_away_distance = 30


  
  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    self.velocity += self._compute_centering_velocity(current_tick, delta_time, agent_perception) + self._compute_avoidance_velocity(
        current_tick, delta_time, agent_perception) + self._compute_match_velocity(current_tick, delta_time, agent_perception) + self._compute_steering_velocity(current_tick, delta_time, agent_perception)

    # compute the rotation matrix making the object avoid the obstacle
    obstacle_avoidance_rotation_matrix: np.ndarray = self._compute_obstacle_avoidance_rotation_matrix(current_tick, delta_time, agent_perception)

    # apply the computed rotation matrix
    self.velocity = self.velocity @ obstacle_avoidance_rotation_matrix.T

    if np.linalg.norm(self.velocity) > 1.0:
        self.velocity = 1.0 * self.velocity / np.linalg.norm(self.velocity)

  def _compute_obstacle_avoidance_rotation_matrix(self, current_tick: int, delta_time: float, agent_perception: AgentPerception) -> np.ndarray:
    """
    Compute the rotation matrix corresponding to performing obstacle avoidance.
    """

    nearby_obstacles = self._get_nearby_obstacles(self.id, agent_perception, self.steer_away_distance)

    sorted_nearby_obstacles = sorted(nearby_obstacles, key=lambda o: np.linalg.norm(o.get_position() - self.position) - o.get_radius())

    for obstacle in sorted_nearby_obstacles:
      check_result = self._check_intersection(self.position, self.velocity, self.size, obstacle)

    

      if check_result != 0:
        return self._compute_rotation_matrix(self.delta_steer_angle)
    
    # no obstacles in range, so do not compute 
    return self._compute_rotation_matrix(0)

    
  def _compute_rotation_matrix(self, angle: float) -> np.ndarray:
    """
    Compute the rotation matrix corresponding to the angle
    """

    return np.array(
      [[np.cos(angle), -np.sin(angle)],
       [np.sin(angle),  np.cos(angle)]
      ]
    )


  def _check_intersection(self, position: np.ndarray, velocity: np.ndarray, agent_size: float, obstacle: Obstacle):
    """
    Assuming that the swarm agent follows the trajectory corresponding to the given velocity, check if it would intersect any obstacles.
    Returns 0, if there is no intersection. If the intersection is above the center of the object, return 1.
    If the intersection is below the center of the object, return -1.
    """

    limit_angle = (agent_size + obstacle.get_radius()) / np.linalg.norm(position - obstacle.get_position())


    relative_position = obstacle.get_position() - position

    real_angle = np.linalg.norm(np.cross(velocity, relative_position)) / np.dot(velocity, relative_position)


    # if there is an intersection, return the direction for the change
    if (-limit_angle <= real_angle) and (real_angle <= limit_angle):
      return 1 if real_angle > 0 else -1
    
    return 0
