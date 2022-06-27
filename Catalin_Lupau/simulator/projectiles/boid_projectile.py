import math
from enum import Enum
from typing import List

import numpy as np
from entities.Agent import Agent, AgentPerception
from entities.Obstacle import Obstacle
from navigation_planners.global_planners.AStarPlanner import AStarPlanner
from navigation_planners.global_planners.GlobalPlanner import GlobalPlanner
from navigation_planners.local_planners.BasicPlanner import BasicPlanner


class BoidsProjectile(Agent):
  """
  Agent that implements the boids model of movement.
  """

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
      """
      Initializes a boids projectile.
      """
      super().__init__(init_position, init_velocity, size,
                       acc_limit, id, perception_distance, swarm_distance)

      # self.min_distance = 10 # the distance to stay away from others
      self.avoid_factor = 0.05  # adjust velocity by this percent
      self.match_factor = 0.05  # adjust velocity by this percent
      self.center_factor = 0.005  # adjust velocity by this percent
      self.steer_factor = 0.1
      self.obstacle_steer_factor = 0.3

  def _get_neighbors(self, agent_id: str, agent_perception: AgentPerception):
    """
    Return the neighbors of the provided agent.
    """
    projectile_agents = agent_perception.get_projectiles()

    agent: Agent = self._retrieve_agent_based_on_id(
        projectile_agents, agent_id)

    neighbors = []

    for other_agent in projectile_agents:
      if other_agent.get_id() == agent.get_id():
        continue

      distance = np.linalg.norm(
          agent.get_position() - other_agent.get_position())

      if distance <= self.perception_distance:
        neighbors.append(other_agent)

    return neighbors

  def _get_nearby_obstacles(self, agent_id: str, agent_perception: AgentPerception):
    """
    Return the list of nearby obstacles
    """
    projectile_agents = agent_perception.get_projectiles()
    agent: Agent = self._retrieve_agent_based_on_id(
        projectile_agents, agent_id)

    nearby_obstacles = []
    obstacle_list = agent_perception.get_obstacles()

    for obstacle in obstacle_list:
      if np.linalg.norm(agent.get_position() - obstacle.get_position()) - obstacle.get_radius() <= self.perception_distance:
        nearby_obstacles.append(obstacle)

    return nearby_obstacles

  def _compute_centering_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the centering velocity of the boids.
    """

    avg_position_neighbors = np.zeros(2)
    neighbors = self._get_neighbors(self.id, agent_perception)
    num_neighbors = len(neighbors)


    for neighbor in neighbors:
      avg_position_neighbors += neighbor.get_position()

    if num_neighbors > 0:
      avg_position_neighbors /= num_neighbors

    return (avg_position_neighbors - self.position) * self.center_factor

  def _compute_avoidance_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the avoidance velocity
    """

    move_direction = np.zeros(2)
    neighbors = self._get_neighbors(self.id, agent_perception)

    for neighbor in neighbors:
      if np.linalg.norm(neighbor.get_position() - self.position) <= self.swarm_distance:
        move_direction += (self.position - neighbor.get_position())

    return move_direction * self.avoid_factor

  def _compute_obstacle_avoidance_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    try to avoid nearby obstacles
    """

    move_direction = np.zeros(2)
    neighbor_obstacles: List[Obstacle] = self._get_nearby_obstacles(
        self.id, agent_perception)

    for obstacle in neighbor_obstacles:
      move_direction += (self.position - obstacle.get_position()) * (
          1.0 + obstacle.get_radius() / np.linalg.norm(self.position - obstacle.get_position()))

    return move_direction * self.obstacle_steer_factor

  def _compute_match_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the centering velocity of the boids.
    """

    avg_velocity_neighbors = np.zeros(2)
    neighbors = self._get_neighbors(self.id, agent_perception)
    num_neighbors = len(neighbors)

    for neighbor in neighbors:
      avg_velocity_neighbors += neighbor.get_velocity()

    if num_neighbors > 0:
      avg_velocity_neighbors /= num_neighbors

    return (avg_velocity_neighbors - self.velocity) * self.match_factor

  def _get_target_position(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Return the position of the target (i.e. the center of the target swarm).
    """

    target_agents = agent_perception.get_swarm_agents()
    target_position = sum(map(lambda a: a.get_position(),
                          target_agents)) / len(target_agents)
    return target_position
  
  def _get_line_of_sight_direction(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Return the direction of the line of sight
    """

    target_position = self._get_target_position(current_tick, delta_time, agent_perception)

    return (target_position - self.get_position()) / np.linalg.norm(target_position - self.get_position())


  def _compute_steering_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the steering velocity of the boids
    """

    target_position = self._get_target_position(current_tick, delta_time, agent_perception)

    return (target_position - self.position) * self.steer_factor

  def _retrieve_agent_based_on_id(self, swarm_agents: List[Agent], id: str):
    """
    Retrieves the agent based on the provided id.

    If no agents was found, raise exception.
    """

    for agent in swarm_agents:
      if agent.id == id:
        return agent

    return None

  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
      self.velocity += self._compute_centering_velocity(current_tick, delta_time, agent_perception) + self._compute_avoidance_velocity(
          current_tick, delta_time, agent_perception) + self._compute_match_velocity(current_tick, delta_time, agent_perception) + self._compute_steering_velocity(current_tick, delta_time, agent_perception) + self._compute_obstacle_avoidance_velocity(current_tick, delta_time, agent_perception)

      if np.linalg.norm(self.velocity) > 1.0:
        self.velocity = 1.0 * self.velocity / np.linalg.norm(self.velocity)
