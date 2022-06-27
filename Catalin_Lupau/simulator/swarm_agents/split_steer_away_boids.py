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


class SplitSteerAwayBoids(SteerAwayBoids):
  """
  Boids that split from the beginning in 2 groups, tricking the center of mass projectiles.
  """


  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)
    self.anti_neighbor_factor = 10.0


  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    self.velocity += self._compute_centering_velocity(current_tick, delta_time, agent_perception) + self._compute_avoidance_velocity(
        current_tick, delta_time, agent_perception) + self._compute_match_velocity(current_tick, delta_time, agent_perception) \
           + self._compute_steering_velocity(current_tick, delta_time, agent_perception) + \
             self._compute_anti_neighbor_velocity(current_tick, delta_time, agent_perception)

    # compute the rotation matrix making the object avoid the obstacle
    obstacle_avoidance_rotation_matrix: np.ndarray = self._compute_obstacle_avoidance_rotation_matrix(
        current_tick, delta_time, agent_perception)

    # apply the computed rotation matrix
    self.velocity = self.velocity @ obstacle_avoidance_rotation_matrix.T

    if np.linalg.norm(self.velocity) > 1.0:
        self.velocity = 1.0 * self.velocity / np.linalg.norm(self.velocity)



  def _compute_anti_neighbor_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute a velocity that repulses the group from the agents that are not it's neighbors
    """
    anti_neighbors = self._get_neighbors(f'agent_{2 if self._get_parity(self.id) == 1 else 1}', agent_perception)

    anti_neighbor_pos = np.zeros(2)
    num = 0

    for anti_neighbor in anti_neighbors:
      if np.linalg.norm(anti_neighbor.get_position() - self.position) <= 20:
        anti_neighbor_pos += anti_neighbor.get_position()
        num += 1
    
    if num == 0:
      return np.zeros(2)

    if num > 0:
      anti_neighbor_pos /= num
    

    return (self.position - anti_neighbor_pos) * self.anti_neighbor_factor



  def _get_parity(self, agent_id: str):
    """
    Returns the parity of the agent id, 0 for even, 1 for odd
    """

    agent_number = int(agent_id.split('_')[1])


    return agent_number % 2

  def _get_neighbors(self, agent_id: str, agent_perception: AgentPerception):
    """
    Return the neighbors of the provided agent.
    """
    swarm_agents = agent_perception.swarm_agents

    

    agent: Agent = self._retrieve_agent_based_on_id(swarm_agents, agent_id)

    if agent is None:
      return []


    neighbors = []

    for other_agent in swarm_agents:
      if other_agent.get_id() == agent.get_id():
        continue

      if self._get_parity(other_agent.get_id()) != self._get_parity(agent.get_id()):
        continue

      distance = np.linalg.norm(
          agent.get_position() - other_agent.get_position())

      if distance <= self.perception_distance:
        neighbors.append(other_agent)

    # print(len(neighbors))
    return neighbors
