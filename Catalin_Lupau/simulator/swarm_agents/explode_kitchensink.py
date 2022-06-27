import math
from enum import Enum
from typing import List

import numpy as np
from entities.Agent import Agent, AgentMessage, AgentPerception
from entities.Obstacle import Obstacle
from navigation_planners.global_planners.AStarPlanner import AStarPlanner
from navigation_planners.global_planners.GlobalPlanner import GlobalPlanner
from navigation_planners.local_planners.BasicPlanner import BasicPlanner
from numpy import ndarray

from swarm_agents.boids_agent import BoidsAgent
from swarm_agents.steer_away_boids import SteerAwayBoids
from swarm_agents.steer_away_directed_whirlwind_repulsive_boids import \
    SteerAwayDirectedWhirlwindRepulsiveBoids


class ExplosiveKitchenSinkBoids(SteerAwayDirectedWhirlwindRepulsiveBoids):
  """
  Boids implementing an projectile avoidance strategy that collapses the swarm, 
  whenever the projectiles get close to one of the agents. The swarm is collapses in an explosion-like manner,
  in a direction opposite to the center of mass of the swarm. 

  The boids are assumed to be able to alert the nearby boids of the danger (boid communication is assumed)
  """

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size,
                     acc_limit, id, perception_distance, swarm_distance)

    self.explosive_factor = 20.0
    self.communication_distance = 5.0

    self.explosion_triggered = False
    self.explosion_delta_time = 50  # number of ticks
    self.explosion_trigger_time = -1.0
    # stores the last agent perception

  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):

    self.velocity += self._compute_centering_velocity(current_tick, delta_time, agent_perception) + self._compute_avoidance_velocity(
        current_tick, delta_time, agent_perception) + self._compute_match_velocity(current_tick, delta_time, agent_perception) \
        + self._compute_steering_velocity(current_tick, delta_time, agent_perception) \
        + self._compute_explosive_velocity(
        current_tick, delta_time, agent_perception) \
        + self._compute_obstacle_avoidance_velocity(current_tick, delta_time, agent_perception)

    # compute the rotation matrix making the object avoid the obstacle
    obstacle_avoidance_rotation_matrix: np.ndarray = self._compute_obstacle_avoidance_rotation_matrix(
        current_tick, delta_time, agent_perception)

    # apply the computed rotation matrix
    self.velocity = self.velocity @ obstacle_avoidance_rotation_matrix.T

    if np.linalg.norm(self.velocity) > 1.0:
        self.velocity = 1.0 * self.velocity / np.linalg.norm(self.velocity)

    self._start_explosion_if_necessary(
        current_tick, delta_time, agent_perception)

  def _start_explosion_if_necessary(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Start an explosion if it is necessary
    """

    # if the explosion was already triggered, no point in doing anything.
    if self.explosion_triggered:
      return

    # check if there are nearyby projectiles
    neighbors = self._get_nearby_projectiles(
        self.id, agent_perception, self.perception_distance)

    # if some projectiles are detected, trigger an explosion
    if len(neighbors) > 0:
      self.trigger_explosion(current_tick, delta_time, agent_perception)

  def _compute_explosive_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the velocity that causes the explosion.
    """

    if self.explosion_triggered is False:
      return np.zeros(2)

    # check if the explosion ended.
    if current_tick - self.explosion_trigger_time > self.explosion_delta_time:
      return np.zeros(2)

    swarm_agents: List[Agent] = agent_perception.get_swarm_agents()

    swarm_center = np.zeros(2)

    for agent in swarm_agents:
      swarm_center += agent.get_position()

    swarm_center /= len(swarm_agents)

    return (self.position - swarm_center) * self.explosive_factor / (np.linalg.norm(self.position - swarm_center) ** 2 + 0.05)

  def _on_message_received(self, message: AgentMessage, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    if message.get_id() == 'explosion':
      self.trigger_explosion(agent_perception)

  def trigger_explosion(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Call this function to trigger the explosion in the swarm
    """

    if (self.explosion_triggered) or (agent_perception is None):
      return

    self.explosion_triggered = True
    self.explosion_trigger_time = current_tick

    # send the message that triggers an explosion to the neighbors
    swarm_agents = agent_perception.get_swarm_agents()

    # propoagate the explosion to other agents in the swarm
    for agent in swarm_agents:
      if agent.get_id() == self.id:
        continue

      if np.linalg.norm(self.position - agent.get_position()) < self.communication_distance:
        agent.process_message(AgentMessage('explosion', self.id))
