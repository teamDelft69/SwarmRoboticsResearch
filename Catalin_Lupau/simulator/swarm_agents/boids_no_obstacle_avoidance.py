import numpy as np
from entities.Agent import AgentPerception

from swarm_agents.boids_agent import BoidsAgent


class BoidsNoObstacleAvoidance(BoidsAgent):
  """
  Dummy boids that does not have any obstacle avoidance. Used as baseline for the other obstacle avoidance algorithms.
  """

  def _compute_obstacle_avoidance_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    return np.zeros(2)

