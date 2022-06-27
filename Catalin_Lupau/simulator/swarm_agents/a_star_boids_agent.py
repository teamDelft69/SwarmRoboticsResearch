import math
from enum import Enum
from typing import List

import numpy as np
from entities.Agent import Agent, AgentPerception
from entities.Obstacle import Obstacle
from navigation_planners.global_planners.AStarPlanner import AStarPlanner
from navigation_planners.global_planners.GlobalPlanner import GlobalPlanner
from navigation_planners.local_planners.BasicPlanner import BasicPlanner

from swarm_agents.boids_agent import BoidsAgent


class AStarBoidsAgent(BoidsAgent):
  """
  Modification of the boids agent that uses the A* algorithm as the main driver behind the steering force, in order to improve obstacle
  avoidance
  """

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)


    self.local_planner: BasicPlanner = None


  def _init_planner(self, agent_perception: AgentPerception):
    """
    Initializes the planner used in the steering force.
    """
    global_planner = AStarPlanner(agent_perception.get_obstacles(), agent_perception.get_map_structure(), 1.0, 10 * self.size)
    path = global_planner.find_path(self.position, agent_perception.get_target_area().get_position())
    self.local_planner: BasicPlanner = BasicPlanner(path, 1.0)
  
  def _compute_steering_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the steering velocity based on the global and local planners.
    """


    # initialize the local planner
    if self.local_planner is None:
      self._init_planner(agent_perception)
    
    return self.local_planner.plan_movement_direction(self.position) * self.steer_factor


