from typing import List

import numpy as np


class LocalPlanner:
  """
  Given an global plan, and the current position of the agent, outputs the direction the agent should travel.
  """

  def __init__(self, plan: List[np.ndarray], target_tolerance: float) -> None:
    """
    Initializes the local planner
    """
    self.plan: List[np.ndarray] = plan
    self.target_tolerance: float = target_tolerance
  
  def plan_movement_direction(self, position: np.ndarray) -> np.ndarray:
    """
    Given the position, outputs a normalized vector indicating the travel direction (in global coordinates).
    """
    pass
  
  def is_target_reached(self, position: np.ndarray) -> np.ndarray:
    """
    Returns true if the target was reached.
    """
    plan = self.plan
    return np.linalg.norm(position - plan[len(plan) - 1]) <= self.target_tolerance


