from turtle import pos
from typing import List

import numpy as np
from navigation_planners.local_planners.LocalPlanner import LocalPlanner


class BasicPlanner(LocalPlanner):
  """
  Very basic local planner, that does not make use of any obstacle avoidance algorithms.
  """

  def __init__(self, plan: List[np.ndarray], target_tolerance: float) -> None:
      super().__init__(plan, target_tolerance)

  def plan_movement_direction(self, position: np.ndarray) -> np.ndarray:
    """
    Return the direction in which the agent should move.
    """

    # if the target was reached, there is no point in moving in any direction
    if self.is_target_reached(position):
      return np.zeros(2)

    # get the the next position in row
    closest_index = self._get_closest_position_in_plan(position, self.plan)
    next_position = self.plan[min(closest_index + 1, len(self.plan) - 1)]

    return (next_position - position) / np.linalg.norm(next_position - position)


  
  def _get_closest_position_in_plan(self, position: np.ndarray, plan: List[np.ndarray]) -> int:
    """
    Return the index of the closest position in the plan.
    """

    closest_index = 0
    current_error = float('inf')

    for index, plan_position in enumerate(plan[:-1]):
      if np.linalg.norm(position - plan_position) < current_error:
        current_error = np.linalg.norm(position - plan_position)
        closest_index = index
    
    return closest_index


