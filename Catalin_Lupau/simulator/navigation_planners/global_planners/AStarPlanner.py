from typing import List, Tuple

import numpy as np
from entities.MapStructure import MapStructure
from entities.Obstacle import Obstacle
from navigation_planners.global_planners.GlobalPlanner import GlobalPlanner
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class AStarPlanner(GlobalPlanner):
  """
  An implementation of the AStar global planner. (It uses the A Star algorithm to find a path between 2 points in the map)
  """

  def __init__(self, obstacle_list: List[Obstacle], map_structure: MapStructure, granularity: float, safety_distance: float) -> None:
      super().__init__(obstacle_list, map_structure, granularity, safety_distance)
  
  def find_path(self, starting_position: np.ndarray, target_position: np.ndarray) -> List[np.ndarray]:
    """
    Returns the path that needs to be followed from the starting position to the target position.
    The path is represented as a list of positions that the agent needs to go to.

    This implementation uses the A star algorithm
    """
    
    digital_starting_poisition = self._get_digitalized_coordinates(starting_position, self.granularity)
    digital_target_position = self._get_digitalized_coordinates(target_position, self.granularity)

    grid = Grid(matrix=self.digital_map)

    start = grid.node(digital_starting_poisition[1], digital_starting_poisition[0])
    end = grid.node(digital_target_position[1], digital_target_position[0])

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, _ = finder.find_path(start, end, grid)

    real_path = list(map(lambda x: self._get_real_coordinates([x[1], x[0]], self.granularity), path))

    return real_path


