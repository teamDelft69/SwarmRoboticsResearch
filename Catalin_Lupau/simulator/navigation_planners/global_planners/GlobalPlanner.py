from typing import List, Tuple
from xmlrpc.client import SafeTransport

import numpy as np
from entities.MapStructure import MapStructure
from entities.Obstacle import Obstacle


class GlobalPlanner:
  """
  Abstract class defining a generic global planner.
  """

  def __init__(self, obstacle_list: List[Obstacle], map_structure: MapStructure, granularity: float, safety_distance: float) -> None:
    """
    Initializes the global planner.
    obstacle_list - the list of obstacles on the map
    map_structure - the structure of the map
    granularity - the size of one square of the digitalized map.
    """
    self.obstacle_list = obstacle_list
    self.map_structure = map_structure
    self.granularity = granularity
    self.map_width = map_structure.danger_area_width + map_structure.safe_area_width
    self.map_height = map_structure.map_height
    self.safety_distance = safety_distance

    self.digital_map = self._create_digital_map(obstacle_list, self.map_width, self.map_height, self.granularity)

  
  def find_path(self, starting_position: np.ndarray, target_position: np.ndarray) -> List[np.ndarray]:
    """
    Returns the path that needs to be followed from the starting position to the target position.
    The path is represented as a list of positions that the agent needs to go to.

    To be implemented by each global planner.
    """
    pass

  
  def _get_digitalized_coordinates(self, position: np.ndarray, granularity: float) -> Tuple[int]:
    """
    Return the coordinates in the digitalized version of the map
    """

    real_x, real_y = position[0], position[1]

    digitalized_col = int(real_x / granularity)
    digitalized_row = int(real_y / granularity)

    return digitalized_row, digitalized_col

  def _get_real_coordinates(self, digital_coordinates: Tuple[int], granularity: float) -> np.ndarray:
    """
    Return the real coordinates (in the center of the square) based on the digital ones
    """

    real_x = float(digital_coordinates[1]) + granularity / 2.0
    real_y = float(digital_coordinates[0]) + granularity / 2.0

    return np.array([real_x, real_y])

  
  def _is_inside_obstacle(self, point: np.ndarray, obstacle_list: List[Obstacle]) -> bool:
    """
    Return true if the point is inside the obstacle, false otherwise.
    """

    for obstacle in obstacle_list:
      if np.linalg.norm(point - obstacle.get_position()) <= obstacle.get_radius() + self.granularity:
        return True
    
    return False

  
  def _create_digital_map(self, obstacle_list: List[Obstacle], map_width: float, map_height: float, granularity: float) -> List[int]:
    """
    Creates a digital map of the envirionment, that marks with a '0', the areas that have an obstacle
    and with a '1', the ones that don't.
    """

    num_cols = int(map_width / granularity) + 1
    num_rows = int(map_height / granularity) + 1

    mp = []

    for row in range(num_rows):
      mp.append([])
      for col in range(num_cols):
        # if the cell corresponds to an obstacle, add a 1, else add a 0
        mp[row].append(0 if self._is_inside_obstacle(self._get_real_coordinates([row, col], granularity), self.obstacle_list) else 1)

    return mp
