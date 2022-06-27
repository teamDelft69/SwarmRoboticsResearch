import numpy as np


class Obstacle:
  """
  Represents an obstacle as part of the environment
  """

  def __init__(self, position: np.ndarray, radius: float) -> None:
    self.position = position
    self.radius = radius

  
  def get_position(self) -> np.ndarray:
    """
    Return the position of the obstacle.
    """
    return self.position
  
  def get_radius(self) -> float:
    """
    Return the radius of the obstacle.
    """
    return self.radius

  def is_in_collision_with(self, agent) -> bool:
    """
    Returns true if the obstacle is in collision with the agent.
    """

    agent_position = agent.get_position()
    agent_radius = agent.get_size()

    center_dist = np.linalg.norm(agent_position - self.position)

    return center_dist < self.radius + agent_radius
  
  def to_summary(self) -> dict:
    """
    Returns a python dictionary representation of the obstacle.
    """

    return {
      'position': list(self.position),
      'radius': self.radius
    }
