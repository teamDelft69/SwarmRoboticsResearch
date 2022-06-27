import numpy as np

from entities.Agent import Agent


class Projectile(Agent):
  """
  Represents a projectile.
  """


  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float, init_tick: int, lifetime_ticks: int = -1) -> None:
      super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)

      self.init_tick = init_tick
      self.lifetime_ticks = lifetime_ticks
      self.active = True

  # def __init__(self, size: float, init_pos: np.ndarray, velocity: np.ndarray, init_tick: int, lifetime_ticks: int = -1) -> None:
  #   """
  #   Constructs a projectile.

  #   size - the size of the projectile
  #   init_pos - the initial position of the projectile
  #   velocity - the velocity of the projectile (constant velocity)
  #   init_tick - the tick at which the projectile was spawned
  #   lifetime_ticks - how many ticks this projectile will survive
  #   """
  #   self.size = size
  #   self.position = init_pos
  #   self.velocity = velocity
  #   self.acceleration = np.zeros(2)
  #   self.init_tick = init_tick
  #   self.lifetime_ticks = lifetime_ticks
  #   self.active = True


  def _apply_velocity(self, target_velocity: np.ndarray):
    """
    Applies velocity to projectile
    """

    self.velocity = target_velocity

  
  def _compute_velocity(self, current_tick: int, delta_time: float, targets: list):
    """
    Implement this method to control the velocity the projectile has at every moment.
    """
    pass

  def _compute_acceleration(self, current_tick: int, delta_time: float, targets: list):
    """
    Implement this method to compute the projectile's acceleration at every moment
    """
    pass

  
  def process(self, current_tick: int, delta_time: float, targets: list):
    """
    Processes the state of the projectile each tick.
    """
    if (self.lifetime_ticks) > 0 and (current_tick - self.init_tick > self.lifetime_ticks):
      self.active = False
      return
    
    self._compute_acceleration(current_tick, delta_time, targets)
    self._compute_velocity(current_tick, delta_time, targets)
    
    self.velocity += self.acceleration * delta_time
    self.position += self.velocity * delta_time
  
  def get_position(self) -> np.ndarray:
    """
    Returns the position of the projectile
    """
    return self.position
  
  def get_velocity(self) -> np.ndarray:
    """
    Returns the velocity of the projectile
    """
    return self.velocity

  
  def is_in_collision_with(self, agent) -> bool:
    """
    Returns true if the projectile is in collision with the provided agent, false otherwise
    """

    center_distance = np.linalg.norm(agent.get_position() - self.position)

    return center_distance < self.size + agent.get_size()

  
  def is_active(self) -> bool:
    """
    Returns true if the projectile is active, false otherwise.
    """
    return self.active

  def to_summary(self) -> dict:
    """
    Returns the summary of a projectile.
    """

    return {
      'position': list(self.position),
      'velocity': list(self.velocity),
      'init_tick': self.init_tick,
      'lifetime_ticks': self.lifetime_ticks
    }


  


    


