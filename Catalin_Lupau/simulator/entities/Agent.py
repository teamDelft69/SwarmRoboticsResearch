
from typing import Any, List

import numpy as np

from entities.MapStructure import MapStructure
from entities.Obstacle import Obstacle
from entities.TargetArea import TargetArea


class AgentPerception:
  """
  Contains all of the objects that the agent is able to perceive.
  """

  def __init__(self, obstacles: list, swarm_agents: list, target_area: TargetArea, projectiles: list, map_structure: MapStructure) -> None:
    """
    Initializes an object storing the agents perception.
    """
    self.obstacles: list = obstacles
    self.swarm_agents: list = swarm_agents
    self.target_area: TargetArea = target_area
    self.projectiles: list = projectiles
    self.map_structure: MapStructure = map_structure


  
  def get_obstacles(self) -> List[Obstacle]:
    return self.obstacles
  
  def get_swarm_agents(self) -> list:
    return self.swarm_agents
  
  def get_target_area(self) -> TargetArea:
    return self.target_area
  
  def get_projectiles(self) -> List[Any]:
    return self.projectiles

  def get_map_structure(self) -> MapStructure:
    return self.map_structure



class AgentMessage:
  """
  Encodes an message to be sent from an agent to another agent.
  """

  def __init__(self, id: str, sender_id: str, content: str='') -> None:
    """
    Initializes and agent message.
    """
    self.id = id
    self.sender_id = sender_id
    self.content = content

  def get_id(self):
    """
    Returns the id of the message
    """
    return self.id
  
  def get_sender_id(self):
    """
    Returns the id of the sender.
    """
    return self.sender_id

  def get_content(self):
    """
    Returns the conent of the agent message.
    """
    return self.content


class Agent:
  """
  Abstract class implementing a generic swarm agent, without any logic associated to it.
  """

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    """
    Initializes the agent with the initial position, velocity, size, and acc_limit
    """
    self.position: np.ndarray = init_position
    self.velocity: np.ndarray = init_velocity
    self.acceleration: np.ndarray = 0
    self.size: float = size
    self.acc_limit: float = acc_limit
    self.id = id
    self.perception_distance: float = perception_distance
    self.swarm_distance = swarm_distance
    
    self.received_messages: List[str] = []

  def is_in_collision_with(self, agent) -> bool:
    """
    Returns true if the provided agent is in collision with the current agent, false otherwise
    """
    return np.linalg.norm(self.position - agent.position) < self.size + agent.size


  def _on_message_received(self, message: AgentMessage, current_tick: int, delta_time: float, perception: AgentPerception):
    """
    Processes the received message.
    """
    pass

  def process_message(self, message: AgentMessage):
    """
    Call this function to make the agent process a new message
    """
    self.received_messages.append(message)


  def process(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Processes the agent
    """

    # process the received messages
    if len(self.received_messages) > 0:
      self._on_message_received(self.received_messages[0], current_tick, delta_time, agent_perception)
      del self.received_messages[0]

    # compute the velocity based on the implemented algorithm
    self._compute_velocity(current_tick, delta_time, agent_perception)
    self._compute_acceleration(current_tick, delta_time, agent_perception)

    self.velocity += self.acceleration * delta_time
    self.position += self.velocity * delta_time




  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Abstract method intended to compute the instant velocity of the swarm agent.
    It needs to be overriden in order to implement a strategy for the agents.
    """

    pass

  def _compute_acceleration(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Abstract method used to compute the instant acceleration of the swarm agent.
    It needs to be overriden in order to
    """

    pass

  
  def get_position(self) -> np.ndarray:
    return self.position
  
  def get_velocity(self) -> np.ndarray:
    return self.velocity
  

  def get_size(self) -> np.ndarray:
    return self.size

  def get_id(self) -> np.ndarray:
    return self.id

  def get_perception_distance(self) -> float:
    return self.perception_distance


  def _apply_velocity(self, target_velocity: np.ndarray) -> None:
    """
    Applies a new velocity to the swarm agent as long as the acceleration difference is smaller than the limit.
    """
    diff: float = np.linalg.norm(self.velocity - target_velocity)

    if diff <= self.acc_limit:
      self.velocity = target_velocity
    else:
      raise Exception(f'Cannot apply velocity. Acceleration too high! Limit: {self.acc_limit}, Actual: {diff}')

  
  def __repr__(self) -> str:
      return f'Agent(position: {list(self.position)}, velocity: {list(self.velocity)}, size: {self.size}, acc limit: {self.acc_limit}, id: {self.id})'
  
  def to_summary(self) -> dict:
    """
    Returns a dictionary representation summerizing the relevant data that defines the agent at this given moment.
    """

    return {
      'position': list(self.position),
      'velocity': list(self.velocity),
      'size': self.size,
      'acc limit': self.acc_limit,
      'perception_distance': self.perception_distance,
      'swarm_distance': self.swarm_distance,
      'id': self.id
    }






    
