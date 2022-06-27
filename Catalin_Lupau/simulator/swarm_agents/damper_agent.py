import math
from enum import Enum
from typing import List

import numpy as np
from entities.Agent import Agent, AgentPerception
from navigation_planners.global_planners.AStarPlanner import AStarPlanner
from navigation_planners.global_planners.GlobalPlanner import GlobalPlanner
from navigation_planners.local_planners.BasicPlanner import BasicPlanner


class DamperAgentState(Enum):
  ORGANIZING = 1,
  PLANNING_PATH = 2,
  MOVING_TO_TARGET = 3,
  HITTING_TARGET = 4,
  FOLLOWING_LEADER = 5,


class DamperAgent(Agent):

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
      super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)


      self.attack_target_mode = False
      self.base_distance = 8.0
      self.obstacle_distance = self.size
      self.k = 1.0
      self.k_obstacle = 1.0
      self.c = 2.0
      self.mass = 10

      self.agent_state = DamperAgentState.ORGANIZING

      self.fixed_neighbors = []

      self.basic_nav_planner: BasicPlanner = None
  



  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Computes the instant velocity of the swarm agent.
    """


    self.velocity = self._compute_acceleration_in_travel_mode(
        current_tick, delta_time, agent_perception)


    if self.agent_state == DamperAgentState.ORGANIZING:

      if np.linalg.norm(self.velocity) < 0.001:
        self.agent_state = DamperAgentState.PLANNING_PATH if self.id == 'agent_1' else DamperAgentState.FOLLOWING_LEADER
        self.fixed_neighbors = self._get_neighbors(self.id, len(agent_perception.get_swarm_agents()), agent_perception)
    
    elif self.agent_state == DamperAgentState.PLANNING_PATH:
      global_planner = AStarPlanner(
            agent_perception.get_obstacles(), agent_perception.get_map_structure(), 1.0, 2 * self.size)

      path = global_planner.find_path(self.position, agent_perception.get_target_area().get_position())
      self.basic_nav_planner = BasicPlanner(path, 1.0)
      self.agent_state = DamperAgentState.MOVING_TO_TARGET

    elif self.agent_state == DamperAgentState.MOVING_TO_TARGET:
      self.velocity += 1.0 * self.basic_nav_planner.plan_movement_direction(self.position)

    # if self.id == 'agent_1':
    #   self.velocity += np.array([-1, 0])

    

  def _compute_acceleration(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Computes the instant acceleration of the swarm agent.
    """

    # if not self.attack_target_mode:
    #   self.acceleration = self._compute_acceleration_in_travel_mode(current_tick, delta_time, agent_perception)

    self.acceleration = np.zeros(2)

    # for obstacle in agent_perception.get_obstacles():
    #   self.acceleration += self._compute_obstacle_force(self.position, obstacle.get_position())
    

    # if self.id == 'agent_1':
    #   self.acceleration = np.array([-1, 0])

  
  def _compute_obstacle_force(self, position: np.ndarray, obstacle_position: np.ndarray) -> float:
    """
    Computes the force corresponding to the provided obstacle
    """

    distance = np.linalg.norm(position - obstacle_position)

    if distance >= self.obstacle_distance:
      return 0.0
    
    return (position - obstacle_position) * self.k_obstacle * (self.obstacle_distance - distance) / np.linalg.norm(position - obstacle_position)
    
    

  def _compute_acceleration_in_travel_mode(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Computes the acceleration in travel mode.
    """

    neighbors_ids = self._get_neighbors(self.id, len(agent_perception.swarm_agents), agent_perception)

    if self.agent_state == DamperAgentState.FOLLOWING_LEADER or self.agent_state == DamperAgentState.MOVING_TO_TARGET:
      neighbors_ids = self.fixed_neighbors

   

    force = np.zeros(2)

    for neighbor_id in neighbors_ids:
      force += self._compute_force(self.id, neighbor_id, agent_perception.swarm_agents)

    

    for obstacle in agent_perception.get_obstacles():
      force  += self._compute_obstacle_force(self.position,
                                            obstacle.get_position())


    # only apply this to the first and the last agent
    

    return force / self.mass

  
  def _compute_force(self, agent1_id: str, agent2_id: str, agent_list: List[Agent], diagonal = False) -> np.ndarray:
    """
    Computes the force that acts on agent 1 and corresponds with the interactions with agent 2
    """

    agent1: Agent = self._retrieve_agent_based_on_id(agent_list, agent1_id)
    agent2: Agent = self._retrieve_agent_based_on_id(agent_list, agent2_id)

    if (agent1 == None) or (agent2 == None):
      return np.zeros(2)

    displacement = agent2.position - agent1.position

    # print(agent2.position)
    # print(displacement)
    # print(np.linalg.norm(displacement))
    direction = displacement / np.linalg.norm(displacement)

    velocity_1_projection = np.dot(agent1.velocity, direction) / np.linalg.norm(direction)
    velocity_2_projection = np.dot(agent2.velocity, direction) / np.linalg.norm(direction)

    if diagonal:
      elongation = np.linalg.norm(displacement) - \
          self.base_distance * 1.732050808
    else:
      elongation = np.linalg.norm(displacement) - self.base_distance
    

    elongation_speed = np.linalg.norm(velocity_2_projection - velocity_1_projection)

    force_value = self.k * elongation + self.c * elongation_speed

    # print(force_value)


    return force_value * direction


  def _get_neighbors(self, agent_id: str, num_agents: int, agent_perception: AgentPerception):
    """
    Return the neighbors of the provided agent.
    """
    swarm_agents = agent_perception.swarm_agents

    agent: Agent = self._retrieve_agent_based_on_id(swarm_agents, agent_id)

    neighbors = []

    for other_agent in swarm_agents:
      if other_agent.get_id() == agent.get_id():
        continue

      distance = np.linalg.norm(agent.get_position() - other_agent.get_position())

      if distance < self.base_distance * 1.2:
        neighbors.append(other_agent.get_id())
    

    return neighbors


    # id_num = int(agent_id.split('_')[1])

    # neighbor_id_nums = [id_num - 2, id_num - 1, id_num + 1, id_num + 2]
    # neighbor_id_nums_diag = [id_num - 3, id_num + 3]
    # neighbor_id_nums = list(filter(lambda x: x > 0 and x <= num_agents, neighbor_id_nums))
    # neighbor_id_nums_diag = list(
    #     filter(lambda x: x > 0 and x <= num_agents, neighbor_id_nums_diag))

    # neighbor_ids = list(map(lambda x: f'agent_{x}', neighbor_id_nums))
    # neighbor_ids_diag = list(
    #     map(lambda x: f'agent_{x}', neighbor_id_nums_diag))

    # return neighbor_ids, neighbor_ids_diag

  
  # def _compute_distance_between_agents(self, agent_id1: str, agent_id2: str, agent_perception: AgentPerception):
  #     """
  #     Computes the distance between the 2 provided agents.
  #     """

  #     swarm_agents: List[Agent] = agent_perception.get_swarm_agents()

  #     agent1 = self._retrieve_agent_based_on_id(swarm_agents, agent_id1)
  #     agent2 = self._retrieve_agent_based_on_id(swarm_agents, agent_id2)

  #     return np.linalg.norm(agent1.position - agent2.position)

  
  def _retrieve_agent_based_on_id(self, swarm_agents: List[Agent], id: str):
    """
    Retrieves the agent based on the provided id.

    If no agents was found, raise exception.
    """

    for agent in  swarm_agents:
      if agent.id == id:
        return agent
    

    return None



