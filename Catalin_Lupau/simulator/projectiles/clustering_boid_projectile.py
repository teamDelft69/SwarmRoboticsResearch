
from re import A

import numpy as np
from clustering.k_means import k_means_clustering
from entities.Agent import Agent, AgentPerception

from projectiles.boid_projectile import BoidsProjectile


class ClusteringBoidsProjectile(BoidsProjectile):
  """
  Strategy used against split swarm avoidance strategies.
  Uses a fast clustering algorithm to detect different swarm clusters.
  """ 

  def __init__(self, init_position: np.ndarray, init_velocity: np.ndarray, size: float, acc_limit: float, id: str, perception_distance: float, swarm_distance: float) -> None:
    super().__init__(init_position, init_velocity, size, acc_limit, id, perception_distance, swarm_distance)

    self.max_clusters = 5

    self.centers = np.array([[0.0, 0.0]])

  def _compute_clusters(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the centers of the cluster in each iteration.
    """
    swarm_positions = list(
        map(lambda a: a.get_position(), agent_perception.get_swarm_agents()))
    centers = k_means_clustering(swarm_positions, self.max_clusters)

    return centers

  def _get_assigned_cluster(self, id: str):
    """
    Return the cluster corresponding to this agent.
    """
    id_number = int(id.split('_')[1])

    return id_number % self.centers.shape[0]



  def _compute_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    
    # compute the centers
    self.centers = self._compute_clusters(current_tick, delta_time, agent_perception)

    return super()._compute_velocity(current_tick, delta_time, agent_perception)

  
  
  def _compute_steering_velocity(self, current_tick: int, delta_time: float, agent_perception: AgentPerception):
    """
    Compute the steering velocity of the boids
    """
    

    assigned_cluster = self._get_assigned_cluster(self.id)

    target_position = self.centers[assigned_cluster]


    return (target_position - self.position) * self.steer_factor
  
  def _get_neighbors(self, agent_id: str, agent_perception: AgentPerception):
    """
    Return the neighbors of the provided agent.
    """
    projectile_agents = agent_perception.get_projectiles()

    agent: Agent = self._retrieve_agent_based_on_id(
        projectile_agents, agent_id)

    neighbors = []

    cluster_id = self._get_assigned_cluster(self.id)

    for other_agent in projectile_agents:
      if other_agent.get_id() == agent.get_id():
        continue
        
      if cluster_id != self._get_assigned_cluster(other_agent.get_id()):
        continue
      
     

      distance = np.linalg.norm(
          agent.get_position() - other_agent.get_position())

      if distance <= self.perception_distance:
        neighbors.append(other_agent)

    return neighbors


