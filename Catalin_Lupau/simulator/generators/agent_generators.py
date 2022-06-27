import numpy as np
from entities.Agent import Agent
from swarm_agents.a_star_boids_agent import AStarBoidsAgent
from swarm_agents.boids_agent import BoidsAgent
from swarm_agents.boids_no_obstacle_avoidance import BoidsNoObstacleAvoidance
from swarm_agents.damper_agent import DamperAgent
from swarm_agents.directed_whirlwind_repulsive_boids import \
    DirectedWhirlwindRepulsiveBoids
from swarm_agents.evasive_kitchensink import EvasiveKitchensinkBoids
from swarm_agents.evasive_steer_away_boids import EvasiveSteerAwayBoids
from swarm_agents.explode_evasive_kitchensink import \
    ExplosiveEvasiveKitchensinkBoids
from swarm_agents.explode_kitchensink import ExplosiveKitchenSinkBoids
from swarm_agents.explosive_steer_away_boids import ExplosiveSteerAwayBoids
from swarm_agents.jump_kitchensink import JumpingKitchensinkBoids
from swarm_agents.jumping_steer_away_boids import JumpingSteerAwayBoids
from swarm_agents.neural_network_boids import BoidNet, NeuralNetworkBoids
from swarm_agents.split_evasive_kitchensink import SplitEvasiveKitchensinkBoids
from swarm_agents.split_kitchensink import SplitKitchensinkBoids
from swarm_agents.split_steer_away_boids import SplitSteerAwayBoids
from swarm_agents.steer_away_boids import SteerAwayBoids
from swarm_agents.steer_away_directed_whirlwind_repulsive_boids import \
    SteerAwayDirectedWhirlwindRepulsiveBoids
from swarm_agents.whirlwind_repulsive_boids import WhirlwindRepulsiveBoids

"""
Contains agent generator functions
"""


from entities.MapStructure import MapStructure


def basic_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width() + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(min_pos_x, max_pos_x)
    y_pos = np.random.uniform(min_pos_y, max_pos_y)

    agents.append(Agent(np.array([x_pos, y_pos]), np.zeros(2), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def damper_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    agents.append(DamperAgent(np.array([x_pos, y_pos]), np.zeros(
        2), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def basic_with_velocity_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  padding = 2 * agent_size

  min_pos_y = 0 + padding
  max_pos_y = map_structure.get_map_height() - padding
  min_pos_x = map_structure.get_danger_area_width() + padding
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width() - padding

  for i in range(num_agents):
    x_pos = np.random.uniform(min_pos_x, max_pos_x)
    y_pos = np.random.uniform(min_pos_y, max_pos_y)

    agents.append(Agent(np.array([x_pos, y_pos]), np.array([-1.0, 0.0]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(BoidsAgent(np.array([x_pos, y_pos]), np.array([x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def a_star_boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(AStarBoidsAgent(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def steer_away_boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(SteerAwayBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def jumping_steer_away_boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(JumpingSteerAwayBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def evasive_steer_away_boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(EvasiveSteerAwayBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def explosive_steer_away_boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(ExplosiveSteerAwayBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def split_steer_away_boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(SplitSteerAwayBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def whirlwind_boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(WhirlwindRepulsiveBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def directed_whirlwind_boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(DirectedWhirlwindRepulsiveBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def steer_away_directed_whirlwind_boids_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(SteerAwayDirectedWhirlwindRepulsiveBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def boids_no_avoidance_agent_generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(BoidsNoObstacleAvoidance(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def neural_network_parameterized_boid_generator(boid_net: BoidNet):
  def generator(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
    agents = []
    min_pos_y = 0
    max_pos_y = map_structure.get_map_height()
    min_pos_x = map_structure.get_danger_area_width()
    max_pos_x = map_structure.get_danger_area_width(
    ) + map_structure.get_safe_area_width()

    for i in range(num_agents):
      x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
      y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

      x_vel = np.random.uniform()
      y_vel = np.random.uniform()

    

      agents.append(NeuralNetworkBoids(boid_net, np.array([x_pos, y_pos]), np.array(
          [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

    return agents
  

  return generator


def evasive_kitchensink(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(EvasiveKitchensinkBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def explode_kitchensink(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(ExplosiveKitchenSinkBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def explode_evasive_kitchensink(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(ExplosiveEvasiveKitchensinkBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def jump_kitchensink(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(JumpingKitchensinkBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def split_kitchensink(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(SplitKitchensinkBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents


def split_evasive_kitchensink(swarm_distance: float, map_structure: MapStructure, agent_size: float, agent_acc_limit: float, agent_perception_distance: float, num_agents: int):
  agents = []

  min_pos_y = 0
  max_pos_y = map_structure.get_map_height()
  min_pos_x = map_structure.get_danger_area_width()
  max_pos_x = map_structure.get_danger_area_width(
  ) + map_structure.get_safe_area_width()

  for i in range(num_agents):
    x_pos = np.random.uniform(max_pos_x - 12, max_pos_x - 20)
    y_pos = np.random.uniform(max_pos_y - 12, max_pos_y - 20)

    x_vel = np.random.uniform()
    y_vel = np.random.uniform()

    agents.append(SplitEvasiveKitchensinkBoids(np.array([x_pos, y_pos]), np.array(
        [x_vel, y_vel]), agent_size, agent_acc_limit, f'agent_{i + 1}', agent_perception_distance, swarm_distance))

  return agents



def retrieve_agent_generator(generator_name: str, boid_net: BoidNet = None):
  """
  Retrieves the right generator based on the provided generator name
  """

  if generator_name == 'basic':
    return basic_agent_generator
  
  if generator_name == 'evasive_kitchensink':
    return evasive_kitchensink
  
  if generator_name == 'jump_kitchensink':
    return jump_kitchensink
  
  if generator_name == 'explode_kitchensink':
    return explode_kitchensink
  
  if generator_name == 'explode_evasive_kitchensink':
    return explode_evasive_kitchensink
  
  if generator_name == 'split_kitchensink':
    return split_kitchensink

  if generator_name == 'split_evasive_kitchensink':
    return split_evasive_kitchensink

  if generator_name == 'basic_vel':
    return basic_with_velocity_agent_generator
  
  if generator_name == 'damper':
    return damper_agent_generator
  
  if generator_name == 'boids':
    return boids_agent_generator
  
  if generator_name == 'boids_no_avoidance':
    return boids_no_avoidance_agent_generator
  
  if generator_name == 'a_star_boids':
    return a_star_boids_agent_generator
  
  if generator_name == 'steer_away_boids':
    return steer_away_boids_agent_generator
  
  if generator_name == 'jumping_steer_away_boids':
    return jumping_steer_away_boids_agent_generator

  if generator_name == 'evasive_steer_away_boids':
    return evasive_steer_away_boids_agent_generator

  if generator_name == 'explosive_steer_away_boids':
    return explosive_steer_away_boids_agent_generator
  
  if generator_name == 'split_steer_away_boids':
    return split_steer_away_boids_agent_generator
  
  if generator_name == 'whirlwind_boids':
    return whirlwind_boids_agent_generator

  if generator_name == 'directed_whirlwind_boids':
    return directed_whirlwind_boids_agent_generator
  
  if generator_name == 'param_neural_net_boids':
    return neural_network_parameterized_boid_generator(boid_net)
  
  if generator_name == 'steer_away_directed_whirlwind_boids':
    return steer_away_directed_whirlwind_boids_agent_generator
  
  if generator_name.startswith('neural_network_boids'):
    _, model_name = generator_name.split('~')
    network = BoidNet.load_from_file(model_name)
    return neural_network_parameterized_boid_generator(network)



  return lambda x, y: []
