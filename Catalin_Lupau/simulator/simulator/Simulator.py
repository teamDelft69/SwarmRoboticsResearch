from simulator.Environment import Environment
from simulator.SimulatorConfiguration import SimulatorConfiguration


class Simulator:
  """
  Class encoding the problem simulator.
  """

  def __init__(self, configuration: SimulatorConfiguration) -> None:
    """
    Initializes a simulator using the provided configuration.
    """
    
    self.configuration = configuration

    self.last_result = {}


  def run(self):
    """
    Generator function that runs the simulation, yielding the environment each tick.
    """

    # build the initial environment
    env: Environment = self.configuration.build_environment()

    yield env.to_summary()

    # extract the max ticks
    max_ticks = self.configuration.get_max_ticks()

    # run the simulation for the provided number of ticks
    for tick in range(max_ticks):
      env.process(tick)
      yield env.to_summary()

      if env.is_simulation_finished():
        break
    
    self.last_result = env.get_results()
  

  def get_last_result(self):
    """
    Return the last result of the simulation.
    If the simulation wasn't run at least once, it returns an empty object.
    """

    return self.last_result
  



