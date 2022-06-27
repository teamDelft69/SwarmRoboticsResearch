class MapStructure:
  """
  Encodes the structure of the overall map of the environment.
  This data structure defines the size and shapes of the safe area and the danger area.
  """

  def __init__(self, map_height, safe_area_width, danger_area_width) -> None:
    """
    map_height - the overall height of the map.
    safe_area_width - the overall width of the map
    danger_area_width - the overall width of the danger area
    """
    self.map_height = map_height
    self.safe_area_width = safe_area_width
    self.danger_area_width = danger_area_width

  
  def get_map_height(self):
    return self.map_height
  

  def get_safe_area_width(self):
    return self.safe_area_width
  
  def get_danger_area_width(self):
    return self.danger_area_width

  def to_summary(self) -> dict:
    """
    Returns a summary of the map structure as a python dictionary
    """

    return {
      'map_height': self.map_height,
      'safe_area_width': self.safe_area_width,
      'danger_area_width': self.danger_area_width
    }
