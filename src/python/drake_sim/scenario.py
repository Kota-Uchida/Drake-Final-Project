import yaml
from abc import ABC, abstractmethod
import os
import sys




class ScenarioWriter:
    def __init__(self, scenario_path: str):
        self.scenario_path = scenario_path
    
    def write_scenario(self, scenario_data: dict):
        # Write the scenario data to a YAML file
        pass

class ScenarioComponent(ABC):
    @abstractmethod
    def add_objects(self):
        pass

    @abstractmethod
    def configure_objects(self):
        # Optional method to configure objects
        pass

    @abstractmethod
    def read_sdf(self, sdf_path: str):
        pass


