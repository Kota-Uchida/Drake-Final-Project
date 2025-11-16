import os
import time
import numpy as np


from pydrake.all import (DiagramBuilder, InverseDynamicsController,
                         ModelInstanceIndex, MultibodyPlant, RigidTransform,
                         RotationMatrix, Simulator, StartMeshcat)
from pydrake.multibody.parsing import Parser
from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
    RobotDiagram
)
from manipulation.utils import RenderDiagram


import hydra
from omegaconf import OmegaConf


class SimulationMaster:
    def __init__(self, scenario_path: str = None, model_path: str = None):
        self.meshcat = StartMeshcat()
        self.builder = DiagramBuilder()
        self.plant = MultibodyPlant(time_step=0.001)
        self.scenario_name = "test.yaml"
        self.scenario_path = scenario_path
        self.scenario = LoadScenario(filename=self.scenario_path)
        self.station = MakeHardwareStation(scenario=self.scenario, meshcat=self.meshcat, parser_preload_callback=lambda p: p.package_map().AddPackageXml(model_path))
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")
        self.plant = self.station.GetSubsystemByName("plant")
        self.diagram = None
        self.simulator = None

        self.context_diagram = None
        self.context_plant = None


    def _add_systems(self):
        self.builder.AddSystem(self.station)

    def _connect_systems(self):
        # Connect systems as needed
        self.diagram = self.builder.Build()

    def _configure_contexts(self):
        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_plant = self.plant.GetMyMutableContextFromRoot(self.context_diagram)
        q0 = np.array([0.0, 0.0, 0.0, -1.2, 0.0, 1.57, 0.0, 0.0, 0.0])
        print("Number of positions:", self.plant.num_positions())
        self.plant.SetPositions(self.context_plant,q0)
        self.plant.SetVelocities(self.context_plant, np.zeros(self.plant.num_velocities()))
        print("Number of objects in plant:", self.plant.num_model_instances())
        

    def _output_diagram(self):
        # Placeholder for outputting diagram

        pass

    def _setup_simulation(self):
        self.simulator = Simulator(self.diagram, self.context_diagram)
        self.simulator.set_target_realtime_rate(1.0)
        self.meshcat.StartRecording()
        self.simulator.AdvanceTo(10.0)
        self.meshcat.PublishRecording()
    
    def _save_results(self):
        # Placeholder for saving results
        pass

    def execute_simulation(self):
        self._add_systems()
        self._connect_systems()
        self._configure_contexts()
        self._output_diagram()
        self._setup_simulation()
        self._save_results()




if __name__ == "__main__":
    scenario_path = os.path.join('../../../scenarios', "test3.yaml")
    model_path = os.path.join('../../../assets', "package.xml")
    sim_master = SimulationMaster(scenario_path=scenario_path, model_path=model_path)
    sim_master.execute_simulation()


