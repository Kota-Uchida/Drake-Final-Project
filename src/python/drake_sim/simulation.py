import os
import time
import numpy as np


from pydrake.all import (DiagramBuilder, InverseDynamicsController,
                         ModelInstanceIndex, MultibodyPlant, RigidTransform,
                         RotationMatrix, Simulator, StartMeshcat, ConstantVectorSource,
                         JointStiffnessController)
from pydrake.multibody.parsing import Parser
from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
    RobotDiagram
)
from pydrake.geometry import ProximityProperties, AddContactMaterial, Sphere
from pydrake.manipulation import SchunkWsgPositionController

from utils import RenderDiagramPNG, ThrowObjectTowardsTarget
from controller import SimpleController
from perception import add_depth_cameras, PointCloudSystem

from dataclasses import dataclass
import yaml

@dataclass
class SimulationConfigs:
    scenario_dir: str
    scenario_name: str
    model_path: str
    simulation_time: float


class SimulationMaster:
    def __init__(self, cfg: SimulationConfigs):
        print("                                                            ")
        print("####################   Starting Meshcat   ##################")
        self.meshcat = StartMeshcat()
        print("############################################################")
        print("                                                            ")
        
        self.cfg = cfg
        self.builder = DiagramBuilder()
        self.scenario_path = os.path.join(cfg.scenario_dir, cfg.scenario_name)
        self.scenario = LoadScenario(filename=self.scenario_path)
        self.station = MakeHardwareStation(scenario=self.scenario, meshcat=self.meshcat, parser_preload_callback=lambda p: p.package_map().AddPackageXml(cfg.model_path))
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")
        self.plant = self.station.GetSubsystemByName("plant")

        self.diagram = None
        self.simulator = None
        self.context_diagram = None
        self.context_plant = None

        self.iiwa_source = None
        self.wsg_source = None
        self.iiwa_plant = None
        self.wsg_plant = None
        self.iiwa = None
        self.wsg = None

        self.controller = None


    def _add_systems(self):
        self.builder.AddSystem(self.station)

        # Controller for iiwa
        q0 = np.array([0, 0.1, 0, -1.2, 0, 0, 0])
        self.controller = SimpleController(k_p=90.0, k_i=500.0, k_d=2000000.0, q_desired=q0)
        self.builder.AddSystem(self.controller)

        # Position controller for WSG
        # TODO: Replace this with force controller
        self.wsg_position_controller = SchunkWsgPositionController(kp_command=100.0, kd_command=5.0, kp_constraint=1500.0, kd_constraint=5.0, default_force_limit=100.0)
        self.builder.AddSystem(self.wsg_position_controller)

        self.wsg_desired_source = ConstantVectorSource(np.array([0.0]))

        self.builder.AddSystem(self.wsg_desired_source)

        self.cameras, camera_transforms,camera_config =add_depth_cameras(self.builder, self.station, self.plant, self.scene_graph , 800, 600, 4, 5, 7.0, [1.0 , 0.0, 1.5])

        self.point_cloud_system = PointCloudSystem(self.cameras, camera_transforms, camera_config, self.builder, self.station, self.meshcat)

        self.builder.AddSystem(self.point_cloud_system)

    def _connect_systems(self):
        # Connect systems as needed
        self.builder.Connect(
            self.station.GetOutputPort("iiwa_state"),
            self.controller.get_input_port(0)
        )
        self.builder.Connect(
            self.controller.get_output_port(0),
            self.station.GetInputPort("iiwa_actuation")
        )
        self.builder.Connect(
            self.station.GetOutputPort("wsg_state"),
            self.wsg_position_controller.get_state_input_port()
        )
        self.builder.Connect(
            self.wsg_desired_source.get_output_port(),
            self.wsg_position_controller.get_desired_position_input_port()
        )
        self.builder.Connect(
            self.wsg_position_controller.get_generalized_force_output_port(),
            self.station.GetInputPort("wsg_actuation")
        )
        self.point_cloud_system.ConnectCameras(self.station, self.builder, self.cameras)

        self.diagram = self.builder.Build()

    def _configure_contexts(self):
        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_plant = self.plant.GetMyMutableContextFromRoot(self.context_diagram)
        
     
        ThrowObjectTowardsTarget(
            plant=self.plant,
            plant_context=self.context_plant,
            object_name="baseball_link",
            initial_position=np.array([0.0, -10.0, 0.5]),
            target_position=np.array([1.035, 0.2, 1.3]), # before change: [0.675, 0.2, 1.3]
            target_speed_xy=7.0,
        )
        # set the initial state of wsg
        wsg_model_instance = self.plant.GetModelInstanceByName("wsg")
        wsg_joint = self.plant.GetJointByName("left_finger_sliding_joint", wsg_model_instance)
        wsg_joint.set_translation(self.context_plant, 0.05)
        wsg_joint = self.plant.GetJointByName("right_finger_sliding_joint", wsg_model_instance)
        wsg_joint.set_translation(self.context_plant, -0.05)

        

    def _output_diagram(self):
        # Uncomment to visualize the system diagram    
        RenderDiagramPNG(self.diagram, max_depth=1)
        pass

    def _setup_simulation(self):
        self.simulator = Simulator(self.diagram, self.context_diagram)
        self.simulator.set_target_realtime_rate(1.0)
        self.meshcat.StartRecording()
        self.simulator.AdvanceTo(self.cfg.simulation_time)
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
    config_path = os.path.join("../../../conf", "config.yaml")
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    cfg = SimulationConfigs(**raw["simulation"])
    sim_master = SimulationMaster(cfg)
    sim_master.execute_simulation()


