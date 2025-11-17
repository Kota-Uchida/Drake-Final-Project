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
from controller import SimpleController, WSGController

import hydra
from omegaconf import OmegaConf


class SimulationMaster:
    def __init__(self, scenario_path: str = None, model_path: str = None):
        print("                                                            ")
        print("####################   Starting Meshcat   ##################")
        self.meshcat = StartMeshcat()
        print("############################################################")
        print("                                                            ")
        self.builder = DiagramBuilder()
        self.scenario_name = "test.yaml"
        self.scenario_path = scenario_path
        self.scenario = LoadScenario(filename=self.scenario_path)
        self.station = MakeHardwareStation(scenario=self.scenario, meshcat=self.meshcat, parser_preload_callback=lambda p: p.package_map().AddPackageXml(model_path))
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")
        self.plant = self.station.GetSubsystemByName("plant")
        self.props = ProximityProperties()
        AddContactMaterial(dissipation=0.1, point_stiffness=1e3, properties=self.props)

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

        q0 = np.array([0, 0.1, 0, -1.2, 0, 0, 0])
        self.controller = SimpleController(k_p=90.0, k_i=500.0, k_d=1000000.0, q_desired=q0)
        self.builder.AddSystem(self.controller)

        # Replaced with SchunkWsgPositionController
        # wsg_desired = np.array([0.0, -0.0])
        # self.wsg_controller = WSGController(position_desired=wsg_desired, k_p=10.0, k_i = 600.0, k_d=100.0, max_torque=6.5)
        # self.builder.AddSystem(self.wsg_controller)

        self.wsg_position_controller = SchunkWsgPositionController(kp_command=100.0, kd_command=5.0, kp_constraint=1500.0, kd_constraint=5.0, default_force_limit=100.0)
        self.builder.AddSystem(self.wsg_position_controller)

        self.wsg_desired_source = ConstantVectorSource(np.array([0.0]))

        self.builder.AddSystem(self.wsg_desired_source)

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
        #RenderDiagramPNG(self.diagram, max_depth=1)
        pass

    def _setup_simulation(self):
        self.simulator = Simulator(self.diagram, self.context_diagram)
        self.simulator.set_target_realtime_rate(1.0)
        self.meshcat.StartRecording()
        self.simulator.AdvanceTo(4.0)
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
    scenario_path = os.path.join('../../../scenarios', "test.yaml")
    model_path = os.path.join('../../../assets', "package.xml")
    sim_master = SimulationMaster(scenario_path=scenario_path, model_path=model_path)
    sim_master.execute_simulation()


