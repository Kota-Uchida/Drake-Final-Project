import os
from dataclasses import dataclass
import numpy as np

from controller import (
    JointStiffnessOptimizationController,
    OptimizeTrajectory,
    SimpleController,
)
from manipulation.station import (
    LoadScenario,
)
from perception import (
    AimingSystem,
    BallTrajectoryEstimator,
    PointCloudAggregatorSystem,
    add_single_depth_cameras,
)
from pydrake.all import (
    CameraInfo,
    ConstantVectorSource,
    DiagramBuilder,
    PixelType,
    Simulator,
    StartMeshcat,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    RigidTransform,
    Rotation,
    RollPitchYaw,
)
from pydrake.manipulation import SchunkWsgPositionController
from pydrake.perception import BaseField, DepthImageToPointCloud
from utils import RenderDiagramPNG, ThrowObjectTowardsTarget


@dataclass
class SimulationConfigs:
    scenario_dir: str = "../../../scenarios"
    scenario_name: str = "test.yaml"
    model_path: str = "../../../assets/package.xml"
    simulation_time: float = 6.0
    q0 = np.array([0, 0.1, 0, -1.2, 0, 0, 0])
    simple_controller_gains: dict = {"k_p": 90.0, "k_i": 500.0, "k_d": 2000000.0}
    simple_wsg_controller_gains: dict = {
        "k_p": 100.0,
        "k_d": 5.0,
        "kp_constraint": 1500.0,
        "kd_constraint": 5.0,
        "default_force_limit": 100.0,
    }
    wsg_constant_value: np.ndarray = np.array([0.05])
    single_camera_configs: dict = {
        "width": 4000,
        "height": 3000,
        "camera_position": np.array([1.0, 3.0, 1.0]),
        "lookat_position": np.array([1.0, -10.0, 1.0]),
    }


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
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=1e-4)
        self.parser = Parser(self.plant, self.scene_graph)

        # Add iiwa model
        self.plant = MultibodyPlant(time_step=1e-4)
        self.iiwa = self.parser.AddModelsFromUrl(
            "package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf"
        )[0]
        self.plant.WeldFrames(
            self.plant.world_frame(), 
            self.plant.GetFrameByName("iiwa_link_0", self.iiwa)
        )
        
        # Add wsg model
        self.wsg = self.parser.AddModelsFromUrl(
            "package://manipulation/hydro/schunk_wsg_50_with_tip.sdf"
        )[0]
        self.plant.WeldFrames(
            self.plant.GetFrameByName("iiwa_link_7"),
            self.plant.GetFrameByName("body", self.wsg),
            X_FM=RigidTransform(
                RollPitchYaw(90, 0, 90).ToRotationMatrix(),
                [0, 0, 0.09]
            )
        )

        # Add baseball model
        self.baseball = self.parser.AddModels(
            "../../../assets/models/baseball/baseball.sdf"
        )

        # Add bat model
        self.bat = self.parser.AddModels(
            "../../../assets/models/bat/bat.sdf"
        )
        self.plant.WeldFrames(
            self.plant.GetFrameByName("body", self.wsg),
            self.plant.GetFrameByName("bat_link", self.bat),
            X_FM=RigidTransform(
                Rotation.Identity(),
                [0, 0.045, 0]
            )
        )

        # Add table model
        self.table = self.parser.AddModels(
            "../../../assets/models/table/table.sdf"
        )
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("table_link", self.table),
            X_FM=RigidTransform(
                Rotation.Identity(),
                [0.0, 0.0, -0.05]
            )
        )

        self.plant.Finalize()

        self.diagram = None
        self.simulator = None
        self.context_diagram = None
        self.context_plant = None

        self.iiwa_source = None
        self.wsg_source = None
        self.iiwa_plant = None
        self.wsg_plant = None
        self.wsg = None

        self.controller = None

    def _add_systems(self):
        self.builder.AddSystem(self.station)

        # Controller for iiwa
        q0 = self.cfg.q0
        gains = self.cfg.simple_controller_gains
        self.controller = SimpleController(
            k_p=gains["k_p"], k_i=gains["k_i"], k_d=gains["k_d"], q_desired=q0
        )
        self.builder.AddSystem(self.controller)

        wsg_gains = self.cfg.simple_wsg_controller_gains
        self.wsg_position_controller = SchunkWsgPositionController(
            kp_command=wsg_gains["k_p"],
            kd_command=wsg_gains["k_d"],
            kp_constraint=wsg_gains["kp_constraint"],
            kd_constraint=wsg_gains["kd_constraint"],
            default_force_limit=wsg_gains["default_force_limit"],
        )
        self.builder.AddSystem(self.wsg_position_controller)

        self.wsg_desired_source = ConstantVectorSource(self.cfg.wsg_constant_value)

        self.builder.AddSystem(self.wsg_desired_source)

        single_camera_cfg = self.cfg.single_camera_configs
        self.cameras, self.camera_transforms, camera_config = add_single_depth_cameras(
            self.builder,
            self.station,
            self.plant,
            self.scene_graph,
            self.meshcat,
            single_camera_cfg["width"],
            single_camera_cfg["height"],
            single_camera_cfg["camera_position"],
            single_camera_cfg["lookat_position"],
        )

        self.depth_to_point_cloud_systems = []
        for i, camera in enumerate(self.cameras):
            depth_to_point_cloud_system = self.builder.AddSystem(
                DepthImageToPointCloud(
                    camera_info=CameraInfo(
                        camera_config.width,
                        camera_config.height,
                        camera_config.focal_x(),
                        camera_config.focal_y(),
                        camera_config.width / 2,
                        camera_config.height / 2,
                    ),
                    pixel_type=PixelType.kDepth32F,
                    scale=1.0,
                    fields=BaseField.kXYZs,
                )
            )
            depth_to_point_cloud_system.set_name(f"depth_to_point_cloud_{i}")
            self.depth_to_point_cloud_systems.append(depth_to_point_cloud_system)

        aggregator = PointCloudAggregatorSystem(
            self.depth_to_point_cloud_systems, self.builder
        )
        self.point_cloud_aggregator = self.builder.AddSystem(aggregator)
        aggregator.ExportOutput()
        self.ball_trajectory_estimator = self.builder.AddSystem(
            BallTrajectoryEstimator(meshcat=self.meshcat)
        )
        self.trajectory_optimizer = self.builder.AddSystem(OptimizeTrajectory())
        self.joint_stiffness_controller = self.builder.AddSystem(
            JointStiffnessOptimizationController(plant=self.plant)
        )
        self.aiming_system = self.builder.AddSystem(
            AimingSystem(
                self.plant,
                self.plant.GetFrameByName("iiwa_link_7"),
                desired_direction=0.0,
                end_effector_offset_range=(0.5, 0.7),
                aiming_distance_range=(0.5, 1.5),
                meshcat=self.meshcat,
            )
        )
        # self.point_cloud_system = PointCloudSystem(self.cameras, self.camera_transforms, camera_config, self.builder, self.station, self.meshcat)
        # self.builder.AddSystem(self.point_cloud_system)

    def _connect_systems(self):
        self.builder.Connect(
            self.station.GetOutputPort("iiwa_state"), self.controller.get_input_port(0)
        )
        self.builder.Connect(
            self.controller.get_output_port(0),
            self.station.GetInputPort("iiwa_actuation"),
        )
        self.builder.Connect(
            self.station.GetOutputPort("wsg_state"),
            self.wsg_position_controller.get_state_input_port(),
        )
        self.builder.Connect(
            self.wsg_desired_source.get_output_port(),
            self.wsg_position_controller.get_desired_position_input_port(),
        )
        self.builder.Connect(
            self.wsg_position_controller.get_generalized_force_output_port(),
            self.station.GetInputPort("wsg_actuation"),
        )
        for i, camera in enumerate(self.cameras):
            self.builder.Connect(
                camera.GetOutputPort("depth_image_32f"),
                self.depth_to_point_cloud_systems[i].get_input_port(0),
            )
            self.builder.Connect(
                camera.GetOutputPort("body_pose_in_world"),
                self.depth_to_point_cloud_systems[i].get_input_port(2),
            )
            self.builder.Connect(
                self.depth_to_point_cloud_systems[i].get_output_port(0),
                self.point_cloud_aggregator.get_input_port(i),
            )

        self.builder.Connect(
            self.point_cloud_aggregator.get_output_port(0),
            self.ball_trajectory_estimator.get_input_port(0),
        )

        self.builder.Connect(
            self.station.GetOutputPort("iiwa_state"),
            self.joint_stiffness_controller.get_input_port(0),
        )

        self.builder.Connect(
            self.trajectory_optimizer.get_output_port(2),
            self.joint_stiffness_controller.get_input_port(1),
        )
        self.builder.Connect(
            self.ball_trajectory_estimator.get_output_port(0),
            self.aiming_system.get_input_port(0),
        )
        self.builder.Connect(
            self.station.GetOutputPort("iiwa_state"),
            self.aiming_system.get_input_port(1),
        )
        self.builder.Connect(
            self.aiming_system.get_output_port(0),
            self.trajectory_optimizer.get_input_port(0),
        )
        self.builder.Connect(
            self.aiming_system.get_output_port(1),
            self.trajectory_optimizer.get_input_port(1),
        )

        # self.builder.Connect(
        #     self.joint_stiffness_controller.get_output_port(0),
        #     self.station.GetInputPort("iiwa_actuation")
        # )

        self.diagram = self.builder.Build()

    def _configure_contexts(self):
        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_plant = self.plant.GetMyMutableContextFromRoot(
            self.context_diagram
        )

        ThrowObjectTowardsTarget(
            plant=self.plant,
            plant_context=self.context_plant,
            object_name="baseball_link",
            initial_position=np.array([0.0, -10.0, 0.5]),
            target_position=np.array([1.3, 0.0, 1.0]),
            target_speed_xy=7.0,
        )

        # set the initial state of wsg
        wsg_model_instance = self.plant.GetModelInstanceByName("wsg")
        wsg_joint = self.plant.GetJointByName(
            "left_finger_sliding_joint", wsg_model_instance
        )
        wsg_joint.set_translation(self.context_plant, 0.05)
        wsg_joint = self.plant.GetJointByName(
            "right_finger_sliding_joint", wsg_model_instance
        )
        wsg_joint.set_translation(self.context_plant, -0.05)

    def _output_diagram(self):
        RenderDiagramPNG(self.diagram, max_depth=1)

    def _setup_simulation(self):
        self.simulator = Simulator(self.diagram, self.context_diagram)
        self.ctx = self.simulator.get_mutable_context()

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
    cfg = SimulationConfigs()
    sim_master = SimulationMaster(cfg)
    sim_master.execute_simulation()
