import os
from dataclasses import dataclass

import numpy as np
import yaml
from controller import (
    JointStiffnessOptimizationController,
    OptimizeTrajectory,
    SimpleController,
)
from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
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
)
from pydrake.manipulation import SchunkWsgPositionController
from pydrake.perception import BaseField, DepthImageToPointCloud
from utils import RenderDiagramPNG, ThrowObjectTowardsTarget


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

        def prefinalize_callback(p):
            pass

        self.station = MakeHardwareStation(
            scenario=self.scenario,
            meshcat=self.meshcat,
            parser_preload_callback=lambda p: p.package_map().AddPackageXml(
                cfg.model_path
            ),
            parser_prefinalize_callback=prefinalize_callback,
        )
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")
        self.plant = self.station.GetSubsystemByName("plant")
        self.iiwa_idx = self.plant.GetModelInstanceByName("iiwa")

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
        self.controller = SimpleController(
            k_p=90.0, k_i=500.0, k_d=2000000.0, q_desired=q0
        )
        self.builder.AddSystem(self.controller)

        # Position controller for WSG
        # TODO: Replace this with force controller
        self.wsg_position_controller = SchunkWsgPositionController(
            kp_command=100.0,
            kd_command=5.0,
            kp_constraint=1500.0,
            kd_constraint=5.0,
            default_force_limit=100.0,
        )
        self.builder.AddSystem(self.wsg_position_controller)

        self.wsg_desired_source = ConstantVectorSource(np.array([0.0]))

        self.builder.AddSystem(self.wsg_desired_source)

        # self.cameras, self.camera_transforms,camera_config = add_depth_cameras(self.builder, self.station, self.plant, self.scene_graph, self.meshcat, 800, 600, 4, 5, 4.0, [1.0 , 0.0, 0.3])
        self.cameras, self.camera_transforms, camera_config = add_single_depth_cameras(
            self.builder,
            self.station,
            self.plant,
            self.scene_graph,
            self.meshcat,
            4000,
            3000,
            np.array([1.0, 3.0, 1.0]),
            np.array([1.0, -10.0, 1.0]),
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
        self.trajectory_optimizer = self.builder.AddSystem(OptimizeTrajectory(self.plant, self.iiwa_idx, "iiwa_link_7", [0, 0, 0], horizon=2.0, N_ctrl=8, num_limit_samples=10, W_a=1.0, W_r=1e-2, W_n=1.0, max_sqp_iters=5))
        self.joint_stiffness_controller = self.builder.AddSystem(
            JointStiffnessOptimizationController(self.plant)
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
        # Connect systems as needed
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
        # self.point_cloud_system.ConnectCameras(self.station, self.builder, self.cameras)
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
            self.ball_trajectory_estimator.get_output_port(1),
            self.aiming_system.get_input_port(0),
        )
        self.builder.Connect(
            self.station.GetOutputPort("iiwa_state"),
            self.aiming_system.get_input_port(1),
        )
        self.builder.Connect(
            self.station.GetOutputPort("iiwa_state"),
            self.trajectory_optimizer.get_input_port(0),
        )
        self.builder.Connect(
            self.aiming_system.get_output_port(0),
            self.trajectory_optimizer.get_input_port(1),
        )
        self.builder.Connect(
            self.aiming_system.get_output_port(1),
            self.trajectory_optimizer.get_input_port(2),
        )
        self.builder.Connect(
            self.aiming_system.get_output_port(2),
            self.trajectory_optimizer.get_input_port(3),
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
        # Uncomment to visualize the system diagram
        RenderDiagramPNG(self.diagram, max_depth=1)

    def _setup_simulation(self):
        self.simulator = Simulator(self.diagram, self.context_diagram)
        self.ctx = self.simulator.get_mutable_context()

        self.simulator.set_target_realtime_rate(1.0)
        self.meshcat.StartRecording()
        # If you just want to do simulation without camera outputs, uncomment the following line
        self.simulator.AdvanceTo(self.cfg.simulation_time)

        # t_sample = np.arange(0, self.cfg.simulation_time, 0.1)
        # for i, t in enumerate(t_sample):
        #     self.simulator.AdvanceTo(t)
        #     self.diagram.ForcedPublish(self.ctx)
        #     pc = self.diagram.GetOutputPort("point_cloud").Eval(self.ctx)
        #     down_sampled_pc = pc.VoxelizedDownSample(0.01)
        # debug: visualize down-sampled point cloud
        # self.meshcat.SetObject(f"{str(self)}PointCloud", down_sampled_pc, point_size=0.01, rgba=Rgba(1, 0, 0))
        # if i % 10 == 0:
        #     image = self.diagram.GetOutputPort("camera0_group0.rgb_image").Eval(self.ctx)
        #     labeled = self.diagram.GetOutputPort("camera0_group0.label_image").Eval(self.ctx)
        #     depth = self.diagram.GetOutputPort("camera0_group0.depth_image").Eval(self.ctx)
        #     image_name = f"../../../output_images/rgb_image_{i:03d}.png"
        #     labeled_name = f"../../../output_images/label_image_{i:03d}.png"
        #     depth_name = f"../../../output_images/depth_image_{i:03d}.png"
        #     from PIL import Image
        #     # Camera produces RGBA (4 channels); reshape accordingly
        #     rgba_data = image.data.reshape((image.height(), image.width(), 4))
        #     # Convert RGBA to RGB by dropping the alpha channel
        #     rgb_data = rgba_data[:, :, :3]
        #     img = Image.fromarray(rgb_data, 'RGB')
        #     img.save(image_name)
        #     # labeled image: convert to uint8 numpy array explicitly to avoid deprecation warning
        #     labeled_array = labeled.data.reshape((labeled.height(), labeled.width())).astype(np.uint8)
        #     img = Image.fromarray(labeled_array, mode='L')
        #     img.save(labeled_name)
        #     # depth image: convert float32 to uint16 (scale to 0-65535 range) for PNG support
        #     depth_array = depth.data.reshape((depth.height(), depth.width()))
        #     # Scale depth: assume max depth ~10m, map to 16-bit range (adjust scaling as needed)
        #     depth_uint16 = (np.clip(depth_array, 0, 10.0) * 6553.5).astype(np.uint16)
        #     img = Image.fromarray(depth_uint16, mode='I;16')
        #     img.save(depth_name)

        # read point cloud
        # pc = self.diagram.GetOutputPort("point_cloud").Eval(self.ctx)

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
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = SimulationConfigs(**raw["simulation"])
    sim_master = SimulationMaster(cfg)
    sim_master.execute_simulation()
