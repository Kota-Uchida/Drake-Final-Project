from pydrake.all import (
    AbstractValue,
    Trajectory,
    PointCloud,
    DiagramBuilder,
    RgbdSensor,
    CameraConfig,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    LeafSystem,
    Context,
    InputPort,
    OutputPort,
    ImageDepth32F,
    ImageRgba8U,
    ImageLabel16I,
    Meshcat,
    PiecewisePolynomial,
    Diagram,
    MultibodyPlant,
    ConstantValueSource,
    MakeRenderEngineGl,
    Rgba,
    CameraInfo,
    PixelType,
    SceneGraph,
    GeometryFrame,
    GeometryInstance,
)

from pydrake.multibody.parsing import Parser
from typing import Optional, Tuple, List
import numpy.typing as npt
import itertools

from pydrake.perception import DepthImageToPointCloud

import matplotlib.pyplot as plt
import numpy as np

"""""""""""
add_cameras function cited from
https://github.com/barci2/6.4210-Robotic-Manipulation/blob/main/src/perception.py
with some minor modifications.
"""""""""""
group_idx = 0
def add_depth_cameras(
        builder: DiagramBuilder,
        station: Diagram,
        plant: MultibodyPlant,
        scene_graph: SceneGraph,
        camera_width: int,
        camera_height: int,
        horizontal_num: int,
        vertical_num: int,
        camera_distance: float,
        cameras_center: npt.NDArray[np.float32]
    ) -> Tuple[List[RgbdSensor], List[RigidTransform], List[ConstantValueSource], CameraConfig]:
    """
    Adds multiple cameras in a hemisphere arrangement around a given center point.
    Args:
        builder: DiagramBuilder to which the cameras will be added.
        station: HardwareStation containing the scene graph.
        plant: MultibodyPlant associated with the station.
        camera_width: Width of each camera image.
        camera_height: Height of each camera image.
        horizontal_num: Number of cameras in the horizontal direction.
        vertical_num: Number of cameras in the vertical direction.
        camera_distance: Distance from the center point to each camera.
        cameras_center: 3D coordinates of the center point around which cameras are arranged.
    Returns:
        A tuple containing:
            - A list of RgbdSensor instances representing the added cameras.
            - A list of RigidTransform instances representing the transforms of each camera.
    """
    global group_idx
    
    # Camera configuration
    camera_config = CameraConfig()
    camera_config.z_far=30
    camera_config.width = camera_width
    camera_config.height = camera_height
    scene_graph = station.GetSubsystemByName("scene_graph")
    if not scene_graph.HasRenderer(camera_config.renderer_name):
        scene_graph.AddRenderer(
            camera_config.renderer_name, MakeRenderEngineGl())
    
    camera_systems = []
    camera_transforms = []
    thetas = np.linspace(0, 2*np.pi, horizontal_num, endpoint=False)
    phis = np.linspace(0, -np.pi/2, vertical_num + 1)[1:]  # Exclude the top point

    for idx, (theta, phi) in enumerate(itertools.product(thetas, phis)):
        name = f"camera{idx}_group{group_idx}"
        transform = RigidTransform(RollPitchYaw(0, 0, theta).ToRotationMatrix() @ RollPitchYaw(phi, 0, 0).ToRotationMatrix(), cameras_center) @ RigidTransform([0, 0, -camera_distance])
        _, depth_camera = camera_config.MakeCameras()
        camera = builder.AddSystem(
            RgbdSensor(
                parent_id=plant.GetBodyFrameIdIfExists(plant.world_frame().body().index()),
                X_PB=transform,
                depth_camera=depth_camera
            )
        )

        parser = Parser(plant=plant)
        model = parser.AddModels("package://assets/camera_box/camera_box.sdf")
        plant.SetFreeBodyPose(
            plant.GetBodyByName("camera_box", model),
            transform
        )
        # scene_id = scene_graph.RegisterSource(f"{name}_source")
        # geometry_frame = GeometryFrame(f"{name}_frame", frame_group_id=1)
        # geometry_instance = GeometryInstance(transform, None, f"{name}_geometry")

        # scene_graph.RegisterGeometry(
        #     scene_id,
        #     geometry_frame,
        #     geometry_instance
        # )

        builder.Connect(
            station.GetOutputPort("query_object"), camera.query_object_input_port()
        )

        # TODO: These exports are just for debugging. Remove them later.
        builder.ExportOutput(
            camera.depth_image_32F_output_port(), f"{name}.depth_image"
        )
        builder.ExportOutput(
            camera.label_image_output_port(), f"{name}.label_image"
        )
        camera.set_name(name)
        camera_systems.append(camera)
        camera_transforms.append(transform)
    group_idx += 1
    return camera_systems, camera_transforms, camera_config


class PointCloudSystem(LeafSystem):
    def __init__(
            self, 
            cameras: list[RgbdSensor], 
            camera_transforms: list[RigidTransform], 
            camera_config: CameraConfig,
            builder: DiagramBuilder, 
            station: Diagram,
            meshcat: Optional[Meshcat] = None
        ):
        super().__init__()
        self.cameras = cameras
        self.camera_transforms = camera_transforms
        self.depth_ports= [camera.depth_image_32F_output_port() for camera in cameras]
        self.label_ports= [camera.label_image_output_port() for camera in cameras]
        self.rgb_ports = [camera.color_image_output_port() for camera in cameras]
        self.meshcat = meshcat
        self.point_clouds_of_each_camera = [PointCloud() for _ in cameras]
        self.point_cloud_processed = PointCloud()

        center_x = camera_config.width / 2.0
        center_y = camera_config.height / 2.0
        self.camera_info = CameraInfo(camera_config.width, camera_config.height, camera_config.focal_x(), camera_config.focal_y(), center_x, center_y)
        self.point_cloud_systems = [
            DepthImageToPointCloud(self.camera_info, pixel_type=PixelType.kDepth32F)
            for _ in cameras
        ]
        for _ in self.point_cloud_systems:
            builder.AddSystem(_)

        # self._camera_rgb_inputs = [
        #     self.DeclareAbstractInputPort(
        #         f"camera_{i}_rgb",
        #         AbstractValue.Make(ImageRgba8U(self.camera_info.width(), self.camera_info.height()))
        #     )
        #     for i in range(len(cameras))
        # ]

        # self._camera_depth_inputs = [
        #     self.DeclareAbstractInputPort(
        #         f"camera_{i}_depth",
        #         AbstractValue.Make(ImageDepth32F(self.camera_info.width(), self.camera_info.height()))
        #     )
        #     for i in range(len(cameras))
        # ]

        # self._camera_label_inputs = [
        #     self.DeclareAbstractInputPort(
        #         f"camera_{i}_label",
        #         AbstractValue.Make(ImageLabel16I(self.camera_info.width(), self.camera_info.height()))
        #     )
        #     for i in range(len(cameras))
        # ]

        self._point_cloud_inputs = [
            self.DeclareAbstractInputPort(
                f"camera_{i}_point_cloud",
                AbstractValue.Make(PointCloud())
            )
            for i in range(len(cameras))
        ]

        self._point_cloud_output = self.DeclareAbstractOutputPort(
            "point_cloud",
            lambda: AbstractValue.Make(PointCloud()),
            self._calc_point_cloud_output
        )
        
    def ConnectCameras(self, station: Diagram, builder: DiagramBuilder, cameras: List[RgbdSensor]):
        for camera, point_cloud_system in zip(cameras, self.point_cloud_systems):
            builder.Connect(
                camera.depth_image_32F_output_port(),
                point_cloud_system.depth_image_input_port()
            )
            builder.Connect(
                camera.color_image_output_port(),
                point_cloud_system.color_image_input_port(),
            )
            builder.Connect(
                camera.body_pose_in_world_output_port(),
                point_cloud_system.camera_pose_input_port()
            )
            builder.Connect(
                point_cloud_system.point_cloud_output_port(),
                self._point_cloud_inputs[cameras.index(camera)]
            )


    def _calc_point_cloud_output(self, context: Context, output: AbstractValue):
        # First, add all point clouds from each camera
        pcs = []
        total = 0
        # for point_cloud_system in self.point_cloud_systems:
        #     pc = point_cloud_system.point_cloud_output_port().Eval(context)
        for i, _ in enumerate(self.point_cloud_systems):
            pc = self._point_cloud_inputs[i].Eval(context)
            # Transform point cloud to world frame
            #pc.mutable_xyzs()[:] = self.camera_transforms.rotation() @ pc.xyzs() + self.camera_transforms.translation().reshape(3, 1)
            pcs.append(pc)
            total += pc.size()
        # Aggregate point clouds
        out_pc = PointCloud(total)
        
        for pc in pcs:
            out_pc.mutable_xyzs()[:, out_pc.size() - pc.size(): out_pc.size()] = pc.xyzs()
            if pc.has_rgbs():
                out_pc.mutable_rgbs()[:, out_pc.size() - pc.size(): out_pc.size()] = pc.rgbs()

        if self.meshcat is not None:
            self.meshcat.SetObject(f"{str(self)}PointCloud", out_pc, point_size=0.01, rgba=Rgba(1, 0.5, 0.5))

        output.set_value(out_pc)






    

