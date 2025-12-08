import itertools
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.all import (
    AbstractValue,
    BasicVector,
    CameraConfig,
    CameraInfo,
    Concatenate,
    Context,
    Diagram,
    DiagramBuilder,
    Frame,
    ImageDepth32F,
    ImageLabel16I,
    ImageRgba8U,
    InverseKinematics,
    LeafSystem,
    MakeRenderEngineGl,
    Meshcat,
    MultibodyPlant,
    PiecewisePolynomial,
    PixelType,
    PointCloud,
    Rgba,
    RgbdSensor,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SceneGraph,
    SpatialVelocity,
)
from pydrake.geometry import Sphere, Cylinder
from pydrake.perception import BaseField, DepthImageToPointCloud, Fields
from pydrake.solvers import Solve
from sklearn.linear_model import RANSACRegressor


def calc_camera_poses(
    horizontal_num: int,
    vertical_num: int,
    camera_distance: float,
    cameras_center: npt.NDArray[np.float32],
) -> List[RigidTransform]:
    """
    Calculate camera poses in a hemisphere arrangement around a given center point.
    Args:
        horizontal_num: Number of cameras in the horizontal direction.
        vertical_num: Number of cameras in the vertical direction.
        camera_distance: Distance from the center point to each camera.
        cameras_center: 3D coordinates of the center point around which cameras are arranged.
    Returns:
        A list of RigidTransform instances representing the transforms of each camera.
    """
    camera_transforms = []
    thetas = np.linspace(0, 2 * np.pi, horizontal_num, endpoint=False)
    phis = np.linspace(np.pi / 2, np.pi, vertical_num + 1)[:-1]  # Exclude the top point
    for theta, phi in itertools.product(thetas, phis):
        transform = RigidTransform(
            RollPitchYaw(0, 0, theta).ToRotationMatrix()
            @ RollPitchYaw(phi, 0, 0).ToRotationMatrix(),
            cameras_center,
        ) @ RigidTransform([0, 0, -camera_distance])
        camera_transforms.append(transform)

    return camera_transforms


"""""" """""
add_cameras function cited from
https://github.com/barci2/6.4210-Robotic-Manipulation/blob/main/src/perception.py
with some minor modifications.
""" """""" ""
group_idx = 0


def add_depth_cameras(
    builder: DiagramBuilder,
    station: Diagram,
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    meshcat: Meshcat,
    camera_width: int,
    camera_height: int,
    horizontal_num: int,
    vertical_num: int,
    camera_distance: float,
    cameras_center: npt.NDArray[np.float32],
) -> Tuple[List[RgbdSensor], List[RigidTransform], CameraConfig]:
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
    camera_config.z_far = 20
    camera_config.width = camera_width
    camera_config.height = camera_height
    scene_graph = station.GetSubsystemByName("scene_graph")
    if not scene_graph.HasRenderer(camera_config.renderer_name):
        scene_graph.AddRenderer(camera_config.renderer_name, MakeRenderEngineGl())

    camera_systems = []
    camera_transforms = []
    thetas = np.linspace(0, 2 * np.pi, horizontal_num, endpoint=False)
    phis = np.linspace(np.pi / 2, np.pi, vertical_num + 1)[:-1]  # Exclude the top point

    for idx, (theta, phi) in enumerate(itertools.product(thetas, phis)):
        name = f"camera{idx}_group{group_idx}"
        transform = RigidTransform(
            RollPitchYaw(0, 0, theta).ToRotationMatrix()
            @ RollPitchYaw(phi, 0, 0).ToRotationMatrix(),
            cameras_center,
        ) @ RigidTransform([0, 0, -camera_distance])
        # print(f"Adding camera: {name} at translation: {transform.translation()}")
        _, depth_camera = camera_config.MakeCameras()
        camera = builder.AddSystem(
            RgbdSensor(
                parent_id=plant.GetBodyFrameIdIfExists(
                    plant.world_frame().body().index()
                ),
                X_PB=transform,
                depth_camera=depth_camera,
            )
        )
        AddMeshcatTriad(
            meshcat=meshcat,
            path=f"{name}_triad",
            length=0.2,
            radius=0.002,
            opacity=0.8,
            X_PT=transform,
        )

        # plant.SetFreeBodyPose(

        #     plant.GetBodyByName("base", camera_model_instance),
        #     transform
        # )
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

        builder.ExportOutput(camera.color_image_output_port(), f"{name}.rgb_image")
        builder.ExportOutput(
            camera.depth_image_32F_output_port(), f"{name}.depth_image"
        )
        builder.ExportOutput(camera.label_image_output_port(), f"{name}.label_image")
        camera.set_name(name)
        camera_systems.append(camera)
        camera_transforms.append(transform)
    group_idx += 1
    return camera_systems, camera_transforms, camera_config


def look_at_transform(camera_position, target_position, world_up=np.array([0, 0, 1.0])):
    """
    Create a Drake RigidTransform that makes the camera's +Z axis
    point toward target_position from camera_position.

    Args:
        camera_position: (3,) array
        target_position: (3,) array
        world_up: preferred up vector (default = [0,0,1])

    Returns:
        RigidTransform
    """
    camera_position = np.asarray(camera_position).reshape(3)
    target_position = np.asarray(target_position).reshape(3)
    world_up = np.asarray(world_up).reshape(3)

    # Forward direction (+Z of the camera)
    d = target_position - camera_position
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-8:
        raise ValueError("Camera position and target position are too close.")
    d = d / d_norm

    # Avoid singularity: if d is almost parallel to world_up
    if abs(np.dot(d, world_up)) > 0.99:
        world_up = np.array([0, 1, 0])

    # Right vector (+X)
    x = np.cross(world_up, d)
    x /= np.linalg.norm(x)

    # True up vector (+Y)
    y = np.cross(d, x)
    y /= np.linalg.norm(y)

    # +Z = d
    z = d

    # Assemble rotation matrix
    R = RotationMatrix(np.column_stack((x, y, z)))

    return RigidTransform(R, camera_position)


def add_single_depth_cameras(
    builder: DiagramBuilder,
    station: Diagram,
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    meshcat: Meshcat,
    camera_width: int,
    camera_height: int,
    camera_position: npt.NDArray[np.float32],
    cameras_center: npt.NDArray[np.float32],
) -> Tuple[List[RgbdSensor], List[RigidTransform], CameraConfig]:
    """
    Adds a single cameras in a hemisphere arrangement around a given center point.
    Args:
        builder: DiagramBuilder to which the cameras will be added.
        station: HardwareStation containing the scene graph.
        plant: MultibodyPlant associated with the station.
        camera_width: Width of each camera image.
        camera_height: Height of each camera image.
        camera_position: 3D coordinates of the camera position.
        cameras_center: 3D coordinates of the center point around which cameras are arranged.
    Returns:
        A tuple containing:
            - A list of RgbdSensor instances representing the added cameras.
            - A list of RigidTransform instances representing the transforms of each camera.
    """
    global group_idx

    # Camera configuration
    camera_config = CameraConfig()
    camera_config.width = camera_width
    camera_config.height = camera_height
    camera_config.fps = 60
    camera_config.z_far = 30.0

    scene_graph = station.GetSubsystemByName("scene_graph")
    if not scene_graph.HasRenderer(camera_config.renderer_name):
        scene_graph.AddRenderer(camera_config.renderer_name, MakeRenderEngineGl())

    camera_systems = []
    camera_transforms = []

    transform = look_at_transform(camera_position, cameras_center)

    camera = builder.AddSystem(
        RgbdSensor(
            parent_id=plant.GetBodyFrameIdIfExists(plant.world_frame().body().index()),
            X_PB=transform,
            depth_camera=camera_config.MakeCameras()[1],
        )
    )
    name = f"camera0_group{group_idx}"
    AddMeshcatTriad(
        meshcat=meshcat,
        path=f"{name}_triad",
        length=0.2,
        radius=0.002,
        opacity=0.8,
        X_PT=transform,
    )
    builder.Connect(
        station.GetOutputPort("query_object"), camera.query_object_input_port()
    )

    builder.ExportOutput(camera.color_image_output_port(), f"{name}.rgb_image")
    builder.ExportOutput(camera.depth_image_32F_output_port(), f"{name}.depth_image")
    builder.ExportOutput(camera.label_image_output_port(), f"{name}.label_image")
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
        meshcat: Optional[Meshcat] = None,
    ):
        super().__init__()
        print("b")
        self.cameras = cameras
        self.camera_transforms = camera_transforms
        self.depth_ports = [camera.depth_image_32F_output_port() for camera in cameras]
        self.label_ports = [camera.label_image_output_port() for camera in cameras]
        self.rgb_ports = [camera.color_image_output_port() for camera in cameras]
        self.meshcat = meshcat
        self.point_clouds_of_each_camera = [PointCloud() for _ in cameras]
        self.point_cloud_processed = PointCloud()
        print("c")

        center_x = camera_config.width / 2.0
        center_y = camera_config.height / 2.0
        self.camera_info = CameraInfo(
            camera_config.width,
            camera_config.height,
            camera_config.focal_x(),
            camera_config.focal_y(),
            center_x,
            center_y,
        )
        self.point_cloud_systems = [
            DepthImageToPointCloud(self.camera_info, pixel_type=PixelType.kDepth32F)
            for _ in cameras
        ]
        for _ in self.point_cloud_systems:
            builder.AddSystem(_)
        print("d")

        self._camera_rgb_inputs = [
            self.DeclareAbstractInputPort(
                f"camera_{i}_rgb",
                AbstractValue.Make(
                    ImageRgba8U(self.camera_info.width(), self.camera_info.height())
                ),
            )
            for i in range(len(cameras))
        ]

        self._camera_depth_inputs = [
            self.DeclareAbstractInputPort(
                f"camera_{i}_depth",
                AbstractValue.Make(
                    ImageDepth32F(self.camera_info.width(), self.camera_info.height())
                ),
            )
            for i in range(len(cameras))
        ]

        self._camera_label_inputs = [
            self.DeclareAbstractInputPort(
                f"camera_{i}_label",
                AbstractValue.Make(
                    ImageLabel16I(self.camera_info.width(), self.camera_info.height())
                ),
            )
            for i in range(len(cameras))
        ]

        self._point_cloud_inputs = [
            self.DeclareAbstractInputPort(
                f"camera_{i}_point_cloud",
                AbstractValue.Make(
                    PointCloud(0, fields=Fields(base_fields=BaseField.kRGBs))
                ),
            )
            for i in range(len(cameras))
        ]

        self._point_cloud_output = self.DeclareAbstractOutputPort(
            "point_cloud",
            lambda: AbstractValue.Make(PointCloud()),
            self._calc_point_cloud_output,
        )

    def ConnectCameras(
        self, station: Diagram, builder: DiagramBuilder, cameras: List[RgbdSensor]
    ):
        for camera, point_cloud_system in zip(cameras, self.point_cloud_systems):
            builder.Connect(
                camera.depth_image_32F_output_port(),
                point_cloud_system.depth_image_input_port(),
            )
            builder.Connect(
                camera.color_image_output_port(),
                point_cloud_system.color_image_input_port(),
            )
            builder.Connect(
                camera.body_pose_in_world_output_port(),
                point_cloud_system.camera_pose_input_port(),
            )
            builder.Connect(
                point_cloud_system.point_cloud_output_port(),
                self._point_cloud_inputs[cameras.index(camera)],
            )
            builder.Connect(
                camera.color_image_output_port(),
                self._camera_rgb_inputs[cameras.index(camera)],
            )
            builder.Connect(
                camera.depth_image_32F_output_port(),
                self._camera_depth_inputs[cameras.index(camera)],
            )
            builder.Connect(
                camera.label_image_output_port(),
                self._camera_label_inputs[cameras.index(camera)],
            )
        builder.ExportOutput(self._point_cloud_output, "point_cloud")

    def _calc_point_cloud_output(self, context: Context, output: AbstractValue):
        # First, add all point clouds from each camera
        print("a")
        pcs = []
        total = 0

        # for point_cloud_system in self.point_cloud_systems:
        #     pc = point_cloud_system.point_cloud_output_port().Eval(context)
        for i, _ in enumerate(self.point_cloud_systems):
            print(f"Evaluating point cloud input for camera {i}")
            pc = self.GetInputPort(f"camera_{i}_point_cloud").Eval(context)
            # pc = self._point_cloud_inputs[i].Eval(context)
            print(f"Evaluated point cloud from camera {i}")
            # Transform point cloud to world frame
            # pc.mutable_xyzs()[:] = self.camera_transforms.rotation() @ pc.xyzs() + self.camera_transforms.translation().reshape(3, 1)
            pcs.append(pc)
            total += pc.size()
            print(f"Camera {i} point cloud size: {pc.size()}")

            # Use default concatenate function to aggregate point clouds
        out_pc = Concatenate(pcs)
        down_sampled_pc = out_pc.VoxelizedDownSample(0.005)
        # debug: visualize down-sampled point cloud
        # self.meshcat.SetObject(f"{str(self)}PointCloud", down_sampled_pc, point_size=0.05, rgba=Rgba(1, 0, 0))

        # Aggregate point clouds
        # out_pc = PointCloud(total)

        # for pc in pcs:
        #     out_pc.mutable_xyzs()[:, out_pc.size() - pc.size(): out_pc.size()] = pc.xyzs()
        #     if pc.has_rgbs():
        #         out_pc.mutable_rgbs()[:, out_pc.size() - pc.size(): out_pc.size()] = pc.rgbs()
        # This will be only implemented when the output is read from other systems
        # self.meshcat.SetObject(f"{str(self)}PointCloud", out_pc, point_size=0.05, rgba=Rgba(1, 0, 0))

        output.set_value(down_sampled_pc)


class PointCloudAggregatorSystem(LeafSystem):
    def __init__(
        self,
        point_cloud_systems: List[DepthImageToPointCloud],
        builder: Optional[DiagramBuilder] = None,
    ):
        super().__init__()
        self.point_cloud_systems = point_cloud_systems
        self._point_cloud_output = self.DeclareAbstractOutputPort(
            "aggregated_point_cloud",
            lambda: AbstractValue.Make(
                PointCloud(fields=Fields(base_fields=BaseField.kXYZs))
            ),
            self._calc_aggregated_point_cloud,
        )
        self._point_cloud_inputs = [
            self.DeclareAbstractInputPort(
                f"point_cloud_{i}",
                AbstractValue.Make(
                    PointCloud(fields=Fields(base_fields=BaseField.kXYZs))
                ),
            )
            for i in range(len(point_cloud_systems))
        ]

        self.builder = builder

    def _calc_aggregated_point_cloud(self, context: Context, output: AbstractValue):
        pcs = PointCloud(fields=Fields(base_fields=BaseField.kXYZs))
        for i in range(len(self.point_cloud_systems)):
            pcs = Concatenate(
                [pcs, self.GetInputPort(f"point_cloud_{i}").Eval(context)]
            )
        output.set_value(pcs)

    def ExportOutput(self):
        if self.builder is not None:
            self.builder.ExportOutput(self._point_cloud_output, "point_cloud")


class BallTrajectoryEstimator(LeafSystem):
    """
    Estimates the trajectory of a ball from point cloud data using Recursive Least Squares.
    inputs:
        - point_cloud (PointCloud): PointCloud representing the observed scene
    outputs:
        - ball_trajectory (PiecewisePolynomial): Estimated trajectory of the ball
    """

    def __init__(self, meshcat: Optional[Meshcat] = None):
        super().__init__()
        self._window_size = 50  # Number of recent observations to consider
        self._prediction_horizon = 4.0  # seconds
        self.estimated_trajectory = PiecewisePolynomial()
        self.meshcat: Optional[Meshcat] = meshcat

        self._samples_state = self.DeclareAbstractState(AbstractValue.Make(deque()))
        self._traj_state = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._pred_traj_state = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._fugure_pos_state = self.DeclareAbstractState(
            AbstractValue.Make(np.zeros(3))
        )
        self.DeclareAbstractInputPort(
            "point_cloud",
            AbstractValue.Make(PointCloud(fields=Fields(base_fields=BaseField.kXYZs))),
        )
        self.DeclareAbstractOutputPort(
            "ball_trajectory",
            lambda: AbstractValue.Make(PiecewisePolynomial()),
            self.CalcBallTrajectory,
        )
        self.DeclareAbstractOutputPort(
            "predicted_trajectory",
            lambda: AbstractValue.Make(PiecewisePolynomial()),
            self.OutputPredTrajectory,
        )
        self.DeclarePeriodicPublishEvent(0.05, 0.0, self.UpdateTrajectoryFromPointCloud)

    def UpdateTrajectoryFromPointCloud(self, context: Context) -> None:
        point_cloud = self.GetInputPort("point_cloud").Eval(context)
        ball_points, center = self.clean_point_cloud(point_cloud)

        if center is None:
            print("No ball detected.")
            return
        t = context.get_time()
        center = np.asarray(center).reshape(3)

        samples = context.get_abstract_state(self._samples_state).get_mutable_value()
        samples.append((t, center))
        if len(samples) > self._window_size:
            samples.popleft()
        if len(samples) < 2:
            return

        times = np.array([ti for (ti, _) in samples])
        positions = np.vstack([pi for (_, pi) in samples]).T

        traj = PiecewisePolynomial.FirstOrderHold(times, positions)

        context.SetAbstractState(self._traj_state, AbstractValue.Make(traj))

        coef_x = np.polyfit(times, positions[0, :], 2)
        coef_y = np.polyfit(times, positions[1, :], 2)
        coef_z = np.polyfit(times, positions[2, :], 2)

        pred_times = np.linspace(t, t + self._prediction_horizon, num=200)
        pred_pos = np.vstack(
            [
                np.polyval(coef_x, pred_times),
                np.polyval(coef_y, pred_times),
                np.polyval(coef_z, pred_times),
            ]
        )

        pred_traj = PiecewisePolynomial.FirstOrderHold(pred_times, pred_pos)
        context.SetAbstractState(self._pred_traj_state, AbstractValue.Make(pred_traj))

        # visualize with meshcat
        if self.meshcat is not None:
            self.meshcat.Delete(f"{str(self)}Ball Center")
            self.meshcat.SetObject(
                f"{str(self)}Ball Center", shape=Sphere(0.03), rgba=Rgba(0, 0, 1, 0.5)
            )
            self.meshcat.SetTransform(f"{str(self)}Ball Center", RigidTransform(center))
            for i in range(len(pred_times)):
                self.meshcat.Delete(f"{str(self)}Predicted Traj {i}")
            for i, t in enumerate(pred_times):
                self.meshcat.SetObject(
                    f"{str(self)}Predicted Traj {i}",
                    shape=Sphere(0.03),
                    rgba=Rgba(1, 0, 0, 0.5),
                )
                pos = pred_traj.value(t).flatten()
                self.meshcat.SetTransform(
                    f"{str(self)}Predicted Traj {i}", RigidTransform(pos)
                )

    def CalcBallTrajectory(self, context: Context, output: AbstractValue) -> None:
        traj = context.get_abstract_state(self._traj_state).get_value()
        output.set_value(traj)

    def OutputPredTrajectory(self, context: Context, output: AbstractValue) -> None:
        pred_traj = context.get_abstract_state(self._pred_traj_state).get_value()
        output.set_value(pred_traj)

    def clean_point_cloud(self, point_cloud: PointCloud) -> PointCloud:
        # Extract ball points with RANSAC
        r = 0.01  # radius of the ball
        points = point_cloud.xyzs().T  # Nx3

        # filter out NaN points
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]

        normal, d, inliers_plane = self.ransac_plane(
            points, threshold=0.01, max_iterations=1000
        )
        points_plane_removed = points[~inliers_plane]
        center, inliers_sphere, masked_points = self.ransac_sphere(
            points_plane_removed, r=r, threshold=0.005, max_iterations=1000
        )
        ball_points = masked_points[inliers_sphere]
        return ball_points, center

    def ransac_plane(self, points, threshold=0.01, max_iterations=1000):
        """
        RANSAC to fit plane ax + by + cz + d = 0
        reutrn: normal, d, inliers
        """
        X = points[:, :2]
        y = -points[:, 2]
        ransac = RANSACRegressor(
            min_samples=3, residual_threshold=threshold, max_trials=max_iterations
        )
        ransac.fit(X, y)
        inliers = ransac.inlier_mask_
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        normal = np.array([a, b, -1.0])
        normal /= np.linalg.norm(normal)
        d = c
        return normal, d, inliers

    def ransac_sphere(self, points, r=0.01, threshold=0.005, max_iterations=1000):
        """
        RANSAC to fit sphere (x - x0)^2 + (y - y0)^2 + (z - z0)^2 = r^2
        return: center, inliers
        """
        best_inliers = None
        best_center = None
        max_inliers = 0
        mask = points[:, 1] < -0.5
        points = points[mask]
        if self.meshcat is not None:
            self.meshcat.Delete(f"{str(self)} filtered points for sphere")
            pc_debug = PointCloud(
                points.shape[0], fields=Fields(base_fields=BaseField.kXYZs)
            )
            pc_debug.mutable_xyzs()[:] = points.T
            self.meshcat.SetObject(
                f"{str(self)} filtered points for sphere",
                pc_debug,
                point_size=0.02,
                rgba=Rgba(1, 0, 0, 0.5),
            )
        N = points.shape[0]

        for _ in range(max_iterations):
            # --- Sample 3 random points ---
            try:
                idx = np.random.choice(N, 3, replace=False)
            except ValueError:
                continue
            p1, p2, p3 = points[idx]

            # --- Compute candidate center ---
            # Intersection of 3 spheres with radius r
            # Better: solve center as least squares from these 3 points
            A = 2 * np.vstack(
                [
                    p2 - p1,
                    p3 - p1,
                ]
            )
            b = np.vstack(
                [
                    (np.linalg.norm(p2) ** 2 - np.linalg.norm(p1) ** 2) - (r**2 - r**2),
                    (np.linalg.norm(p3) ** 2 - np.linalg.norm(p1) ** 2) - (r**2 - r**2),
                ]
            )

            # Solve A * c = b
            try:
                center = np.linalg.lstsq(A, b, rcond=None)[0].flatten()
            except:
                continue

            # --- Compute residual errors ---
            d = np.linalg.norm(points - center, axis=1)
            residuals = np.abs(d - r)

            inliers = residuals < threshold
            num_inliers = np.sum(inliers)

            # Update best model
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_inliers = inliers
                best_center = center

        return best_center, best_inliers, points


class AimingSystem(LeafSystem):
    """
    Calculates end-effector transform and spatial velocity to aim at a predicted ball trajectory. Solve it as optimization problem.
    inputs:
        - predicted_trajectory (PiecewisePolynomial): Predicted trajectory of the ball
    outputs:
        - end_effector_transform (RigidTransform): Desired end-effector transform to aim at the ball
        - end_effector_spatial_velocity (SpatialVelocity): Desired end-effector spatial velocity to aim at the ball
    parameters:
        - desirecd_direction (float): Desired direction angle in radians
        - end_effector_offset_range (tuple(float, float)): Offset range of the end-effector from the aiming point
        - aiming_distance_range (tuple(float, float)): Distance range of the ball to be aimed at
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        ee_frame: Frame,
        desired_direction: float = 0.0,
        end_effector_offset_range: tuple = (0.50, 0.60),
        aiming_distance_range: tuple = (0.50, 2.0),
        meshcat: Optional[Meshcat] = None,
    ):
        super().__init__()
        self.desired_direction = desired_direction
        self.end_effector_offset_range = end_effector_offset_range
        self.aiming_distance_range = aiming_distance_range
        self.plant = plant
        self.ee_frame = ee_frame
        self.meshcat = meshcat

        self._plant_context = plant.CreateDefaultContext()

        self.transform = self.DeclareAbstractState(AbstractValue.Make(RigidTransform()))
        self.spatial_velocity = self.DeclareAbstractState(
            AbstractValue.Make(SpatialVelocity())
        )
        self.DeclareAbstractInputPort(
            "predicted_trajectory", AbstractValue.Make(PiecewisePolynomial())
        )
        self.DeclareVectorInputPort("iiwa_state", BasicVector(14))
        self.DeclareAbstractOutputPort(
            "end_effector_transform",
            lambda: AbstractValue.Make(RigidTransform()),
            self.OutputEndEffectorTransform,
        )
        self.DeclareAbstractOutputPort(
            "end_effector_spatial_velocity",
            lambda: AbstractValue.Make(SpatialVelocity()),
            self.OutputEndEffectorSpatialVelocity,
        )
        self.DeclarePeriodicPublishEvent(0.05, 0.0, self.CalcInverseKinematics)

    def OutputEndEffectorTransform(
        self, context: Context, output: AbstractValue
    ) -> None:
        transform = context.get_abstract_state(self.transform).get_value()
        output.set_value(transform)

    def OutputEndEffectorSpatialVelocity(
        self, context: Context, output: AbstractValue
    ) -> None:
        spatial_velocity = context.get_abstract_state(self.spatial_velocity).get_value()
        output.set_value(spatial_velocity)

    def CalcInverseKinematics(self, context: Context) -> None:
        pred_traj = self.GetInputPort("predicted_trajectory").Eval(context)

        try:
            if type(pred_traj) is not PiecewisePolynomial:
                pred_traj = pred_traj.get_value()
        except Exception as e:
            print(f"Error retrieving predicted trajectory: {e}")
            return

        try:
            if pred_traj.get_number_of_segments() == 0:
                print("No predicted trajectory available.")
                return
        except Exception as e:
            print(f"Error checking predicted trajectory: {e}")
            return

        iiwa_state = self.GetInputPort("iiwa_state").Eval(context)
        num_q = self.plant.num_positions()
        q_current = iiwa_state[:num_q]

        t_horizon = 4.0
        N_samples = 200

        t_now = context.get_time()
        t_values = np.linspace(t_now, t_now + t_horizon, num=N_samples)
        positions = np.array([pred_traj.value(t).flatten() for t in t_values])
        distances = np.linalg.norm(positions[:, :2], axis=1)

        r_min, r_max = self.aiming_distance_range
        # distance_mask = (distances >= r_min) & (distances <= r_max)
        distance_mask = np.ones_like(
            distances, dtype=bool
        )  # for debug, aim at all distances

        thetas = np.arctan2(positions[:, 1], positions[:, 0])

        candidate_indices = np.where(distance_mask)[0]

        if len(candidate_indices) == 0:
            print("No valid aiming point found in the predicted trajectory.")
            return

        # ---------- 打点候補の「コスト」（方向の近さ） ----------
        def angle_wrap(delta: float) -> float:
            return np.arctan2(np.sin(delta), np.cos(delta))

        theta_des = self.desired_direction

        def cost_index(i: int) -> float:
            # 位置の方位角
            theta_pos = thetas[i]
            dtheta_pos = angle_wrap(theta_pos - theta_des)

            # 速度の方位角（水平成分）
            dt = 1e-3
            t = t_values[i]
            t_before = max(t - dt, t_values[0])
            t_after = min(t + dt, t_values[-1])
            p_before = pred_traj.value(t_before).flatten()
            p_after = pred_traj.value(t_after).flatten()
            v = (p_after - p_before) / (t_after - t_before)
            v_xy = v[:2]
            if np.linalg.norm(v_xy) < 1e-6:
                theta_vel = theta_pos
            else:
                theta_vel = np.arctan2(v_xy[1], v_xy[0])
            dtheta_vel = angle_wrap(theta_vel - theta_des)

            w_pos = 0.2
            w_vel = 1.0
            return w_pos * dtheta_pos**2 + w_vel * dtheta_vel**2

        # コストが小さい順に候補を並べる
        sorted_candidates = sorted(candidate_indices, key=cost_index)

        # ---------- 各候補に対して IK を試す ----------
        feasible_found = False
        X_WE_best = None
        V_WE_best = None

        world = self.plant.world_frame()

        for idx in sorted_candidates:
            t_hit = t_values[idx]
            p_hit = positions[idx]  # shape (3,)

            # ボール速度（有限差分）
            dt = 1e-3
            t_before = max(t_hit - dt, t_values[0])
            t_after = min(t_hit + dt, t_values[-1])
            p_before = pred_traj.value(t_before).flatten()
            p_after = pred_traj.value(t_after).flatten()
            v_hit = (p_after - p_before) / (t_after - t_before)

            # バット軸方向（EE の x軸）: desired_direction に沿う水平ベクトル
            dir_xy = np.array([np.cos(theta_des), np.sin(theta_des), 0.0])
            dir_xy = dir_xy / np.linalg.norm(dir_xy)

            # バット上の打点位置 s（グリップからの距離）を決める
            s_min, s_max = self.end_effector_offset_range
            # EE 原点を原点に近づける s* = dir_xy·p_hit をクリップ
            s_star = float(np.dot(dir_xy, p_hit))
            s_opt = float(np.clip(s_star, s_min, s_max))

            # EE 原点位置
            p_ee = p_hit - s_opt * dir_xy

            # EE 姿勢：x軸=dir_xy, z軸=上、y=z×x
            z_axis = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(dir_xy, z_axis)) > 0.99:
                z_axis = np.array([0.0, 1.0, 0.0])

            x_axis = dir_xy / np.linalg.norm(dir_xy)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis)

            R_WE = RotationMatrix(np.column_stack((x_axis, y_axis, z_axis)))
            X_WE_desired = RigidTransform(R_WE, p_ee)

            # ---- IK セットアップ ----
            plant_context = self.plant.CreateDefaultContext()
            if len(q_current) < self.plant.num_positions():
                q_current = np.concatenate(
                    [q_current, np.zeros(self.plant.num_positions() - len(q_current))]
                )
            self.plant.SetPositions(plant_context, q_current)

            ik = InverseKinematics(self.plant, plant_context, with_joint_limits=True)
            prog = ik.get_mutable_prog()
            q = ik.q()

            # 位置制約: EE 原点が p_ee 近傍
            pos_tol = 1e-3  # 許容誤差 [m]
            ik.AddPositionConstraint(
                frameB=self.ee_frame,
                p_BQ=np.zeros(3),
                frameA=world,
                p_AQ_lower=p_ee - pos_tol,
                p_AQ_upper=p_ee + pos_tol,
            )

            # 姿勢制約: EE 回転が R_WE に近い
            theta_tol = 5.0 * np.pi / 180.0  # 5度

            ik.AddOrientationConstraint(
                world,  # frameAbar
                RotationMatrix(),  # R_AbarA
                self.ee_frame,  # frameBbar
                R_WE,  # R_BbarB
                theta_tol,  # theta_bound
            )

            # 初期値に現在姿勢をセット
            prog.SetInitialGuess(q, q_current)

            result = Solve(prog)
            if not result.is_success():
                # この打点候補は IIWA では実現できない → 次の候補へ
                # print(
                #     f"IK not feasible for candidate at t={t_hit:.2f}s, p_hit={p_hit}, s_opt={s_opt:.2f}m"
                # )
                continue

            # IK feasible な姿勢を採用
            q_sol = result.GetSolution(q)
            # 採用する EE 姿勢（実際にはq_solから再度X_WEを計算してもよい）
            X_WE_best = X_WE_desired

            # EE 空間速度：迎え角をボール速度に合わせる
            v_norm = np.linalg.norm(v_hit)
            if v_norm < 1e-6:
                v_dir = np.zeros(3)
            else:
                v_dir = v_hit / v_norm

            bat_speed_scale = 1.0  # 必要なら調整
            v_E = bat_speed_scale * v_dir
            w_E = np.zeros(3)
            V_WE_best = SpatialVelocity(w_E, v_E)

            feasible_found = True
            break

        if not feasible_found:
            print("No IK-feasible aiming point found.")
            return
        if feasible_found:
            print(
                f"Aiming at t={t_hit:.2f}s, p_hit={p_hit}, s_opt={s_opt:.2f}m"
            )
            self.meshcat.Delete(f"{str(self)}Aiming EE")
            self.meshcat.SetObject(
                f"{str(self)}Aiming EE", shape=Sphere(0.05), rgba=Rgba(0, 1, 0, 0.5)
            )
            self.meshcat.SetTransform(
                f"{str(self)}Aiming EE", X_WE_best
            )
            self.meshcat.Delete(f"{str(self)}Aiming velocity EE")
            self.meshcat.SetObject(
                f"{str(self)}Aiming velocity EE", shape=Cylinder(0.01, 0.2), rgba=Rgba(1, 0, 0, 0.8)
            )
            v = V_WE_best.translational()
            v_norm = np.linalg.norm(v)

            if v_norm < 1e-9:
                # fallback: identity (or skip drawing)
                R = RotationMatrix()
            else:
                z = v / v_norm  # normalized direction → will become the +Z axis

                # choose an arbitrary vector that is not parallel to z
                arbitrary = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 1., 0.])

                # create an orthonormal basis
                x = np.cross(arbitrary, z)
                x /= np.linalg.norm(x)

                y = np.cross(z, x)

                R = RotationMatrix(np.column_stack((x, y, z))) 


            V_arrow_transform = RigidTransform(
                R,
                X_WE_best.translation(),
            )
            self.meshcat.SetTransform(
                f"{str(self)}Aiming velocity EE", V_arrow_transform
            )

        context.SetAbstractState(self.transform, X_WE_best)
        context.SetAbstractState(self.spatial_velocity, V_WE_best)
