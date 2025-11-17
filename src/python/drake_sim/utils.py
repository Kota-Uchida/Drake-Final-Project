import numpy as np
import pydot
import matplotlib.pyplot as plt
from PIL import Image
import io

from pydrake.systems.framework import System
from pydrake.all import (
    MultibodyPlant,
    Context,
    RigidTransform,
    RotationMatrix,
    SpatialVelocity,
)

def LoadScenarioFromFile(scenario_path: str):
    # Placeholder for loading scenario from file
    pass

def WriteBallObject(ball_path: str, radius: float, mass: float):
    # Placeholder for writing ball SDF file and OBJ file
    pass

def WriteBatObject(bat_path: str, length: float, radius: float, mass: float):
    # Placeholder for writing bat SDF file and OBJ file
    pass

def RenderDiagramPNG(system: System, max_depth: int | None = None):
    """
    Render the GraphViz diagram of `system` as a PNG and display with matplotlib.
    Args:
        system (System): The Drake system (or diagram) to render.
        max_depth (int, optional): Limit the depth of nested diagrams. Use 0 to
          render the diagram as a single block. Defaults to 1.
    """
    dot = system.GetGraphvizString(max_depth=max_depth)
    graphs = pydot.graph_from_dot_data(dot)
    if not graphs:
        raise RuntimeError("pydot failed to create graph from dot data")
    png_bytes = graphs[0].create_png()
    if png_bytes is None:
        raise RuntimeError("pydot failed to create PNG output")
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    plt.figure(figsize=(min(12, img.width / 100), min(12, img.height / 100)))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def ThrowObject(plant: MultibodyPlant, plant_context: Context, object_name: str, initial_position: np.ndarray, initial_rotation: RotationMatrix, initial_velocity: np.ndarray, initial_angular_velocity: np.ndarray, ):
    """
    Set the initial position and velocity of an object in the MultibodyPlant.
    This function takes arguments of the initial state.
    Refer to ThrowObjectTowardsTarget for a higher-level function that computes these values.
    Args:
        plant (MultibodyPlant): The MultibodyPlant containing the object.
        plant_context (Context): The context of the MultibodyPlant.
        object_name (str): The name of the object to be thrown.
        initial_position (np.ndarray): The initial position of the object as a 3D vector.
        initial_rotation (RotationMatrix): The initial rotation of the object.
        initial_velocity (np.ndarray): The initial linear velocity of the object as a 3D vector.
        initial_angular_velocity (np.ndarray): The initial angular velocity of the object as a 3D vector.
    """
    body = plant.GetBodyByName(object_name)
    initial_pose = RigidTransform(initial_rotation, initial_position)
    plant.SetFreeBodyPose(plant_context, body, initial_pose)

    spatial_velocity = SpatialVelocity(
        initial_angular_velocity,
        initial_velocity
    )
    plant.SetFreeBodySpatialVelocity(plant_context, body, spatial_velocity)

def ComputeThrowParameters(start_position: np.ndarray, target_position: np.ndarray, target_speed_xy: float, gravity: float = 9.81):
    """
    Compute the initial velocity and angular velocity required to throw an object from start_position to target_position.
    Takes into account the gravitational acceleration.
    Args:
        start_position (np.ndarray): The starting position of the object as a 3D vector.
        target_position (np.ndarray): The target position of the object as a 3D vector.
        target_speed (float): The speed at which to throw the object.
        gravity (float): The gravitational acceleration (positive value).
    Returns:
        initial_velocity (np.ndarray): The computed initial linear velocity as a 3D vector.
        initial_angular_velocity (np.ndarray): The computed initial angular velocity as a 3D vector.
    """
    delta_pos = target_position - start_position
    horizontal_distance = np.linalg.norm(delta_pos[:2])
    vertical_distance = delta_pos[2]
    T = horizontal_distance / target_speed_xy
    initial_velocity_xy = (delta_pos[:2] / horizontal_distance) * target_speed_xy
    initial_velocity_z = (vertical_distance + 0.5 * gravity * T**2) / T
    initial_velocity = np.array([initial_velocity_xy[0], initial_velocity_xy[1], initial_velocity_z])
    initial_angular_velocity = np.array([0.0, 0.0, 0.0])
    return initial_velocity, initial_angular_velocity

def ThrowObjectTowardsTarget(plant: MultibodyPlant, plant_context: Context, object_name: str, initial_position: np.ndarray, target_position: np.ndarray, target_speed_xy: float, initial_rotation: RotationMatrix = RotationMatrix.Identity()):
    """
    Set the initial position and velocity of an object in the MultibodyPlant to throw it towards a target position.
    This function computes the required initial velocity and angular velocity based on the desired throw speed.
    Args:
        plant (MultibodyPlant): The MultibodyPlant containing the object.
        plant_context (Context): The context of the MultibodyPlant.
        object_name (str): The name of the object to be thrown.
        initial_position (np.ndarray): The initial position of the object as a 3D vector.
        target_position (np.ndarray): The target position to throw the object towards as a 3D vector.
        throw_speed (float): The speed at which to throw the object.
        initial_rotation (RotationMatrix, optional): The initial rotation of the object. Defaults to identity rotation.
    """
    initial_velocity, initial_angular_velocity = ComputeThrowParameters(
        initial_position,
        target_position,
        target_speed_xy,
    )
    print("Initial velocity:", initial_velocity)
    print("Initial angular velocity:", initial_angular_velocity)
    ThrowObject(
        plant,
        plant_context,
        object_name,
        initial_position,
        initial_rotation,
        initial_velocity,
        initial_angular_velocity
    )



