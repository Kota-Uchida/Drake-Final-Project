import os
import sys

from pydrake.all import (AddMultibodyPlantSceneGraph, Context, Diagram,
                         DiagramBuilder, MeshcatVisualizer, Parser, Simulator,
                         StartMeshcat)

assets_path = os.path.join("../../assets/")
ball_path = os.path.join(assets_path, "baseball", "baseball.sdf")
bat_path = os.path.join(assets_path, "bat", "bat.sdf")

meshcat = StartMeshcat()
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
parser = Parser(plant, scene_graph)

iiwa = parser.AddModelsFromUrl(
    "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)[0]
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

ball = parser.AddModels(ball_path)
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("baseball_link"))

bat = parser.AddModels(bat_path)
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("bat_link"))

visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

plant.Finalize()

diagram = builder.Build()

context = diagram.CreateDefaultContext()

simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(10.0)
