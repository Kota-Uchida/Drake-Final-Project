#include <string>
#include <iostream>

#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_visualizer.h"

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/visualization/visualization_config_functions.h"

static const double DefaultSimulationTime = 20.0;
static const double DefaultPenetrationAllowance = 1.0E-3;
static const double DefaultStictionTolerance = 1.0E-3;
static const double DefaultMbpDiscreteUpdatePeriod = 0.01;
static const char DefaultContactApproximation[] = "sap";

using drake::math::RigidTransformd;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using Eigen::Translation3d;
using Eigen::VectorXd;


/* It is better to use appropriate namespace if you want to align to the convention of drake repository */
namespace drake{
namespace examples {
namespace test {
int do_main(int argc, char* argv[]) {
    // Default configurations
    const double simulation_time = DefaultSimulationTime;
    const double penetration_allowance = DefaultPenetrationAllowance;
    const double stiction_tolerance = DefaultStictionTolerance;
    const double mbp_discrete_update_period = DefaultMbpDiscreteUpdatePeriod;
    const std::string contact_approximation = DefaultContactApproximation;

    // Define Builder
    systems::DiagramBuilder<double> builder;

    // Set plant configurations
    MultibodyPlantConfig plant_config;
    plant_config.time_step = mbp_discrete_update_period;
    plant_config.stiction_tolerance = stiction_tolerance;
    plant_config.penetration_allowance = penetration_allowance;
    plant_config.discrete_contact_approximation = contact_approximation;

    // Get plant and scene graph with AddMultibodyPlant
    auto [plant, scene_graph] = multibody::AddMultibodyPlant(plant_config, &builder);

    // Use AddModelsFromUrl to add iiwa model
    multibody::Parser(&plant).AddModelsFromUrl(
        "package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf"
    );

    // Add simulation environment configurations
    const double static_friction = 0.9;
    const Vector4<double> green(0.5, 1.0, 0.5, 1.0);
    plant.RegisterVisualGeometry(plant.world_body(), RigidTransformd(),
                                 geometry::HalfSpace(), "GroundVisualGeometry",
                                 green
                                 );
    const multibody::CoulombFriction<double> ground_friction(static_friction,
                                                           static_friction);
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransformd(),
                                    geometry::HalfSpace(),
                                    "GroundCollisionGeometry", ground_friction);

    // Don't forget to weld the base of the iiwa, otherwise it will fall
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"));

    // Finalize plant after all configurations
    plant.Finalize();
    plant.set_penetration_allowance(penetration_allowance);

    // Call meshcat visualizer
    // Use shared pointer because meshcat visualizer can be shared with other systems
    auto meshcat = std::make_shared<drake::geometry::Meshcat>();
    drake::geometry::MeshcatVisualizerd::AddToBuilder(&builder, scene_graph,
                                                    meshcat);
    std::cout << "MeshCat web URL: " << meshcat->web_url() << std::endl;

    // Build the diagram
    auto diagram = builder.Build();

    // Create the default context with CreateDefaultContext
    // Use unique pointer because the default context can be reused by other systems
    std::unique_ptr<systems::Context<double>> diagram_context = diagram->CreateDefaultContext();

    // To set the context configuration, get mutable context from plant
    systems::Context<double>& plant_context = diagram->GetMutableSubsystemContext(plant, diagram_context.get());
    // Set zero actuation input
    const VectorXd tau = VectorXd::Zero(plant.num_actuated_dofs());
    plant.get_actuation_input_port().FixValue(&plant_context, tau);

    // Create simulator
    auto simulator = std::make_unique<systems::Simulator<double>>(
        *diagram, std::move(diagram_context)
    );
    // Visualize in real time
    simulator->set_target_realtime_rate(1.0);
    // Advance simulation
    simulator->AdvanceTo(simulation_time);

    return 0;
}
}
}
}

int main(int argc, char* argv[]) {
    return drake::examples::test::do_main(argc, argv);
}