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

namespace drake{
namespace examples {
namespace test {
int do_main(int argc, char* argv[]) {
    const double simulation_time = DefaultSimulationTime;
    const double penetration_allowance = DefaultPenetrationAllowance;
    const double stiction_tolerance = DefaultStictionTolerance;
    const double mbp_discrete_update_period = DefaultMbpDiscreteUpdatePeriod;
    const std::string contact_approximation = DefaultContactApproximation;

    systems::DiagramBuilder<double> builder;
    MultibodyPlantConfig plant_config;
    plant_config.time_step = mbp_discrete_update_period;
    plant_config.stiction_tolerance = stiction_tolerance;
    plant_config.penetration_allowance = penetration_allowance;
    plant_config.discrete_contact_approximation = contact_approximation;
    auto [plant, scene_graph] = multibody::AddMultibodyPlant(plant_config, &builder);

    multibody::Parser(&plant).AddModelsFromUrl(
        "package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf"
    );

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

    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"));

    plant.Finalize();
    plant.set_penetration_allowance(penetration_allowance);
    std::cout << "Number of bodies: " << plant.num_bodies() << std::endl;


    // Explicitly attach a Meshcat visualizer so we know a Meshcat server is
    // created and can print the URL for debugging. Some environments may not
    // configure the default visualizer to use Meshcat.
    auto meshcat = std::make_shared<drake::geometry::Meshcat>();
    drake::geometry::MeshcatVisualizerd::AddToBuilder(&builder, scene_graph,
                                                    meshcat);
    std::cout << "MeshCat web URL: " << meshcat->web_url() << std::endl;

    auto diagram = builder.Build();

    std::unique_ptr<systems::Context<double>> diagram_context = diagram->CreateDefaultContext();
    systems::Context<double>& plant_context = diagram->GetMutableSubsystemContext(plant, diagram_context.get());

    const VectorXd tau = VectorXd::Zero(plant.num_actuated_dofs());
    plant.get_actuation_input_port().FixValue(&plant_context, tau);

    auto simulator = std::make_unique<systems::Simulator<double>>(
        *diagram, std::move(diagram_context)
    );
    simulator->set_target_realtime_rate(1.0);
    simulator->AdvanceTo(simulation_time);

    return 0;
}
}
}
}

int main(int argc, char* argv[]) {
    return drake::examples::test::do_main(argc, argv);
}