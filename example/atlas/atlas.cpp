#include <memory>
#include <utility>

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/visualization/visualization_config_functions.h"

static const double kDefaultSimulationTime = 20.0;
static const double kDefaultPenetrationAllowance = 1.0E-3;
static const double kDefaultStictionTolerance = 1.0E-3;
static const double kDefaultMbpDiscreteUpdatePeriod = 0.01;
static const char kDefaultContactApproximation[] = "sap";

namespace drake {
namespace examples {
namespace atlas {
namespace {

using drake::math::RigidTransformd;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using Eigen::Translation3d;
using Eigen::VectorXd;

int do_main(int argc, char* argv[]) {
    double simulation_time = kDefaultSimulationTime;
    if (argc > 1) {
        try {
            simulation_time = std::stod(argv[1]);
        } catch (...) {
            simulation_time = kDefaultSimulationTime;
        }
    }

    const double penetration_allowance = kDefaultPenetrationAllowance;
    const double stiction_tolerance = kDefaultStictionTolerance;
    const double mbp_discrete_update_period = kDefaultMbpDiscreteUpdatePeriod;
    const std::string contact_approximation = kDefaultContactApproximation;

    if (mbp_discrete_update_period < 0) {
        throw std::runtime_error(
                "mbp_discrete_update_period must be a non-negative number.");
    }

  // Build a generic multibody plant.
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
    plant_config.time_step = mbp_discrete_update_period;
    plant_config.stiction_tolerance = stiction_tolerance;
    plant_config.discrete_contact_approximation = contact_approximation;
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlant(plant_config, &builder);

  multibody::Parser(&builder).AddModelsFromUrl(
      "package://drake_models/atlas/atlas_convex_hull.urdf");

  // Add model of the ground.
  const double static_friction = 1.0;
  const Vector4<double> green(0.5, 1.0, 0.5, 1.0);
  plant.RegisterVisualGeometry(plant.world_body(), RigidTransformd(),
                               geometry::HalfSpace(), "GroundVisualGeometry",
                               green);
  // For a time-stepping model only static friction is used.
  const multibody::CoulombFriction<double> ground_friction(static_friction,
                                                           static_friction);
  plant.RegisterCollisionGeometry(plant.world_body(), RigidTransformd(),
                                  geometry::HalfSpace(),
                                  "GroundCollisionGeometry", ground_friction);

  plant.Finalize();
    plant.set_penetration_allowance(penetration_allowance);

  // Set the speed tolerance (m/s) for the underlying Stribeck friction model
  // For two points in contact, this is the maximum allowable drift speed at the
  // edge of the friction cone, an approximation to true stiction.
    plant.set_stiction_tolerance(stiction_tolerance);

  // Sanity check model size.
  DRAKE_DEMAND(plant.num_velocities() == 36);
  DRAKE_DEMAND(plant.num_positions() == 37);

  // Verify the "pelvis" body is a floating base body and modeled with
  // quaternions dofs before moving on with that assumption.
  const drake::multibody::RigidBody<double>& pelvis =
      plant.GetBodyByName("pelvis");
  DRAKE_DEMAND(pelvis.is_floating_base_body());
  DRAKE_DEMAND(pelvis.has_quaternion_dofs());
  // Since there is a single floating body, we know that the positions for it
  // lie first in the state vector.
  DRAKE_DEMAND(pelvis.floating_positions_start() == 0);
  // Similarly for velocities. The velocities for this floating pelvis are the
  // first set of velocities and should start at the beginning of v.
  DRAKE_DEMAND(pelvis.floating_velocities_start_in_v() == 0);

  visualization::AddDefaultVisualization(&builder);

  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  const VectorXd tau = VectorXd::Zero(plant.num_actuated_dofs());
  plant.get_actuation_input_port().FixValue(&plant_context, tau);

  // Set the pelvis frame P initial pose.
  const Translation3d X_WP(0.0, 0.0, 0.95);
  plant.SetFloatingBaseBodyPoseInWorldFrame(&plant_context, pelvis, X_WP);

  auto simulator = std::make_unique<systems::Simulator<double>>(
      *diagram, std::move(diagram_context));
  simulator->set_target_realtime_rate(1.0);
  simulator->AdvanceTo(simulation_time);

  return 0;
}

}  // namespace
}  // namespace atlas
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::examples::atlas::do_main(argc, argv);
}